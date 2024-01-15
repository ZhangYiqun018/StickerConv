import logging
import contextlib
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn.utils import rnn
from transformers import (LlamaConfig, LlamaTokenizer, LlamaForCausalLM,
                          Blip2Config, Blip2ForConditionalGeneration,
                          CLIPTextModel,
                          BitsAndBytesConfig, StoppingCriteriaList)
from transformers.modeling_outputs import CausalLMOutput
from peft import LoraConfig, get_peft_model, TaskType
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL

from pegs.model.projection_layers import InputProjectionLayer, OutputProjectionLayer
from pegs.model.utils import StoppingCriteriaSub, disabled_train, convert_weights_to_fp16
from pegs.model.outputs import PegsOutput, PegsGenerationOutput
from pegs.register import registry


@registry.register_model("pegs")
class PEGS(nn.Module):
    def __init__(
        self,
        enable_perception: bool = False,
        enable_generation: bool = False,
        llama_pretrained_model_path_or_name: str = None,
        use_lora: bool = False,
        # perception
        blip2_pretrained_model_path_or_name: str = None,
        freeze_vision_model: bool = True,
        freeze_qformer: bool = True,
        vision_precision: str = "fp16",
        # generation
        stable_diffusion_pretrained_model_path_or_name: str = None,
        freeze_stable_diffusion: bool = True,
        num_clip_tokens: int = 77,
        prompt_embeddings_dim: int = 768,
        # other settings
        padding_side: str = "right",
        max_text_length: int = None,
        image_prefix_token: str = "<Img>",
        image_postfix_token: str = "</Img>",
        image_placeholder_token: str = "<IMG>",
        num_image_tokens: int = 32,
        use_prefix_prompt: bool = False,
        prefix_prompt: Optional[str] = None,
        low_resource: bool = False,  # use 8 bit
        device_8bit: int = 0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.enable_perception: bool = enable_perception
        self.enable_generation: bool = enable_generation
        
        self.low_resource: bool = low_resource
        self.device_8bit: int = device_8bit
        
        if enable_perception:
            logging.info("Loading Vision Encoder and Q-Former...")
            self.blip2_config, self.vision_model, self.qformer, self.query_tokens = self._init_from_blip2(
                blip2_pretrained_model_path_or_name,
            )
                
            if vision_precision == "fp16":
                convert_weights_to_fp16(self.vision_model)
        
            if freeze_vision_model:
                for _, param in self.vision_model.named_parameters():
                    param.requires_grad = False
                self.vision_model.eval()
                self.vision_model.train = disabled_train
                logging.info("Freeze Vision Encoder.")
            logging.info("Vision Encoder has been loaded.")
        
            self.qformer_config = self.blip2_config.qformer_config
            if freeze_qformer:
                for _, param in self.qformer.named_parameters():
                    param.requires_grad = False
                self.qformer.eval()
                self.qformer.train = disabled_train
                self.query_tokens.requires_grad = False
                logging.info("Freeze Q-Former.")
            logging.info("Q-Former has been loaded.")
        else:
            self.blip2_config = Blip2Config.from_pretrained(blip2_pretrained_model_path_or_name)
            self.qformer_config = self.blip2_config.qformer_config 
        
        logging.info("Loading LLM...")
        self.llm_config, self.tokenizer, self.llm_model = self._init_from_llama(
            llama_pretrained_model_path_or_name, device_8bit
        )

        # tokenizer settings
        self.tokenizer.padding_side = padding_side
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if max_text_length is None:
            logging.warn("max_text_length is None.")
        self.max_text_length = max_text_length if max_text_length is not None else 512
        
        # image tokens
        self.image_prefix_token = image_prefix_token
        self.image_postfix_token = image_postfix_token
        self.image_placeholder_token = image_placeholder_token  # not as a special token. refer to the image
        self._add_image_tokens(num_image_tokens)
        
        if enable_generation:
            self.num_image_tokens = num_image_tokens
            with torch.no_grad():
                self.input_embeds_grad_mask = torch.ones_like(self.llm_model.get_input_embeddings().weight.data)
                self.output_embeds_grad_mask = torch.ones_like(self.llm_model.get_output_embeddings().weight.data)
                self.input_embeds_grad_mask[:-self.num_image_tokens] = 0
                self.output_embeds_grad_mask[:-self.num_image_tokens] = 0
        
        self.prefix_prompt = prefix_prompt if use_prefix_prompt else None
        
        self.use_lora = use_lora
        if self.use_lora:
            logging.info("Use LoRA")
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            lora_r = 8
            lora_alpha = 16
            lora_dropout = 0.05
            self.lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=['lm_head','embed_tokens']
            )
            self.llm_model = get_peft_model(self.llm_model, self.lora_config)
            self.llm_model.base_model.model.model.embed_tokens.original_module.weight.requires_grad = False
            self.llm_model.base_model.model.lm_head.original_module.weight.requires_grad = False
        else:
            for name, param in self.llm_model.named_parameters():
                if enable_generation and "embed_tokens" in name:
                    param.requires_grad = True
                elif enable_perception and enable_generation and "lm_head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            logging.info("Freeze LLM, except embedding layers.")
        logging.info("LLM has been loaded.")
        
        if enable_generation:
            text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_pretrained_model_path_or_name, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(stable_diffusion_pretrained_model_path_or_name, subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained(stable_diffusion_pretrained_model_path_or_name, subfolder="unet")
            if freeze_stable_diffusion:
                text_encoder.requires_grad_(False)
                unet.requires_grad_(False)
                vae.requires_grad_(False)
                logging.info("Freeze Stable Diffusion.")
            self.stable_diffusion = StableDiffusionPipeline.from_pretrained(
                stable_diffusion_pretrained_model_path_or_name,
                text_encoder=text_encoder, unet=unet, vae=vae,
            )
            logging.info(f"type(self.stable_diffusion) = {type(self.stable_diffusion)}")
            logging.info("Stable Diffusion has been loaded.")

        # projection layer
        if enable_perception:
            self.inputProjection = InputProjectionLayer(self.qformer_config.hidden_size, self.llm_config.hidden_size)
        if enable_generation:
            self.outputProjection = OutputProjectionLayer(self.llm_config.hidden_size, num_clip_tokens, prompt_embeddings_dim)
        
    def _init_from_blip2(
        self, 
        blip2_pretrained_model_path_or_name: str,
    ):
        config = Blip2Config.from_pretrained(blip2_pretrained_model_path_or_name)

        blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            blip2_pretrained_model_path_or_name
        )
        vision_model = blip2_model.vision_model
        qformer = blip2_model.qformer
        query_tokens = blip2_model.query_tokens  # [1, 32, 768]
        del blip2_model
        
        return config, vision_model, qformer, query_tokens

    def _init_from_llama(
        self, 
        llama_pretrained_model_path_or_name: str,
        device_8bit: Optional[int] = None
    ) -> (LlamaConfig, LlamaTokenizer, LlamaForCausalLM):
        config = LlamaConfig.from_pretrained(llama_pretrained_model_path_or_name)
        tokenizer = LlamaTokenizer.from_pretrained(llama_pretrained_model_path_or_name)
        if self.low_resource:
            quantization_config = BitsAndBytesConfig(llm_int8_threshold=4.0)
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_pretrained_model_path_or_name,
                torch_dtype=torch.float16,
                device_map={'': device_8bit},
                load_in_8bit=True,
                quantization_config=quantization_config,
            )
        else:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_pretrained_model_path_or_name,
                torch_dtype=torch.float16,
            )
        
        return config, tokenizer, llama_model

    def _add_image_tokens(self, num_image_tokens: int = 0):
        special_tokens = [self.image_prefix_token, self.image_postfix_token]
        logging.info(f"\nadd special tokens: {special_tokens}")
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        logging.info(f"\nspecial_tokens_ids: {self.tokenizer.convert_tokens_to_ids(special_tokens)}")
        
        if self.enable_generation:
            self.image_tokens = [f"[IMG{i+1}]" for i in range(num_image_tokens)]
            logging.info(f"\nadd image tokens: {self.image_tokens}")
            for image_token in self.image_tokens:
                self.tokenizer.add_tokens([image_token])
            self.image_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.image_tokens)
            logging.info(f"\nimage_tokens_ids: {self.image_tokens_ids}")
        
        # resize token embeddings
        self.llm_model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.save_pretrained("tokenizer")
        # embeddings
        with torch.no_grad():
            image_prefix_token_id = self.tokenizer(self.image_prefix_token, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.llm_model.device)
            self._image_prefix_embeds = self.llm_model.model.embed_tokens(image_prefix_token_id)
            image_postfix_token_id = self.tokenizer(self.image_postfix_token, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.llm_model.device)
            self._image_postfix_embeds = self.llm_model.model.embed_tokens(image_postfix_token_id)
            
            if self.enable_generation:
                self._image_tokens_input_ids = torch.tensor(self.image_tokens_ids).unsqueeze(0)  # [1, num_image_tokens]
                self._image_tokens_attention_mask = torch.ones(self._image_tokens_input_ids.size(), dtype=torch.long)  # [1, num_image_tokens]
                self._image_tokens_labels = torch.tensor(self.image_tokens_ids).unsqueeze(0)  # [1, num_image_tokens]

    @property
    def image_prefix_embeds(self):
        return self._image_prefix_embeds.to(self.llm_model.device)
    
    @property
    def image_postfix_embeds(self):
        return self._image_postfix_embeds.to(self.llm_model.device)
    
    @property
    def image_tokens_input_ids(self):
        return self._image_tokens_input_ids.to(self.llm_model.device)

    @property
    def image_tokens_embeds(self):
        return self.get_input_embeddings(self.image_tokens_input_ids)
    
    @property
    def image_tokens_attention_mask(self):
        return self._image_tokens_attention_mask.to(self.llm_model.device)
    
    @property
    def image_tokens_labels(self):
        return self._image_tokens_labels.to(self.llm_model.device)

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.llm_model.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def reset_embeddings(self):
        with torch.no_grad():
            if self.use_lora:
                for name, param in self.llm_model.named_parameters():
                    if param.grad is None:
                        continue
                    if "embed_tokens" in name:
                        param.grad = param.grad * self.input_embeds_grad_mask.to(param.device)
                    elif "lm_head" in name:
                        param.grad = param.grad * self.output_embeds_grad_mask.to(param.device)
            else:
                self.llm_model.get_input_embeddings().weight.grad = self.llm_model.get_input_embeddings().weight.grad * self.input_embeds_grad_mask.to(self.llm_model.device)
                if self.enable_perception:
                    self.llm_model.get_output_embeddings().weight.grad = self.llm_model.get_output_embeddings().weight.grad * self.output_embeds_grad_mask.to(self.llm_model.device)
        
    def image_encoding(self, pixel_values: torch.FloatTensor):
        device = self.vision_model.device
        
        with self.maybe_autocast():  # Required, otherwise an error "RuntimeError: expected scalar type Half but found Float" will be reported.
            pixel_values = pixel_values.to(self.vision_model.device)  # [b_s, 3, 224, 224]
        
            vision_outputs = self.vision_model(pixel_values)
        
            image_embeds = vision_outputs.last_hidden_state.to(device)  # [b_s, 257(=1+16*16), 1408]
            image_attention_mask = torch.ones(image_embeds.size()[:-1],
                                              dtype=torch.long,
                                              device=device)  # [b_s, 257]
        
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask
            )  # last_hidden_state, pooler_output, hidden_states, past_key_values, attentions, cross_attentions
            query_output = query_outputs.last_hidden_state  # [b_s, 32, 768]  # 32 is the number of query tokens
        
            image_content_embeds = self.inputProjection(query_output)  # [b_s, 32, 4096]
            image_prefix_embeds = self.image_prefix_embeds.expand(image_embeds.shape[0], -1, -1)  # [b_s, 1, 4096]
            image_postfix_embeds = self.image_postfix_embeds.expand(image_embeds.shape[0], -1, -1)  # [b_s, 1, 4096]
            inputs_embeds = torch.cat([image_prefix_embeds, image_content_embeds, image_postfix_embeds], dim=1)  # [b_s, 34(1+32+1), 4096]
            inputs_attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)  # [b_s, 34(1+32+1)]
            labels = inputs_attention_mask * (-100)
            
        return inputs_embeds, inputs_attention_mask, labels
    
    def get_input_embeddings(self, input_ids):
        if self.use_lora:
            embed_tokens = self.llm_model.base_model.model.model.embed_tokens
        else:
            embed_tokens = self.llm_model.model.embed_tokens
        inputs_embeds = embed_tokens(input_ids)

        return inputs_embeds
            
    def get_bos_encoding(self, batch_size):
        bos_input_ids = torch.ones([batch_size, 1], dtype=torch.int64, device=self.llm_model.device) * self.tokenizer.bos_token_id
        bos_embeds = self.get_input_embeddings(bos_input_ids)  # [b_s, 1, 4096]
        bos_attention_mask = torch.ones([batch_size, 1], dtype=torch.int64, device=self.llm_model.device)  # [b_s, 1]
        bos_labels = bos_attention_mask * (-100)
        
        if self.prefix_prompt is not None:
            prefix_prompt_encoded = self.tokenizer([self.prefix_prompt] * batch_size,
                                                   add_special_tokens=False,
                                                   return_tensors="pt").to(self.llm_model.device)
            prefix_prompt_embeds = self.get_input_embeddings(prefix_prompt_encoded["input_ids"])
            prefix_prompt_attention_mask = prefix_prompt_encoded["attention_mask"]
            prefix_prompt_labels = prefix_prompt_attention_mask * (-100)
            
            bos_input_ids = torch.cat([bos_input_ids, prefix_prompt_encoded["input_ids"]], dim=1)
            bos_embeds = torch.cat([bos_embeds, prefix_prompt_embeds], dim=1)
            bos_attention_mask = torch.cat([bos_attention_mask, prefix_prompt_attention_mask], dim=1)
            bos_labels = torch.cat([bos_labels, prefix_prompt_labels], dim=1)
        
        return bos_input_ids, bos_embeds, bos_attention_mask, bos_labels

    def text_encoding(self, text: str):
        text_tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_text_length,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.llm_model.device)
        input_ids, attention_mask = text_tokens["input_ids"], text_tokens["attention_mask"]
        
        text_embeds = self.get_input_embeddings(input_ids)  # [b_s, seq len, 4096]
        
        return input_ids, text_embeds, attention_mask
    
    def build_one_batch_perception(self, pixel_values: torch.FloatTensor, text: List[str]):
        image_embeds, image_attention_mask, image_labels = self.image_encoding(pixel_values)
        input_ids, text_embeds, text_attention_mask = self.text_encoding(text)
        
        # input
        batch_size = text_embeds.shape[0]
        _, bos_embeds, bos_attention_mask, bos_labels = self.get_bos_encoding(batch_size)
        
        input_embeds = torch.cat([bos_embeds, image_embeds, text_embeds], dim=1)
        input_attention_mask = torch.cat([bos_attention_mask, image_attention_mask, text_attention_mask], dim=1)

        # labels
        text_labels = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)  # [b_s, seq_len]
        labels = torch.cat([bos_labels, image_labels, text_labels], dim=1)  # [b_s, bos + <Img> + queries + </Img> + text]
        
        return None, input_embeds, input_attention_mask, labels

    def build_one_batch_generation(self, text: List[str], **kwargs):
        input_ids_list, input_embeds_list, input_attention_mask_list, labels_list = [], [], [], []
        for one_text in text:
            text_input_ids, text_embeds, text_attention_mask = self.text_encoding(one_text) # [1, seq_len, 4096], [1, seq_len], [1, seq_len]
            text_labels = text_attention_mask * (-100)  # [1, seq_len]
            bos_input_ids, bos_embeds, bos_attention_mask, bos_labels = self.get_bos_encoding(1)
            
            input_ids = torch.cat([bos_input_ids, text_input_ids, self.image_tokens_input_ids], dim=1)  # [1, 1+seq_len+32])
            input_embeds = torch.cat([bos_embeds, text_embeds, self.image_tokens_embeds], dim=1)  # [1, 1+seq_len+32, 4096])
            input_attention_mask = torch.cat([bos_attention_mask, text_attention_mask, self.image_tokens_attention_mask], dim=1)  # [1, 1+seq_len+32]
            labels = torch.cat([bos_labels, text_labels, self.image_tokens_labels], dim=1)  # [1, 1+seq_len+32]
            
            input_ids_list.append(input_ids.squeeze(0))
            input_embeds_list.append(input_embeds.squeeze(0))
            input_attention_mask_list.append(input_attention_mask.squeeze(0))
            labels_list.append(labels.squeeze(0))
        
        input_ids = rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)  # [b_s, max_seq_len]
        input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)  # [b_s, max_seq_len, 4096]
        input_attention_mask = rnn.pad_sequence(input_attention_mask_list, batch_first=True, padding_value=0)  # [b_s, max_seq_len]
        labels = rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)  # [b_s, max_seq_len]
        assert input_ids.shape == input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape
        
        return input_ids, input_embeds, input_attention_mask, labels
    
    def build_one_instance_perception_and_generation(self, pixel_values: torch.FloatTensor, text: str):
        text_list = text.split(self.image_placeholder_token)
        
        input_ids_list, input_embeds_list, input_attention_mask_list, labels_list = [], [], [], []
        for i, one_text in enumerate(text_list):
            if one_text != "":  # text
                one_input_ids, one_text_embeds, one_text_attention_mask = self.text_encoding(one_text)
                
                input_ids_list.append(one_input_ids)  # one_input_ids: [1, seq len]
                input_embeds_list.append(one_text_embeds)  # one_text_embeds: [1, seq len, 4096]
                input_attention_mask_list.append(one_text_attention_mask)  # one_text_attention_mask: [1, seq len]
                labels_list.append(one_input_ids.masked_fill(one_input_ids == self.tokenizer.pad_token_id, -100))
            if i != len(text_list) - 1:  # image
                if "Human" == one_text.split("### ")[-1][:5]:  # image for perception
                    one_image_embeds, one_image_attention_mask, one_image_labels = self.image_encoding(pixel_values[i].unsqueeze(0))
                    one_input_ids = torch.zeros(one_image_labels.size(), dtype=torch.long, device=self.llm_model.device)  # Just as a placeholder input_ids. Don't affect training
                    
                    input_ids_list.append(one_input_ids)  # one_input_ids: [1, 34(=1+32+1)]
                    input_embeds_list.append(one_image_embeds)  # one_image_embeds: [1, 34(=1+32+1), 4096]
                    input_attention_mask_list.append(one_image_attention_mask)  # one_image_attention_mask: [1, 34(=1+32+1)]
                    labels_list.append(one_image_labels)  # one_image_labels: [1, 34(=1+32+1)]
                elif "Assistant" == one_text.split("### ")[-1][:9]:   # image for generation
                    input_ids_list.append(self.image_tokens_input_ids)  # image_tokens_input_ids: [1, 32]
                    input_embeds_list.append(self.image_tokens_embeds)  # image_tokens_embeds: [1, 32, 4096]
                    input_attention_mask_list.append(self.image_tokens_attention_mask)  # image_tokens_attention_mask: [1, 32]
                    labels_list.append(self.image_tokens_labels)  # image_tokens_labels: [1, 32]
                else:
                    logging.warning(f"Training data is irregular!\n{one_text.split('###')[-1]}")

        bos_input_ids, bos_embeds, bos_attention_mask, bos_labels = self.get_bos_encoding(1)
        input_ids = torch.cat([bos_input_ids] + input_ids_list, dim=1)  # [1, seq len]
        input_embeds = torch.cat([bos_embeds] + input_embeds_list, dim=1)  #[1, seq len, 4096]
        input_attention_mask = torch.cat([bos_attention_mask] + input_attention_mask_list, dim=1)  # [1, seq len]
        labels = torch.cat([bos_labels] + labels_list, dim=1)  # [1, seq len]
        assert input_ids.shape == input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape
        
        return input_ids, input_embeds, input_attention_mask, labels
    
    def build_one_batch_perception_and_generation(self, pixel_values: torch.FloatTensor, text: List[str]):
        input_ids_list, input_embeds_list, input_attention_mask_list, labels_list = [], [], [], []
        for one_pixel_values, one_text in zip(pixel_values, text):
            one_input_ids, one_input_embeds, one_input_attention_mask, one_labels = self.build_one_instance_perception_and_generation(one_pixel_values, one_text)

            input_ids_list.append(one_input_ids.squeeze(0))
            input_embeds_list.append(one_input_embeds.squeeze(0))
            input_attention_mask_list.append(one_input_attention_mask.squeeze(0))
            labels_list.append(one_labels.squeeze(0))
        
        input_ids = rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)[:, :self.max_text_length]  # [b_s, min(seq len, max_text_length)]
        input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)[:, :self.max_text_length, :]  # [b_s, min(seq len, max_text_length), 4096]
        input_attention_mask = rnn.pad_sequence(input_attention_mask_list, batch_first=True, padding_value=0)[:, :self.max_text_length]  # [b_s, min(seq len, max_text_length)]
        labels = rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)[:, :self.max_text_length]  # [b_s, min(seq len, max_text_length)]
        assert input_ids.shape == input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape

        return input_ids, input_embeds, input_attention_mask, labels

    def build_one_instance_for_demo(self, pixel_values: torch.FloatTensor, text: str):
        text_list = text.split(self.image_placeholder_token)
        
        input_ids_list, input_embeds_list, input_attention_mask_list, labels_list = [], [], [], []
        for i, one_text in enumerate(text_list):
            if one_text != "":
                one_input_ids, one_text_embeds, one_text_attention_mask = self.text_encoding(one_text)
                
                input_ids_list.append(one_input_ids)  # one_input_ids: [1, seq len]
                input_embeds_list.append(one_text_embeds)  # one_text_embeds: [1, seq len, 4096]
                input_attention_mask_list.append(one_text_attention_mask)  # one_text_attention_mask: [1, seq len]
                labels_list.append(one_input_ids.masked_fill(one_input_ids == self.tokenizer.pad_token_id, -100))
            if i != len(text_list) - 1:
                one_image_embeds, one_image_attention_mask, one_image_labels = self.image_encoding(pixel_values[i].unsqueeze(0))
                one_input_ids = torch.zeros(one_image_labels.size(), dtype=torch.long, device=self.llm_model.device)  # Just as a placeholder input_ids. Don't affect training
                    
                input_ids_list.append(one_input_ids)  # one_input_ids: [1, 34(=1+32+1)]
                input_embeds_list.append(one_image_embeds)  # one_image_embeds: [1, 34(=1+32+1), 4096]
                input_attention_mask_list.append(one_image_attention_mask)  # one_image_attention_mask: [1, 34(=1+32+1)]
                labels_list.append(one_image_labels)  # one_image_labels: [1, 34(=1+32+1)]
                
        bos_input_ids, bos_embeds, bos_attention_mask, bos_labels = self.get_bos_encoding(1)
        
        input_ids = torch.cat([bos_input_ids] + input_ids_list, dim=1)  # [1, seq len]
        input_embeds = torch.cat([bos_embeds] + input_embeds_list, dim=1)  # [1, seq len, 4096]
        input_attention_mask = torch.cat([bos_attention_mask] + input_attention_mask_list, dim=1)  # [1, seq len]
        labels = torch.cat([bos_labels] + labels_list, dim=1)  # [1, seq len]
        assert input_ids.shape == input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape
        
        return input_ids, input_embeds, input_attention_mask, labels
        
    def forward(
        self,
        text: List[str],
        pixel_values: Optional[torch.FloatTensor] = None,
        caption: Optional[torch.FloatTensor] = None
    ) -> CausalLMOutput:
        if self.enable_perception and not self.enable_generation:
            build_batch_function = self.build_one_batch_perception
        elif not self.enable_perception and self.enable_generation:
            build_batch_function = self.build_one_batch_generation
        elif self.enable_perception and self.enable_generation:
            build_batch_function = self.build_one_batch_perception_and_generation
        
        input_ids, inputs_embeds, inputs_attention_mask, labels = build_batch_function(
            pixel_values=pixel_values, text=text
        )
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=labels
        )  # 'loss', 'logits', 'past_key_values', 'hidden_states'
        lm_loss = outputs.loss
        
        mse_loss = 0
        mse_loss_function = nn.MSELoss()
        if not self.enable_perception and self.enable_generation:  # train generation-only
            image_start_position = (labels == self.image_tokens_ids[0]).nonzero(as_tuple=False)[:, 1].tolist()
            image_end_position = (labels == self.image_tokens_ids[-1]).nonzero(as_tuple=False)[:, 1].tolist()
            
            hidden_states_list = []
            for b, (start, end) in enumerate(zip(image_start_position, image_end_position)):
                hidden_states_list.append(outputs.hidden_states[-1][b, start: end + 1, :])
            hidden_states = torch.stack(hidden_states_list, dim=0)  # [b_s, num_image_tokens, 4096]
            projected_hidden_states = self.outputProjection(hidden_states)  # [b_s, 77, 768]

            with torch.no_grad():
                sd_text_embeddings, _ = self.stable_diffusion.encode_prompt(
                    prompt=text,
                    do_classifier_free_guidance=False,
                    num_images_per_prompt=1,
                    device=self.llm_model.device
                )
            
            mse_loss = mse_loss_function(projected_hidden_states, sd_text_embeddings)
            mse_loss = mse_loss.mean()
                
        elif self.enable_perception and self.enable_generation:
            mse_loss_function = nn.MSELoss()
            count_system_have_sticker = 0
            
            for b, (one_labels, per_caption) in enumerate(zip(labels, caption)):
                hidden_states_list = []
                
                image_start_position = (one_labels == self.image_tokens_ids[0]).nonzero(as_tuple=False)[:, 0].tolist()
                image_end_position = (one_labels == self.image_tokens_ids[-1]).nonzero(as_tuple=False)[:, 0].tolist()
                num_image = min(len(image_start_position), len(image_end_position))
                
                if num_image > 0:
                    count_system_have_sticker += 1
                    image_start_position, image_end_position = image_start_position[:num_image], image_end_position[:num_image]
                    for (start, end) in zip(image_start_position, image_end_position):
                        hidden_states_list.append(outputs.hidden_states[-1][b, start: end + 1, :])
                    hidden_states = torch.stack(hidden_states_list, dim=0)  # [num_image, 32, 4096]
                    projected_hidden_states = self.outputProjection(hidden_states)  # [32, 77, 768]

                    with torch.no_grad():
                        sd_text_embeddings, _ = self.stable_diffusion.encode_prompt(
                            prompt=per_caption,
                            do_classifier_free_guidance=False,
                            num_images_per_prompt=1,
                            device=self.llm_model.device
                        )  # sd_text_embeddings: [num_image, 77, 768]
                
                    mse_loss += mse_loss_function(projected_hidden_states, sd_text_embeddings)
                
            if count_system_have_sticker > 0:
                mse_loss = mse_loss / count_system_have_sticker
            else:
                mse_loss = 0

        if not self.enable_perception and self.enable_generation:
            loss = mse_loss
        else:
            loss = lm_loss + mse_loss
            
        return PegsOutput(
            loss=loss,
            loss_lm=lm_loss,
            loss_gen=mse_loss
        )
            
    
    @torch.no_grad()
    def generate(
        self, 
        pixel_values: Optional[torch.FloatTensor] = None,
        text: Optional[str] = None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        min_length: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negativate_prompt: Optional[str] = None,
    ):
        generated_image = None
        # generation setting
        stop_words_ids = [
            torch.tensor([835]).to(self.llm_model.device),          # '###' can be encoded in two different ways.
            torch.tensor([2277, 29937]).to(self.llm_model.device),  # {"###": 835, "##": 2277, "#": 29937}
            torch.tensor([32033]).to(self.llm_model.device)         # [IMG32]
        ]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        if self.enable_perception and not self.enable_generation:
            _, input_embeds, input_attention_mask, _ = self.build_one_batch_perception(pixel_values, text)  # [1, seq_len, 4096], [1, seq_len]
            input_embeds = input_embeds[:, -self.max_text_length:, :]  # [1, min(seq_len, max_text_length), 4096]
            input_attention_mask = input_attention_mask[:, -self.max_text_length:]  # [1, min(seq_len, max_text_length)]
            
            outputs = self.llm_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_attention_mask,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=6,
                stopping_criteria=stopping_criteria,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
            )  # 'sequences', 'hidden_states', 'past_key_values'
            output_sequence = outputs.sequences[0][1:]
            
        elif not self.enable_perception and self.enable_generation:
            _, input_embeds, input_attention_mask, labels = self.build_one_batch_generation([text])

            outputs = self.llm_model(
                inputs_embeds=input_embeds,
                attention_mask=input_attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            print(f"outputs.hidden_states[-1].shape = {outputs.hidden_states[-1].shape}")
            hidden_states_list = []
            image_start_position = (labels == self.image_tokens_ids[0]).nonzero(as_tuple=False)[:, 1].tolist()  # len = b_s
            image_end_position = (labels == self.image_tokens_ids[-1]).nonzero(as_tuple=False)[:, 1].tolist()  # len = b_s
            # print(outputs.hidden_states[-1].shape)  # torch.Size([b_s, seq len, 4096])
            print(f"image_start_position = {image_start_position}")
            print(f"image_end_position = {image_end_position}")
            for b, (start, end) in enumerate(zip(image_start_position, image_end_position)):
                hidden_states_list.append(outputs.hidden_states[-1][b, start: end + 1, :])
            hidden_states = torch.stack(hidden_states_list, dim=0)  # torch.Size([b_s, 32, 4096])
            print(f"hidden_states.shape = {hidden_states.shape}")
            projected_hidden_states = self.outputProjection(hidden_states)  # torch.Size([b_s, 77, 768])
            print(f"projected_hidden_states.shape = {projected_hidden_states.shape}")
            
            outputs = self.stable_diffusion(
                prompt_embeds=projected_hidden_states,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negativate_prompt,
            )
            
            output_sequence = None
            generated_image = outputs.images[0]
        else:
            _, input_embeds, input_attention_mask, _ = self.build_one_instance_for_demo(pixel_values, text)
            outputs = self.llm_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_attention_mask,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=6,
                stopping_criteria=stopping_criteria,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
            )  # 'sequences', 'hidden_states', 'past_key_values'
            output_sequence = outputs.sequences[0][1:]
            output_embeddings = []
            for _hidden_states in outputs.hidden_states[1:]:
                output_embeddings.append(_hidden_states[-1])
            output_hidden_states = torch.cat(output_embeddings, dim=1)
            
            hidden_states_list = []
            if self.image_tokens_ids[0] in output_sequence:
                image_start_position = (output_sequence == self.image_tokens_ids[0]).nonzero(as_tuple=False)[:, 0].tolist()[0]  # len = b_s
                image_end_position = (output_sequence == self.image_tokens_ids[-1]).nonzero(as_tuple=False)[:, 0].tolist()[0]  # len = b_s
                
                if image_end_position + 1 - image_start_position == self.num_image_tokens:
                    gen_hidden_states = output_hidden_states[:, image_start_position: image_end_position + 1, :]
                    projected_hidden_states = self.outputProjection(gen_hidden_states)  # torch.Size([b_s, 77, 768])

                    logging.info(f"negativate_prompt = \n{negativate_prompt}")
                    random_seed = torch.seed()
                    generator = torch.Generator(device="cuda").manual_seed(random_seed)
                    sd_outputs = self.stable_diffusion(
                        prompt_embeds=projected_hidden_states,
                        negativate_prompt=negativate_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )
                    logging.info(f"is_NSFW: {sd_outputs.nsfw_content_detected[0]}")
                    if sd_outputs.nsfw_content_detected[0] == False:
                        generated_image = sd_outputs.images[0]
            
        
        return PegsGenerationOutput(
            output_sequence=output_sequence,
            image=generated_image
        )
        
        
        
        
        