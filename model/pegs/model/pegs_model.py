import os
import logging
import contextlib
from typing import Optional, List, Union, Dict
import pickle as pkl
from PIL import Image
import json

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.nn.utils import rnn
from torch.utils import tensorboard
import deepspeed

import torch.nn.functional as F
from transformers import (LlamaConfig, LlamaTokenizer, LlamaForCausalLM,
                          Blip2Config, Blip2ForConditionalGeneration,
                          CLIPTokenizer, CLIPTextModel, CLIPVisionModel,
                          BitsAndBytesConfig, StoppingCriteriaList)
from transformers.modeling_outputs import CausalLMOutput
from peft import LoraConfig, get_peft_model, TaskType
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, StableDiffusionImg2ImgPipeline, \
                        StableDiffusionInstructPix2PixPipeline
from diffusers.training_utils import compute_snr

from pegs.model.projection_layers import InputProjectionLayer, OutputProjectionLayer, RetrievalTextMapper, \
                                            PEGSMLP, PEGSSeqMLP, LinearPoolProjection, SLP
from pegs.model.utils import StoppingCriteriaSub, disabled_train, convert_weights_to_fp16
from pegs.model.loss import contrastive_loss, contrastive_acc
from pegs.datasets.builders.base_builder import BaseDatasetBuilder
from pegs.model.outputs import PegsOutput, PegsGenerationOutput
from pegs.register import registry


@registry.register_model("pegs_rag")
class PEGS(nn.Module):
    def __init__(
        self,
        llama_pretrained_model_path_or_name: str = None,
        use_lora: bool = False,
        enable_perception: bool = False,
        enable_retrieval: bool = False,
        enable_generation: bool = False,
        # perception
        blip2_pretrained_model_path_or_name: str = None,
        freeze_vision_model: bool = True,
        freeze_qformer: bool = True,
        vision_precision: str = "fp16",
        # retrieval
        ret_emb_dim: int = 256,
        visual_emb_matrix_path_or_name: str = None,
        vision_processor: str = None,
        visual_encoder: str = None,
        tau: float = 0.07,  # temperature coefficient of InfoNCE
        train_logit_scale: bool = False,
        # generation
        stable_diffusion_pretrained_model_path_or_name: str = None,
        freeze_stable_diffusion: bool = True,
        num_clip_tokens: int = 77,
        prompt_embeddings_dim: int = 1024,
        # other settings
        padding_side: Optional[str] = None,
        max_text_length: Optional[int] = None,
        image_prefix_token: Optional[str] = "<Img>",
        image_postfix_token: Optional[str] = "</Img>",
        image_placeholder_token: Optional[str] = "<IMG>",
        num_image_tokens_for_retrieval: int = 32,
        num_image_tokens_for_generation: int = 32,
        use_prefix_prompt: bool = False,
        prefix_prompt: Optional[str] = None,
        low_resource: bool = False,  # use 8 bit and put vit in cpu
        device_8bit: int = 0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        feature_accumulation_steps: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.enable_perception: bool = enable_perception
        self.enable_retrieval: bool = enable_retrieval
        self.enable_generation: bool = enable_generation
        
        self.low_resource: bool = low_resource
        self.device_8bit = device_8bit
        self.ret_emb_dim = ret_emb_dim
        self.feature_accumulation_steps = feature_accumulation_steps
        self.prompt_embeddings_dim = prompt_embeddings_dim
              
        if enable_perception or enable_retrieval:
            logging.info("Loading Vision Encoder and Q-Former...")
            self.blip2_config, self.vision_model, self.qformer, self.query_tokens = self._init_from_blip2(
                    blip2_pretrained_model_path_or_name,
                )
            
            self.visual_encoder = CLIPVisionModel.from_pretrained(visual_encoder)
                
            if vision_precision == "fp16":
                convert_weights_to_fp16(self.vision_model)
                convert_weights_to_fp16(self.visual_encoder)
        
            if freeze_vision_model:
                for _, param in self.vision_model.named_parameters():
                    param.requires_grad = False
                self.vision_model.eval()
                self.vision_model.train = disabled_train
                
                for _, param in self.visual_encoder.named_parameters():
                    param.requires_grad = False
                self.visual_encoder.eval()
                self.visual_encoder.train = disabled_train
                
                logging.info("Freeze Vision Encoder.")
            logging.info("Vision Encoder has been loaded.")
        
            self.qformer_config = self.blip2_config.qformer_config
            self.vision_config = self.blip2_config.vision_config 
            if (enable_perception or enable_retrieval) and freeze_qformer:
                for _, param in self.qformer.named_parameters():
                    param.requires_grad = False
                self.qformer.eval()
                self.qformer.train = disabled_train
                self.query_tokens.requires_grad = False
                logging.info("Freeze Q-Former.")
            logging.info("Q-Former has been loaded.")
        
        logging.info("Loading LLM...")
        self.llm_config, self.tokenizer, self.llm_model = self._init_from_llama(
            llama_pretrained_model_path_or_name, device_8bit
        )
        
        self.use15 = False
        if "1.5" in llama_pretrained_model_path_or_name:
            self.use15 = True
        
        # tokenizer settings
        self.tokenizer.padding_side = padding_side if padding_side is not None else "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_text_length = max_text_length if max_text_length is not None else 1024
        
        # image tokens
        self.image_prefix_token = image_prefix_token
        self.image_postfix_token = image_postfix_token
        self.image_placeholder_token =image_placeholder_token  # not as a special token. refer to the image
        self.num_image_tokens_for_retrieval = num_image_tokens_for_retrieval
        self.num_image_tokens_for_generation = num_image_tokens_for_generation
        self._add_image_tokens()
        
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
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        logging.info("LLM has been loaded.")
        
        self._init_embed_tokens()
        
        if enable_generation:
            self.sd_tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_pretrained_model_path_or_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_pretrained_model_path_or_name, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(stable_diffusion_pretrained_model_path_or_name, subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained(stable_diffusion_pretrained_model_path_or_name, subfolder="unet")
            if freeze_stable_diffusion:
                self.text_encoder.requires_grad_(False)
                unet.requires_grad_(False)
                vae.requires_grad_(False)
                logging.info("Freeze Stable Diffusion.")
            self.stable_diffusion = StableDiffusionImg2ImgPipeline.from_pretrained(
                stable_diffusion_pretrained_model_path_or_name,
                text_encoder=self.text_encoder, unet=unet, vae=vae
                )
            self.stable_diffusion_t2i = StableDiffusionPipeline.from_pretrained(
                stable_diffusion_pretrained_model_path_or_name,
                text_encoder=self.text_encoder, unet=unet, vae=vae
            ).to(torch.float16)
            logging.info("Stable Diffusion has been loaded.")

            self.num_clip_tokens = num_clip_tokens
            self.vision_processor = BaseDatasetBuilder._build_processor_from_config(vision_processor)
            logging.info(f'vision_processor:{vision_processor}')
            

        # projection layer
        if enable_perception:
            self.inputProjection = InputProjectionLayer(self.qformer_config.hidden_size, self.llm_config.hidden_size)
            # self.inputProjection.requires_grad_(False)
        if enable_retrieval:
            self.text_fc = RetrievalTextMapper(self.llm_config.hidden_size, self.ret_emb_dim).to(torch.float16)
            self.visual_fc = nn.Linear(self.visual_encoder.config.hidden_size, self.ret_emb_dim)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau))
            if not train_logit_scale:  
                self.logit_scale.requires_grad = False
        if enable_generation:
            self.outputProjection = OutputProjectionLayer(self.llm_config.hidden_size, num_clip_tokens, prompt_embeddings_dim)
            
        if enable_retrieval:
            # loading visual_emb_matrix
            self._init_from_visual_emb_matrix_path_or_name(visual_emb_matrix_path_or_name)
            self.ret_num = 0
        
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
        query_tokens = blip2_model.query_tokens  # torch.Size([1, 32, 768])
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

    def _add_image_tokens(self):
        self.image_tokens = [f"[IMG{i+1}]" for i in range(self.num_image_tokens_for_retrieval + self.num_image_tokens_for_generation)]
        self.num_image_tokens = len(self.image_tokens)
        special_tokens = [self.image_prefix_token, self.image_postfix_token]
        logging.info(f"\nadd: special_tokens = {special_tokens}")
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        for image_token in self.image_tokens:
            self.tokenizer.add_tokens([image_token])
        self.image_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.image_tokens)
        self.tokenizer.save_pretrained("PEGS_tokenizer")
            
        # resize token embeddings
        self.llm_model.resize_token_embeddings(len(self.tokenizer))
        # embeddings / token id
        with torch.no_grad():
            image_prefix_token_id = self.tokenizer(self.image_prefix_token, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.llm_model.device)
            self._image_prefix_embeds = self.llm_model.model.embed_tokens(image_prefix_token_id)
            image_postfix_token_id = self.tokenizer(self.image_postfix_token, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.llm_model.device)
            self._image_postfix_embeds = self.llm_model.model.embed_tokens(image_postfix_token_id)
            
            self._image_tokens_input_ids = torch.tensor(self.image_tokens_ids).unsqueeze(0)  # torch.Size([1, 32])
            self._image_tokens_attention_mask = torch.ones(self._image_tokens_input_ids.size(), dtype=torch.long)  # torch.Size([1, 32])
            self._image_tokens_labels = torch.tensor(self.image_tokens_ids).unsqueeze(0)  # torch.Size([1, 32])
            
    def _init_from_visual_emb_matrix_path_or_name(
        self,
        visual_emb_matrix_path_or_name: str
    ):
        if visual_emb_matrix_path_or_name is None:
            return
        self.visual_emb_matrix = []
        self.path_imgs = []
        with open(visual_emb_matrix_path_or_name, 'rb') as f:
            vector_db = pkl.load(f)
        self.visual_emb_matrix = np.stack(vector_db['embeddings'], axis=0)  # emb of imgs
        self.path_imgs = vector_db['path']  # path of the image to be retrieved
        # preproccess the visual_emb_matrix
        logit_scale = self.logit_scale.exp()
        self.visual_emb_matrix = torch.tensor(self.visual_emb_matrix, dtype=logit_scale.dtype).to(logit_scale.device)
        logging.info(f'self.visual_emb_matrix:{self.visual_emb_matrix.shape}')
        self.visual_emb_matrix = self.visual_emb_matrix / self.visual_emb_matrix.norm(dim=-1, keepdim=True)
        self.visual_emb_matrix = logit_scale * self.visual_emb_matrix
        logging.info("loaded the visual embedding matrix")

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
        return self.llm_model.model.model.embed_tokens(self.image_tokens_input_ids) if hasattr(self.llm_model.model, "model") else self.llm_model.model.embed_tokens(self.image_tokens_input_ids)
    
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
                        param.grad = param.grad * self.input_embeds_grad_mask
                    elif "lm_head" in name:
                        param.grad = param.grad * self.output_embeds_grad_mask
            else:
                self.llm_model.get_input_embeddings().weight.grad = self.llm_model.get_input_embeddings().weight.grad * self.input_embeds_grad_mask.to(self.llm_model.device)
                if self.enable_perception:
                    self.llm_model.get_output_embeddings().weight.grad = self.llm_model.get_output_embeddings().weight.grad * self.output_embeds_grad_mask.to(self.llm_model.device)

    def image_encoding(self, pixel_values: torch.FloatTensor):
        device = self.vision_model.device

        with self.maybe_autocast():  # Required, otherwise an error "RuntimeError: expected scalar type Half but found Float" will be reported.
            pixel_values = pixel_values.to(self.vision_model.device)  # torch.Size([batch size, 3, 224, 224])
            vision_outputs = self.vision_model(pixel_values)
        
            image_embeds = vision_outputs.last_hidden_state.to(device)  # torch.Size([batch size, 257(=1+16*16), 1408])
            image_attention_mask = torch.ones(image_embeds.size()[:-1],
                                              dtype=torch.long,
                                              device=device)  # torch.Size([batch size, 257])
        
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask
            )  # BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state, pooler_output, hidden_states, past_key_values, attentions, cross_attentions)
            query_output = query_outputs.last_hidden_state  # torch.Size([batch size, 32, 768])  # 32 is the number of query tokens
        
            image_content_embeds = self.inputProjection(query_output)  # torch.Size([batch size, 32, 4096])
            image_prefix_embeds = self.image_prefix_embeds.expand(image_embeds.shape[0], -1, -1)  # torch.Size([batch size, 4, 4096])
            image_postfix_embeds = self.image_postfix_embeds.expand(image_embeds.shape[0], -1, -1)  # torch.Size([batch size, 4, 4096])
            inputs_ids = torch.cat([image_prefix_embeds, image_content_embeds, image_postfix_embeds], dim=1)  # torch.Size([batch size, 38(32+3+3), 4096])
            inputs_attention_mask = torch.ones(inputs_ids.size()[:-1], dtype=torch.long, device=device)  # torch.Size([batch size, 38(32+3+3)])
            labels = inputs_attention_mask * (-100)
            
        return inputs_ids, inputs_attention_mask, labels
    
    def get_visual_embed(self, pixel_values: torch.FloatTensor, only_clip=False):
        with self.maybe_autocast():
            pixel_values = pixel_values.to(self.visual_encoder.device)  # torch.Size([batch size, 3, 224, 224])
            vision_outputs = self.visual_encoder(pixel_values)
            image_embeds = vision_outputs.pooler_output.to(self.visual_encoder.device) # torch.Size([batch size, 1408])
            if only_clip:  #only use clip to encode
                visual_embs = image_embeds
            else:
                visual_embs = self.visual_fc(image_embeds)  # torch.Size([batch size, 256(ret_emb_dim)])
            visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 1, -1)) # torch.Size([batch size, 1, 256])
        return visual_embs
    
    def get_visual_fc(self, image_embeds: torch.FloatTensor):
        with self.maybe_autocast():
            image_embeds = image_embeds.to(self.visual_encoder.device) # torch.Size([batch size, 1, 1408])
            image_embeds = torch.reshape(image_embeds, (1, -1)) # torch.Size([batch size, 1408])
            visual_embs = self.visual_fc(image_embeds)  # torch.Size([batch size, 256(ret_emb_dim)])
            visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 1, -1)) # torch.Size([batch size, 1, 256])
        return visual_embs
    
    def _init_embed_tokens(self) -> None:
        if self.use_lora:
            self.embed_tokens = self.llm_model.base_model.model.model.embed_tokens
        else:
            self.embed_tokens = self.llm_model.model.embed_tokens
            
    def get_bos_encoding(self, batch_size):
        bos_input_ids = torch.ones([batch_size, 1], dtype=torch.int64, device=self.llm_model.device) * self.tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos_input_ids)  # torch.Size([batch_size, 1, 4096])
        bos_attention_mask = torch.ones([batch_size, 1], dtype=torch.int64, device=self.llm_model.device)  # torch.Size([batch_size, 1])
        bos_labels = bos_attention_mask * (-100)
        
        if self.prefix_prompt is not None:
            prefix_prompt_encoded = self.tokenizer([self.prefix_prompt] * batch_size,
                                                   add_special_tokens=False,
                                                   return_tensors="pt").to(self.llm_model.device)
            prefix_prompt_embeds = self.embed_tokens(prefix_prompt_encoded["input_ids"])
            prefix_prompt_attention_mask = prefix_prompt_encoded["attention_mask"]
            prefix_prompt_labels = prefix_prompt_attention_mask * (-100)
            
            bos_input_ids = torch.cat([bos_input_ids, prefix_prompt_encoded["input_ids"]], dim=1)
            bos_embeds = torch.cat([bos_embeds, prefix_prompt_embeds], dim=1)
            bos_attention_mask = torch.cat([bos_attention_mask, prefix_prompt_attention_mask], dim=1)
            bos_labels = torch.cat([bos_labels, prefix_prompt_labels], dim=1)
            
        self.bos_tokens_num = bos_labels.shape[1]
        
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
        
        text_embeds = self.embed_tokens(input_ids)  # torch.Size([batch size, sequence length, 4096])
        
        return input_ids, text_embeds, attention_mask
    
    def build_one_instance_perception(self, pixel_values: torch.FloatTensor, text: str):
      
        input_ids_list, input_embeds_list, input_attention_mask_list, labels_list = [], [], [], []
        
        image_embeds, image_attention_mask, image_labels = self.image_encoding(pixel_values)
        image_input_ids = torch.zeros(image_labels.size(), dtype=torch.long, device=self.llm_model.device)  # Just as a placeholder input_ids. Don't affect training
        input_ids_list.append(image_input_ids)
        input_embeds_list.append(image_embeds)  # one_image_embeds: [1, 34(=1+32+1), 4096]
        input_attention_mask_list.append(image_attention_mask)  # one_image_attention_mask: [1, 34(=1+32+1)]
        labels_list.append(image_labels)  # one_image_labels: [1, 34(=1+32+1)]
        
        text_input_ids, text_embeds, text_attention_mask = self.text_encoding(text)
        input_ids_list.append(text_input_ids)
        input_embeds_list.append(text_embeds)  # one_text_embeds: [1, sequence length, 4096]
        input_attention_mask_list.append(text_attention_mask)  # one_text_attention_mask: [1, sequence length]
        labels_list.append(text_input_ids.masked_fill(text_input_ids == self.tokenizer.pad_token_id, -100))  # one_input_ids: [1, sequence length]

        bos_input_ids, bos_embeds, bos_attention_mask, bos_labels = self.get_bos_encoding(1)
        
        input_ids = torch.cat([bos_input_ids] + input_ids_list, dim=1)
        input_embeds = torch.cat([bos_embeds] + input_embeds_list, dim=1)  # [1, sequence length, 4096]
        input_attention_mask = torch.cat([bos_attention_mask] + input_attention_mask_list, dim=1)  # [1, sequence length]
        labels = torch.cat([bos_labels] + labels_list, dim=1)  # [1, sequence length]

        assert input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape
        
        return input_ids, input_embeds, input_attention_mask, labels
    
    def build_one_instance_generation(self, pixel_values: torch.FloatTensor, text: str):
        device = self.llm_model.device
        # input_ids, text_embeds, attention_mask
        text_input_ids, text_embeds, text_attention_mask = self.text_encoding(text)  # torch.Size([1, seq len, 4096]), torch.Size([1, seq len]), torch.Size([1, seq len])
        max_txt_length = self.max_text_length - self.num_image_tokens - 1
        text_input_ids = text_input_ids[:, :max_txt_length]
        text_embeds = text_embeds[:, :max_txt_length, :]
        text_attention_mask = text_attention_mask[:, :max_txt_length]
        text_labels = text_attention_mask * (-100)  # torch.Size([1, seq len])
        # text_labels = torch.clone(text_input_ids)
        # print(f'text_labels:{text_labels}, shape:{text_labels.shape}')
        
        bos_input_ids, bos_embeds, bos_attention_mask, bos_labels = self.get_bos_encoding(1)
        
        input_ids = torch.cat([bos_input_ids, text_input_ids, self.image_tokens_input_ids], dim=1)  # torch.Size([1, 1 + seq len + 34])
        input_embeds = torch.cat([bos_embeds, text_embeds, self.image_tokens_embeds], dim=1)  # torch.Size([1, 1 + seq len + 34, 4096])
        input_attention_mask = torch.cat([bos_attention_mask, text_attention_mask, self.image_tokens_attention_mask], dim=1)  # torch.Size([1, 1 + seq len + 34])
        labels = torch.cat([bos_labels, text_labels, self.image_tokens_labels], dim=1)  # torch.Size([1, 1 + seq len + 34])
        assert input_ids.shape == input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape
        
        return input_ids, input_embeds, input_attention_mask, labels
        
    def build_one_instance_perception_and_generation(self, pixel_values: torch.FloatTensor, text: str):
        text_list = text.split(self.image_placeholder_token)
        # print(f'text_list:{len(text_list)}')
        input_ids_list, input_embeds_list, input_attention_mask_list, labels_list = [], [], [], []
        split_word = "### "
        human_begin = "Human"
        assistant_begin = "Assistant"
        if self.use15:
            split_word = "\n"
            human_begin = "USER:"
            assistant_begin = "ASSISTANT"
        for i, one_text in enumerate(text_list):
            if one_text != "":
                one_input_ids, one_text_embeds, one_text_attention_mask = self.text_encoding(one_text)
                
                input_ids_list.append(one_input_ids)  # one_input_ids: [1, sequence length]
                input_embeds_list.append(one_text_embeds)  # one_text_embeds: [1, sequence length, 4096]
                input_attention_mask_list.append(one_text_attention_mask)  # one_text_attention_mask: [1, sequence length]
                labels_list.append(one_input_ids.masked_fill(one_input_ids == self.tokenizer.pad_token_id, -100))  # one_input_ids: [1, sequence length]
            if i != len(text_list) - 1:
                if human_begin == one_text.split(split_word)[-1][:5]:  # image for perception
                    one_image_embeds, one_image_attention_mask, one_image_labels = self.image_encoding(pixel_values[i].unsqueeze(0))
                    one_input_ids = torch.zeros(one_image_labels.size(), dtype=torch.long, device=self.llm_model.device)  # Just as a placeholder input_ids. Don't affect training
                    # print(f'input_ids_list:{input_ids_list[len(input_ids_list)-1].shape}')
                    input_ids_list.append(one_input_ids)  # one_input_ids: [1, 34(=1+32+1)]
                    input_embeds_list.append(one_image_embeds)  # one_image_embeds: [1, 34(=1+32+1), 4096]
                    input_attention_mask_list.append(one_image_attention_mask)  # one_image_attention_mask: [1, 34(=1+32+1)]
                    labels_list.append(one_image_labels)  # one_image_labels: [1, 34(=1+32+1)]
                elif assistant_begin == one_text.split(split_word)[-1][:9]:   # image for generation
                    # print(f'input_ids_list:{input_ids_list[len(input_ids_list)-1].shape}')
                    input_ids_list.append(self.image_tokens_input_ids)  # image_tokens_input_ids: [1, 32(8+24)]
                    input_embeds_list.append(self.image_tokens_embeds)  # image_tokens_embeds: [1, 32(8+24), 4096]
                    input_attention_mask_list.append(self.image_tokens_attention_mask)  # image_tokens_attention_mask: [1, 32(8+24)])
                    labels_list.append(self.image_tokens_labels)  # image_tokens_labels: [1, 32(8+24)])
                else:
                    logging.warning(f"Training data is irregular!\n{one_text.split(split_word)[-1]}")

        bos_input_ids, bos_embeds, bos_attention_mask, bos_labels = self.get_bos_encoding(1)
        input_ids = torch.cat([bos_input_ids] + input_ids_list, dim=1)  # [1, sequence length]
        input_embeds = torch.cat([bos_embeds] + input_embeds_list, dim=1)  #[1, sequence length, 4096]
        input_attention_mask = torch.cat([bos_attention_mask] + input_attention_mask_list, dim=1)  # [1, sequence length]
        labels = torch.cat([bos_labels] + labels_list, dim=1)  # [1, sequence length]

        assert input_ids.shape == input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape
        
        return input_ids, input_embeds, input_attention_mask, labels
    
    def build_one_batch_pertrain_data(self, pixel_values: List[torch.FloatTensor], text: List[str], mode):
        """
        build a batch data for pertrain
        """
        if 'captioning' in mode:
            build_instance_function = self.build_one_instance_perception
        elif 'retrieval' in mode or 'generation' in mode:
            build_instance_function = self.build_one_instance_generation
        else:
            build_instance_function = self.build_one_instance_generation
            
        input_ids_list, input_embeds_list, input_attention_mask_list, labels_list = [], [], [], []
        for one_pixel_values, one_text in zip(pixel_values, text):
            one_input_ids, one_input_embeds, one_input_attention_mask, one_labels = build_instance_function(one_pixel_values, one_text)
            # logging.info(f'one_input_ids:{one_input_ids}, one_input_embeds:{one_input_embeds}, one_input_attention_mask:{one_input_attention_mask},\
            #       one_labels:{one_labels}')
            if one_input_ids is not None:
                input_ids_list.append(one_input_ids.squeeze(0))
            input_embeds_list.append(one_input_embeds.squeeze(0))
            input_attention_mask_list.append(one_input_attention_mask.squeeze(0))
            labels_list.append(one_labels.squeeze(0))
        
        if input_ids_list != []:
            input_ids = rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)  # torch.Size([batch size, seq len])
        else:
            input_ids = None
        input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)[:, :self.max_text_length, :]  # torch.Size([Batch size, min(sequence length, max_text_length), 4096])
        input_attention_mask = rnn.pad_sequence(input_attention_mask_list, batch_first=True, padding_value=0)[:, :self.max_text_length]  # torch.Size([Batch size, min(sequence length, max_text_length)])
        labels = rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)[:, :self.max_text_length]  # torch.Size([Batch size, min(sequence length, max_text_length)])
        assert input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape
        # print(f'labels:{labels}, shape:{labels.shape}')
        return input_ids, input_embeds, input_attention_mask, labels
    
    def build_one_batch(self, pixel_values: torch.FloatTensor, text: List[str]):
        if self.enable_perception and self.enable_generation:
            build_instance_function = self.build_one_instance_perception_and_generation
        
        input_ids_list, input_embeds_list, input_attention_mask_list, labels_list = [], [], [], []
        for one_pixel_values, one_text in zip(pixel_values, text):
            one_input_ids, one_input_embeds, one_input_attention_mask, one_labels = build_instance_function(one_pixel_values, one_text)

            input_ids_list.append(one_input_ids.squeeze(0))
            input_embeds_list.append(one_input_embeds.squeeze(0))
            input_attention_mask_list.append(one_input_attention_mask.squeeze(0))
            labels_list.append(one_labels.squeeze(0))
        
        input_ids = rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)[:, :self.max_text_length]  # [batch size, min(seq len, max_text_length)]
        input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)[:, :self.max_text_length, :]  # [Batch size, min(seq len, max_text_length), 4096]
        input_attention_mask = rnn.pad_sequence(input_attention_mask_list, batch_first=True, padding_value=0)[:, :self.max_text_length]  # [Batch size, min(seq len, max_text_length)]
        labels = rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)[:, :self.max_text_length]  # [Batch size, min(seq len, max_text_length)]
        assert input_ids.shape == input_embeds.shape[:-1] == input_attention_mask.shape == labels.shape

        return input_ids, input_embeds, input_attention_mask, labels
    
    def generate_tokens_embeddings(self, input_embeds, max_new_tokens, temperature: float = 0.0, top_p: float = 1.0, eval: bool = False):
        """
        This function is used to generate the tokens and output embeddings that employed to generate images/videos/audios
        inputs: dict
        input_embeds: tensor
        return:
            out: the output tokens index
            output_embeddings: output embeddings for synthesizing images
            output_logits: the logits of output tokens
            video_output_embedding: output embeddings for synthesizing video
        """
        with torch.no_grad():  # no tracking history
            # init output with image tokens
            out = None
            output_embeddings = []
            
            stop_words_ids = [
                          torch.tensor([835]).to(self.llm_model.device),  # '###' can be encoded in two different ways.
                          torch.tensor([2277, 29937]).to(self.llm_model.device),
                          torch.tensor([2]).to(self.llm_model.device),  # "</s>"
                          torch.tensor([13]).to(self.llm_model.device),  # "\n"
                        #   torch.tensor([32002]).to(self.llm_model.device),
                        #   torch.tensor([32006]).to(self.llm_model.device),
                        #   torch.tensor([32007]).to(self.llm_model.device),
                        #   torch.tensor([32033]).to(self.llm_model.device),
                        #   torch.tensor(self.image_tokens_ids[-1]).to(self.llm_model.device)
                          ]  # {"###": 835, "##": 2277, "#": 29937, "[IMG{32}]": 32033}
            stop_words_ids.extend([torch.tensor([img_idx]).to(self.llm_model.device) for img_idx in self.image_tokens_ids])
            logging.info(f'stop_words_ids:{stop_words_ids}')

            for i in range(max_new_tokens):
                logging.info(f'input_embeds:{input_embeds}, shape:{input_embeds.shape}')
                output = self.llm_model(
                            inputs_embeds=input_embeds, use_cache=False,
                            output_hidden_states=True
                        )

                output_embeddings.append(output.hidden_states[-1])

                stop_count = 0
                
                for stop in stop_words_ids:
                    if not (out is None) and torch.all((stop == out[0][-len(stop):])).item():  # out[0]: "logits"
                        stop_count += 1
                if stop_count >= 1:
                    break

                logits = output.logits[:, -1, :]  # (N, vocab_size)

                # Prevent the model from generating the [IMG1..n] tokens.
                filter_value = -float('Inf')
                logits[:, self.image_tokens_ids[1:]] = filter_value
                if self.image_tokens_ids and self.image_tokens_ids[0] != -1:
                    if i < 0:
                        # Eliminate probability of generating [IMG] if this is earlier than min_word_tokens.
                        logits[:, self.image_tokens_ids] = filter_value

                if temperature == 0.0:
                    if top_p != 1.0:
                        raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                else:
                    logits = logits / temperature

                    # Apply top-p filtering.
                    if top_p < 1.0:
                        assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # (N, D)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        # print('sorted_indices shape: ', sorted_indices.shape)
                        for j in range(sorted_indices.shape[0]):
                            indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
                            logits[j, indices_to_remove] = filter_value
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)

                # Force generation of the remaining [IMG] tokens if [IMG0] is generated.
                if next_token.shape[0] == 1 and next_token.item() == self.image_tokens_ids[0]:
                    next_token = torch.tensor(self.image_tokens_ids)[None, :].long().to(
                        input_embeds.device)  # (1, num_tokens)
                    # next_token_res = self.tokenizer.decode(
                    # next_token[0],
                    # add_special_tokens=False,
                    # skip_special_tokens=True
                    # )
                    # logging.info(f'next_token:{next_token} next_token_res:{next_token_res}')
                    # break
                else:
                    next_token = next_token.long().to(input_embeds.device)
                    
                # Force generation of [IMGr] 
                if eval:
                    next_token = torch.tensor(self.image_tokens_ids)[None, :].long().to(
                            input_embeds.device)

                if out is not None:
                    out = torch.cat([out, next_token], dim=-1)
                else:
                    out = next_token
                next_token_res = self.tokenizer.decode(
                    next_token[0],
                    add_special_tokens=False,
                    skip_special_tokens=True
                )
                logging.info(f'next_token:{next_token} next_token_res:{next_token_res}')
                next_embedding = self.embed_tokens(next_token)
                input_embeds = torch.cat([input_embeds, next_embedding], dim=1)
        # logging.info(f'out:{out}')
        return out, output_embeddings
        
    def retrieval_image(
        self,
        seen_image_idx: list,
        labels: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        multi_img: bool = False
    ):
        self.ret_num += 1
        if self.ret_num >= 10:
            seen_image_idx.clear()
            self.ret_num = 0
        # 1.get position of <IMG>
        ret_image_start_position = (labels == self.image_tokens_ids[0]).nonzero(as_tuple=False)[:, 0].tolist()[0]  # len = batch size
        ret_image_end_position = (labels == self.image_tokens_ids[self.num_image_tokens_for_retrieval-1]).nonzero(as_tuple=False)[:, 0].tolist()[0]

        gen_image_start_position = (labels == self.image_tokens_ids[self.num_image_tokens_for_retrieval]).nonzero(as_tuple=False)[:, 0].tolist()[0]
        gen_image_end_position = (labels == self.image_tokens_ids[self.num_image_tokens_for_retrieval + self.num_image_tokens_for_generation - 1]).nonzero(as_tuple=False)[:, 0].tolist()[0]  # len = batch size
        
        # 2.encode hidden states of <IMG> for retrieval
        input_embs = self.image_tokens_embeds[:, 0: self.num_image_tokens_for_retrieval, :]  #
        # logging.info(f'input_embs:{input_embs}, shape:{input_embs.shape}')
        ret_hidden_states = hidden_states[0, ret_image_start_position: ret_image_end_position + 1, :] # torch.Size([1, 8, 4096])
        gen_hidden_states = hidden_states[0, gen_image_start_position: gen_image_end_position + 1, :] # torch.Size([1, 24, 4096])

        ret_emb = self.text_fc(ret_hidden_states.unsqueeze(0))
        ret_emb = ret_emb[:, 0, :]

        ret_emb = ret_emb / ret_emb.norm(dim=-1, keepdim=True)
        ret_emb = ret_emb.type(self.visual_emb_matrix.dtype)
        self.visual_emb_matrix = self.visual_emb_matrix.to(ret_emb.device)
        # 3.calculate scores
        scores = self.visual_emb_matrix @ ret_emb.t()  # Iterate over each image
        # don not retrieval seen images.
        for seen_idx in seen_image_idx:
            scores[seen_idx, :] -= 1000
        
        # 4.Get the top image
        top_image_score, top_image_idx = scores.squeeze().topk(5)  

        ret_imgs = []
        for i, img_idx in enumerate(top_image_idx):
            seen_image_idx.append(img_idx)
            ret_img_tmp = Image.open(self.path_imgs[img_idx])
            if multi_img:
                ret_imgs.append(ret_img_tmp)

        ret_img = Image.open(self.path_imgs[top_image_idx[0]])
        if multi_img:
            return ret_imgs, ret_hidden_states, gen_hidden_states
        return ret_img, ret_hidden_states, gen_hidden_states
        
    def forward(
        self,
        pixel_values: List[torch.FloatTensor],
        text: List[str],
        emo_text: Optional[List[str]] = None,
        caption: Optional[torch.FloatTensor] = None,
        mode: Optional[Union[str, List]] = None,  # ['captioning', 'retrieval', 'generation']
        step: int = None,
        **kwargs
    ) -> CausalLMOutput:
        logging.info('......new forwarding......')
        lm_factor = 1.0
        ret_factor = 1.0
        gen_factor = 1.0
        if 'captioning' in mode:
            input_ids, inputs_embeds, inputs_attention_mask, labels = self.build_one_batch(pixel_values, text)  # instruction finetuning
        else:
            input_ids, inputs_embeds, inputs_attention_mask, labels = self.build_one_batch_pertrain_data(pixel_values, text, mode)  # pertrain
        
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=labels
        )  # 'loss', 'logits', 'past_key_values', 'hidden_states'
        lm_loss = outputs.loss  # language model loss
        logging.info(f"lm_loss:{lm_loss}") 
        
        loss = 0
        ret_loss = 0
        gen_loss = 0
        
        lm_loss = lm_factor * lm_loss
        logging.info(f"lm_loss:{lm_loss}") 
            
        if 'captioning' in mode:  # perception
            loss += lm_loss
            
        if 'retrieval' in mode:  # retrieval-only
            if step % self.feature_accumulation_steps == 1:  # init at the first step
                self.visual_embs_list = []
                self.text_embs_list = []
                # self.hidden_states = []
            logit_scale = self.logit_scale.exp()
            for idx, label in enumerate(labels):  # batch by batch processing
                ret_start_position = (label == self.image_tokens_ids[0]).nonzero(as_tuple=False)  # len = batch size
                ret_end_position = (label == self.image_tokens_ids[self.num_image_tokens_for_retrieval-1]).nonzero(as_tuple=False)  # len = batch size
                ret_start_position = ret_start_position[:, 0].tolist()
                ret_end_position = ret_end_position[:, 0].tolist()
                for num_ret_img, (start, end) in enumerate(zip(ret_start_position, ret_end_position)):  # process per img
                    num_perception = 0
                    if "captioning" in mode:
                        num_perception = (torch.count_nonzero(torch.eq(label[:start], -100))-self.bos_tokens_num) // 34  # self.bos_tokens_num=24
                    hidden_states = outputs.hidden_states[-1][idx, start: end + 1, :].clone().detach() # torch.Size([batch size, 8, 4096])
                    text_emb = self.text_fc(hidden_states.unsqueeze(0))  # torch.Size([1, 1, 256]).
                    visual_emb = self.get_visual_embed(pixel_values[idx][num_perception].unsqueeze(0)) \
                        if pixel_values[idx][num_perception].dim()==3 else self.get_visual_embed(pixel_values[idx].unsqueeze(0)) # torch.Size([1, 1, 256])
                    visual_emb = visual_emb[:, 0, :]  # torch.Size([1, 256])
                    visual_emb = visual_emb / visual_emb.norm(dim=-1, keepdim=True)
                    
                    text_emb = text_emb[:, 0, :]
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    
                    visual_emb = logit_scale * visual_emb
                    
                    text_emb = text_emb.type(torch.float16)
                    self.visual_embs_list.append(visual_emb)
                    self.text_embs_list.append(text_emb)
            
            if step % self.feature_accumulation_steps == 0:  # compute loss at the given step
                if not self.visual_embs_list:
                    ret_loss = 0
                else:
                    visual_embs = torch.cat(self.visual_embs_list)
                    text_embs = torch.cat(self.text_embs_list)
                    logits_per_image = visual_embs @ text_embs.t()  # Iterate over each image
                    logits_per_text = logits_per_image.t()  # Iterate over each text
                    caption_loss = contrastive_loss(logits_per_text)

                    image_loss = contrastive_loss(logits_per_image) 
                    caption_acc1, caption_acc5 = contrastive_acc(logits_per_text, topk=(1, 5))
                    image_acc1, image_acc5 = contrastive_acc(logits_per_image, topk=(1, 5))
                    ret_loss = ret_factor * (caption_loss + image_loss) / 2.0
                    loss += ret_loss
                    # hidden_states = torch.stack(hidden_states_list, dim=0)  # torch.Size([batch size, 8, 4096])
                    
                logging.info(f"ret_loss:{ret_loss}")
                logging.info(f"loss:{loss}")

        if 'generation' in mode:  # retrieval and generation
            # align with text encoder directly
            mse_loss = 0
            mse_loss_function = nn.MSELoss()
            if caption is not None:
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
                        hidden_states = torch.stack(hidden_states_list, dim=0)  # [num_image, num_image_tokens, 4096]
                        projected_hidden_states = self.outputProjection(hidden_states)  # [num_image, 77, 1024]
                        
                        with torch.no_grad():
                            sd_text_embeddings, _ = self.stable_diffusion.encode_prompt(
                                prompt=per_caption,
                                do_classifier_free_guidance=False,  # True
                                num_images_per_prompt=1,
                                device=self.llm_model.device
                            )  # sd_text_embeddings: [num_image, 77, 1024]

                        mse_loss += mse_loss_function(projected_hidden_states, sd_text_embeddings)
                    
                if count_system_have_sticker > 0:
                    mse_loss = mse_loss / count_system_have_sticker # mse_loss / len(caption)
                else:
                    mse_loss = 0
                    
            else:
                image_start_position = (labels == self.image_tokens_ids[0]).nonzero(as_tuple=False)[:, 1].tolist()
                image_end_position = (labels == self.image_tokens_ids[-1]).nonzero(as_tuple=False)[:, 1].tolist()
                
                hidden_states_list = []
                for b, (start, end) in enumerate(zip(image_start_position, image_end_position)):
                    hidden_states_list.append(outputs.hidden_states[-1][b, start: end + 1, :])
                hidden_states = torch.stack(hidden_states_list, dim=0)  # torch.Size([batch size, num_image_tokens, 4096])
                projected_hidden_states = self.outputProjection(hidden_states)  # torch.Size([batch size, 77, 1024])

                with torch.no_grad():
                    sd_text_embeddings, _ = self.stable_diffusion.encode_prompt(
                        prompt=emo_text[:self.max_text_length] if emo_text is not None else text ,
                        do_classifier_free_guidance=False,  # True
                        num_images_per_prompt=1,
                        device=self.llm_model.device
                    )
                mse_loss = mse_loss_function(projected_hidden_states, sd_text_embeddings)
                mse_loss = mse_loss.mean()

            gen_loss = gen_factor * mse_loss
            logging.info(f"gen_loss:{gen_loss}")
            loss += gen_loss
            
        logging.info(f"loss:{loss}")
        
        return PegsOutput(
                loss=loss,
                lm_loss=lm_loss,
                ret_loss=ret_loss,
                gen_loss=gen_loss
            )
        
    @torch.no_grad()
    def generate(
        self, 
        pixel_values: Optional[torch.FloatTensor] = None,
        text: Optional[str] = None,
        seen_image_idx: list = [],
        max_new_tokens: int = 512,
        do_sample: bool = True,
        min_length: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        num_inference_steps: int = 50,\
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        one_by_one: bool = True,
        eval: bool = False,  # whether to force generation of [IMGr]
        only_ret: bool = False,  # control whether to generate
        multi_img: bool = False,  # whether to retrieval Multiple images
        which_prompt: str = "gen",
    ):
        # stop word setting
        stop_words_ids = [torch.tensor([835]).to(self.llm_model.device),          # '###' can be encoded in two different ways.
                          torch.tensor([2277, 29937]).to(self.llm_model.device),
                          torch.tensor([self.image_tokens_ids[-1]]).to(self.llm_model.device),
                          torch.tensor([2]).to(self.llm_model.device),  # "</s>"
                          torch.tensor([13]).to(self.llm_model.device),  # "\n"
                          ]  # {"###": 835, "##": 2277, "#": 29937}
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        # build the suitable form for the data
        _, input_embeds, input_attention_mask, labels = self.build_one_instance_for_demo(pixel_values, text)
        
        if one_by_one:
            output, output_embeddings = self.generate_tokens_embeddings(
                input_embeds=input_embeds, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eval=eval            
            )
            output = output[0]
            output_embeddings = output_embeddings[-1]
        else:
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
            output = outputs.sequences[0][1:]
            output_hidden_states = []
            print(f'outputs.hidden_states:{len(list(outputs.hidden_states))}')
            for _hidden_states in outputs.hidden_states:
                output_hidden_states.append(_hidden_states[-1])
            output_embeddings = torch.cat(output_hidden_states, dim=1)
            
        generated_image = None
        
        response = self.tokenizer.decode(
            output,
            add_special_tokens=False,
            skip_special_tokens=True
        )
        
        print(f'out:{output}, response:{response}')
        # check if the output contains image tokens
        if set(self.image_tokens_ids).issubset(set(output.tolist())):  # need Retrieval and generation
            print('@@@ Retrieval and Generation...')
            # retrieval an image
            ret_img, _, gen_hidden_states = self.retrieval_image(
                seen_image_idx=seen_image_idx, labels=output, hidden_states=output_embeddings, multi_img=multi_img)
            if only_ret:
                generated_image = ret_img
            else:
                if isinstance(ret_img, list):
                    generated_image = []
                    for img in ret_img:
                        img = img.convert("RGB")
                        pixel_values = self.vision_processor(img)
                        logging.info(f'gen_hidden_states:{gen_hidden_states.unsqueeze(0)}, shape:{gen_hidden_states.unsqueeze(0).shape}')
                        # encode hidden states of <IMG> for generate as prompt_embedding
                        gen_emb = self.outputProjection(gen_hidden_states.unsqueeze(0))
                        if len(prompt) > 5:
                            prompt_embed = self.text_encoder(self.sd_tokenizer(prompt[:self.num_clip_tokens],
                                                            return_tensors="pt").input_ids.to(self.llm_model.device))[0]
                            logging.info(f'prompt:{prompt}')
                            prompt_embeds = torch.cat((prompt_embed, gen_emb), dim=-2)[:, :self.num_clip_tokens, :]
                            outputs = self.stable_diffusion(
                                prompt_embeds=prompt_embeds,
                                negative_prompt=negative_prompt,
                                image=pixel_values,
                                num_inference_steps=num_inference_steps,
                                strength=strength,
                                guidance_scale=guidance_scale,
                            )
                        else:
                            outputs = self.stable_diffusion(
                                prompt_embeds=gen_emb,
                                negative_prompt=negative_prompt,
                                image=pixel_values,
                                num_inference_steps=num_inference_steps,
                                strength=strength,
                                guidance_scale=guidance_scale,
                            )
                        generated_image.append(outputs.images[0])
                else:
                    ret_img = ret_img.convert("RGB")
                    # pixel_values = self.vision_processor(ret_img)
                    pixel_values = ret_img
                    # encode hidden states of <IMG> for generate as prompt_embedding
                    gen_emb = self.outputProjection(gen_hidden_states.unsqueeze(0))
                    
                    logging.info(f"negative_prompt = \n{negative_prompt}")
                    if "gen" in which_prompt:
                        outputs = self.stable_diffusion(
                            prompt_embeds=gen_emb,
                            negative_prompt=negative_prompt,
                            image=pixel_values,
                            num_inference_steps=num_inference_steps,
                            strength=strength,
                            guidance_scale=guidance_scale,
                        )
                    elif "manual" in which_prompt:
                        logging.info(f'prompt:{prompt}')
                        outputs = self.stable_diffusion(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=pixel_values,
                            num_inference_steps=num_inference_steps,
                            strength=strength,
                            guidance_scale=guidance_scale,
                        )
                    elif "trunc" in which_prompt:
                        prompt_embed = self.text_encoder(self.sd_tokenizer(prompt[:self.num_clip_tokens],
                                                                           return_tensors="pt").input_ids.to(self.llm_model.device))[0]
                        logging.info(f'prompt:{prompt}')
                        logging.info(f'prompt_embed:{prompt_embed.shape}')
                        logging.info(f'gen_emb:{gen_emb.shape}')
                        prompt_embeds = torch.cat((prompt_embed, gen_emb), dim=-2)[:, :self.num_clip_tokens, :]
                        logging.info(f'prompt_embeds:{prompt_embeds.shape}')
                        outputs = self.stable_diffusion(
                            prompt_embeds=prompt_embeds,
                            negative_prompt=negative_prompt,
                            image=pixel_values,
                            num_inference_steps=num_inference_steps,
                            strength=strength,
                            guidance_scale=guidance_scale,
                        )
                    elif "pooling" in which_prompt:
                        prompt_embed = self.text_encoder(self.sd_tokenizer(prompt[:self.num_clip_tokens],
                                                                           return_tensors="pt").input_ids.to(self.llm_model.device))[0]
                        logging.info(f'prompt:{prompt}')
                        exp_length = self.num_clip_tokens - prompt_embed.size(1)
                        step_size = gen_emb.size(1) // exp_length
                        pooled_gen_emb = torch.stack([torch.mean(gen_emb[:, i*step_size:(i+1)*step_size, :], dim=1) for i in range(exp_length)], dim=1)
                        prompt_embeds = torch.cat((prompt_embed, pooled_gen_emb), dim=-2)[:, :self.num_clip_tokens, :]
                        outputs = self.stable_diffusion(
                            prompt_embeds=prompt_embeds,
                            negative_prompt=negative_prompt,
                            image=pixel_values,
                            num_inference_steps=num_inference_steps,
                            strength=strength,
                            guidance_scale=guidance_scale,
                        )
                    
                    generated_image = outputs.images[0]
                
        return (
            output,
            generated_image
        )
        
     