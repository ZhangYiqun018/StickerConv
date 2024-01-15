import os
import logging
from datetime import datetime

import gradio as gr
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

import torch
from pegs.register import registry


class ConversationalAgent:
    def __init__(self, config: OmegaConf) -> None:
        self.config = config
        
        self.model = self.init_model()
        self.set_model_to_device()
        self._load_checkpoint(self.checkpoint_path)
        self.model.eval()
        
        self.build_processors()
        self.uploaded_images_storage = os.path.join(self.outputs_dir, "uploaded")
        os.makedirs(self.uploaded_images_storage, exist_ok=True)
        
        self.generated_images_storage = os.path.join(self.outputs_dir, "generated")
        os.makedirs(self.generated_images_storage, exist_ok=True)
        
    @property
    def model_config(self):
        return self.config.model
    
    @property
    def datasets_config(self):
        return self.config.datasets
    
    @property
    def processors_config(self):
        return self.config.processors
    
    @property
    def run_config(self):
        return self.config.run
    
    @property
    def device(self):
        return self.run_config.device
    
    @property
    def checkpoint_path(self):
        return self.model_config.get("checkpoint", None)
    
    @property
    def outputs_dir(self):
        return self.run_config.outputs_dir
    
    def init_model(self):
        return registry.get_model_class(self.model_config.arch)(**self.model_config)
    
    def set_model_to_device(self):
        # move model to device
        self.model = self.model.to(self.device)
        if hasattr(self.model, 'stable_diffusion'):
            self.model.stable_diffusion.to(self.device)

    def _load_checkpoint(self, filename):
        """
        load trained model from a checkpoint/checkpoints.
        """
        logging.info(f"type(filename) = {type(filename)}")
        if isinstance(filename, str):
            if os.path.isfile(filename):
                checkpoint = torch.load(filename, map_location=self.device)
                state_dict = checkpoint["model"]
                self.model.load_state_dict(state_dict, strict=False)
            else:
                raise ValueError("checkpoint path is invalid!")
        elif isinstance(filename, ListConfig):
            for one_filename in filename:
                checkpoint = torch.load(one_filename, map_location=self.device)
                state_dict = checkpoint["model"]
                self.model.load_state_dict(state_dict, strict=False)
        logging.info("Load checkpoint from {}".format(filename))
    
    def build_processors(self):
        vision_processor_config = self.processors_config.get("vision_processor", None)
        text_processor_config = self.processors_config.get("text_processor", None)

        if vision_processor_config is not None and text_processor_config is not None:
            self.vision_processor = self._build_processor_from_config(vision_processor_config)
            self.text_processor = self._build_processor_from_config(text_processor_config)
        else:
            raise ValueError("Please set the processors config!")
        
    @staticmethod
    def _build_processor_from_config(config):
        return (
            registry.get_processor_class(config.name).from_config(config)
            if config is not None
            else None
        )
    
    def start_chat(self, chat_state):
        logging.info("=" * 30 + "Start Chat" + "=" * 30)
        chat_state.append(["", "", []])
        
        return (
            gr.update(interactive=True, placeholder='input the text.'),  # [input_text] Textbox
            gr.update(interactive=False),  # [start_btn] Button
            gr.update(interactive=True),  # [clear_btn] Button
            gr.update(interactive=True),  # [image] Image
            chat_state  # [chat_state] State
        )
        
    def restart_chat(self, chat_state):
        logging.info("=" * 30 + "End Chat" + "=" * 30)
        
        return (
            None,  # [chatbot] Chatbot
            gr.update(interactive=False, placeholder="Please click the <Start Chat> button to start chat!"),  # [input_text] Textbox
            gr.update(interactive=True),  # [start] Button
            gr.update(interactive=False),  # [clear] Button
            gr.update(value=None, interactive=False),  # [image] Image
            []  # [chat_state] State
        )
        
    def undo(self, chatbot, chat_state):
        logging.info("-" * 30 + "   Undo   " + "-" * 30)
        
        message, _ = chatbot.pop()
        chat_state.pop()
        
        logging.info(f"\nchatbot: {chatbot}")
        
        return message, chatbot, chat_state
    
    def respond(
        self,
        message,
        image,
        chat_history: gr.Chatbot,
        do_sample,
        top_p,
        temperature,
        num_inference_steps,
        guidance_scale,
        negativate_prompt,
        chat_state,
    ):
        current_time = datetime.now().strftime("%b%d-%H:%M:%S")
        logging.info(f"Time: {current_time}")
        logging.info(f"User: {message}")
        
        _, context, pixel_values_list = chat_state[-1]
        
        if image is not None:
            save_image_path = os.path.join(self.uploaded_images_storage, "{}.jpg".format(len(os.listdir(self.uploaded_images_storage))))
            image.save(save_image_path)
            logging.info(f"image save path: {save_image_path}")

            pixel_values = self.vision_processor(image)
            pixel_values_list.append(pixel_values)
            pixel_values = torch.stack(pixel_values_list)
            input_text = context + "### Human: " + message + "\n" + self.model_config.image_placeholder_token + "\n### Assistant:"
        else:
            pixel_values = torch.stack(pixel_values_list) if len(pixel_values_list) > 0 else None
            input_text = context + "### Human: " + message + "\n### Assistant:"
        
        logging.info(f"input_text = \n{input_text}")
        logging.info(f"generation settings\ndo_sample = {do_sample}\ntop_p={top_p}\ntemperature={temperature}")

        outputs = self.model.generate(
            pixel_values=pixel_values,
            text=input_text,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negativate_prompt=negativate_prompt,
        )
        output_sequence = outputs.output_sequence
        generated_image = outputs.image
        
        response = self.model.tokenizer.decode(
            output_sequence,
            add_special_tokens=False,
            skip_special_tokens=True
        )
        logging.info(f"generated text = \n{response}")
        response = response.split("[IMG1]")[0].split("###")[0]
        logging.info(f"response = \n{response}")
        
        if generated_image is not None:
            generated_image_save_path = os.path.join(self.generated_images_storage, "{}.jpg".format(len(os.listdir(self.generated_images_storage))))
            generated_image.save(generated_image_save_path)
            logging.info(f"generated image is saved to: {generated_image_save_path}")
            
            if image is not None:
                chat_history.append((
                    f'{message}\n<img src="./file={save_image_path}" style="display: inline-block;">', 
                    f'''{response}\n<img src="./file={generated_image_save_path}" style="display: inline-block;">'''
                ))
            else:
                chat_history.append((
                    message, 
                    f'''{response}\n<img src="./file={generated_image_save_path}" style="display: inline-block;">'''
                ))
        else:
            if image is not None:
                chat_history.append((f'{message}\n\n<img src="./file={save_image_path}" style="display: inline-block;">', response))
            else:
                chat_history.append((message, response))

        context = input_text + " " + response + ("" if response.endswith('\n') else "\n")
        chat_state.append([message, context, pixel_values_list])
         
        return "", None, chat_history, chat_state, gr.update(interactive=True)
