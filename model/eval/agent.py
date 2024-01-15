import os
import logging
from typing import Optional, List

from PIL import Image
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

import torch
from pegs.model.outputs import ConversationalAgentOutput
from pegs.register import registry


class AgentForEval:
    def __init__(self, config: OmegaConf) -> None:
        self.config = config
        
        self.model = self.init_model()
        self.set_model_to_device()
        self._load_checkpoint(self.checkpoint_path)
        self.model.eval()
        
        self.build_processors()
        
        self.generated_images_storage = os.path.join(self.outputs_dir, "generated")
        os.makedirs(self.generated_images_storage, exist_ok=True)

    @property
    def model_config(self):
        return self.config.model
    
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
        if self.model_config.enable_generation:
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
    
    def respond(
        self,
        context: str,
        images: Optional[List[Image.Image]] = None,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        pixel_values_list = []
        for img in images:
            pixel_values = self.vision_processor(img)
            pixel_values_list.append(pixel_values)
        pixel_values = torch.stack(pixel_values_list) if len(pixel_values_list) > 0 else None
        
        output_sequence, generated_image = self.model.generate(
            pixel_values=pixel_values,
            text=context,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        response = self.model.tokenizer.decode(
            output_sequence,
            add_special_tokens=False,
            skip_special_tokens=True
        )
        response = response.split("[IMG1]")[0].split("###")[0]
        
        if generated_image is not None:
            generated_image_save_path = os.path.join(self.generated_images_storage, "{}.jpg".format(len(os.listdir(self.generated_images_storage))))
            generated_image.save(generated_image_save_path)
         
        return ConversationalAgentOutput(
            text_response=response,
            image=generated_image
        )
