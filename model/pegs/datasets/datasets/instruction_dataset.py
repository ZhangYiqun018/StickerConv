import os
from PIL import Image
import torch

from pegs.datasets.datasets.base_dataset import BaseDataset


class InstructionDataset(BaseDataset):
    def __init__(self, vision_processor, text_processor, images_root, annotation_path):
        super().__init__(vision_processor, text_processor, images_root, annotation_path)
        
    def __getitem__(self, index):
        data = self.annotation[index]
        
        text = data["text"]
        caption = data["caption"]
        
        pixel_values_list = []
        for image in data["image"]:
            image_path = os.path.join(self.images_root, image)
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.vision_processor(image)
            pixel_values_list.append(pixel_values)
        
        stacked_pixel_values = torch.stack(pixel_values_list)
        
        return {
            "text": text,
            "pixel_values": stacked_pixel_values,
            "caption": caption
        }
    
    def collate_fn(self, batch):
        text = [instance["text"] for instance in batch]
        pixel_values = [instance["pixel_values"] for instance in batch]
        caption = [instance["caption"] for instance in batch]
        
        return {
            "text": text,
            "pixel_values": pixel_values,
            "caption": caption
        }