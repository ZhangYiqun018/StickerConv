import json
from typing import Optional

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from utils import load_json


class BaseDataset(Dataset):
    def __init__(
        self, 
        vision_processor=None, 
        text_processor=None, 
        images_root: Optional[str] = None, 
        annotation_path: Optional[str] = None
    ):
        """
        images_root (string): Root directory of images (e.g. coco/images/)
        annotation_path (string): directory to store the annotation file
        """
        self.vision_processor = vision_processor
        self.text_processor = text_processor
        
        self.images_root = images_root
        if annotation_path is not None:
            self.annotation = load_json(annotation_path)

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)
    
