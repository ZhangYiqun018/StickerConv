import os
from PIL import Image
import webdataset as wds

from pegs.datasets.datasets.base_dataset import BaseDataset


class LaionDataset(BaseDataset):
    def __init__(self, vision_processor, text_processor, storage_path):
        super().__init__(vision_processor=vision_processor, text_processor=text_processor)
        
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(urls=storage_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vision_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "pixel_values": sample[0],
            "text": self.text_processor(sample[1]["caption"]),
        }


class CommonImageTextDataset(BaseDataset):
    def __init__(self, vision_processor, text_processor, images_root, annotation_path):
        super().__init__(vision_processor, text_processor, images_root, annotation_path)
        
    def __getitem__(self, index):
        data = self.annotation[index]
        
        text = data["text"]
        image = data["image"]
        
        image_path = os.path.join(self.images_root, image)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.vision_processor(image)
        
        return {
            "text": text,
            "pixel_values": pixel_values,
        }