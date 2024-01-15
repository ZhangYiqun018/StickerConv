import os
from pegs.datasets.datasets.image_text_dataset import LaionDataset, CommonImageTextDataset
from pegs.datasets.builders.base_builder import BaseDatasetBuilder
from pegs.register import registry


@registry.register_builder("laion")
class LaionDatasetBuilder(BaseDatasetBuilder):
    train_dataset_class = LaionDataset

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        dataset = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_class = self.train_dataset_class
        dataset[split] = dataset_class(
            vision_processor=self.vision_processors[split],
            text_processor=self.text_processors[split],
            storage_path=build_info.storage,
        ).inner_dataset

        return dataset
    

# @registry.register_builder("common_image_text")
@registry.register_builder("sticker_text")
class CommonImageTextDatasetBuilder(BaseDatasetBuilder):
    train_dataset_cls = CommonImageTextDataset

    def build(self):
        self.build_processors()
        
        storage = self.config.storage
        images_root = storage.images
        annotation_path = storage.annotation

        datasets = dict()
        split = "train"

        if not os.path.exists(images_root):
            raise ValueError("storage path {} does not exist.".format(images_root))
        if not os.path.exists(annotation_path):
            raise ValueError("storage path {} does not exist.".format(annotation_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vision_processor=self.vision_processors[split],
            text_processor=self.text_processors[split],
            images_root=images_root,
            annotation_path=annotation_path
        )

        return datasets

