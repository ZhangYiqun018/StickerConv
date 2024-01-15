import os

from pegs.datasets.builders.base_builder import BaseDatasetBuilder
from pegs.datasets.datasets.instruction_dataset import InstructionDataset
from pegs.register import registry


@registry.register_builder("instruction")
class InstructionDatasetBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstructionDataset

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
