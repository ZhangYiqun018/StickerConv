import logging

from utils.merger import MergedDatasetConfig
from pegs.processors import BaseProcessor
from pegs.register import registry


class BaseDatasetBuilder:
    train_dataset_class, eval_dataset_class = None, None

    def __init__(self, dataset_config: MergedDatasetConfig = None):
        super().__init__()

        self.config = dataset_config

        self.vision_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_dataset(self):
        logging.info("Building datasets...")
        dataset = self.build()  # dataset['train'/'val'/'test']

        return dataset

    def build_processors(self):
        vision_processor_config = self.config.get("vision_processor")
        text_processor_config = self.config.get("text_processor")

        if vision_processor_config is not None:
            vision_train_config = vision_processor_config.get("train")
            vision_eval_config = vision_processor_config.get("eval")

            self.vision_processors["train"] = self._build_processor_from_config(vision_train_config)
            self.vision_processors["eval"] = self._build_processor_from_config(vision_eval_config)

        if text_processor_config is not None:
            text_train_config = text_processor_config.get("train")
            text_eval_config = text_processor_config.get("eval")

            self.text_processors["train"] = self._build_processor_from_config(text_train_config)
            self.text_processors["eval"] = self._build_processor_from_config(text_eval_config)
    
    @staticmethod
    def _build_processor_from_config(config):
        return (
            registry.get_processor_class(config.name).from_config(config)
            if config is not None
            else None
        )