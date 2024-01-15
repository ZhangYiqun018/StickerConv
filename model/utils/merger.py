from omegaconf import OmegaConf, DictConfig
from typing import List


class MergedConfig(DictConfig):
    def __init__(self, content) -> None:
        super().__init__(content)
        
        
class MergedDatasetConfig(DictConfig):
    def __init__(self, content) -> None:
        super().__init__(content)
        

def merge_config(configs: List[DictConfig]):
    merged_config = configs[0]
    for config in configs:
        merged_config = OmegaConf.merge(merged_config, config)
    
    return MergedConfig(merged_config)


