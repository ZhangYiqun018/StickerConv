import json
from omegaconf import OmegaConf


def load_json(file_name: str):
    if isinstance(file_name, str) and file_name.endswith("json"):
        with open(file_name, 'r') as file:
            data = json.load(file)
    else:
        raise ValueError("The file path you passed in is not a json file path.")
    
    return data


def load_yaml(file_name: str):
    if isinstance(file_name, str) and file_name.endswith("yaml"):
        config = OmegaConf.load(file_name)
    else:
        raise ValueError("The file path you passed in is not a yaml file path.")
        
    return config
    