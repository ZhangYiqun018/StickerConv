import configparser
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from typing import List, Dict
import json
from collections import Counter
from datasets import load_dataset

def save_result(result: Dict, save_path: str) -> None:
    fp = open(save_path, 'a+')
    print(
        json.dumps(result), file=fp
    )
    
def get_llm(
    config: configparser.ConfigParser,
    type: str = 'azure',
    temperature: float = 0.1,
    # the max tokens model generate
    max_tokens: int = 500,
    # use cache
    cache: bool = False,
    # generate number
    n: int = 1
) -> ChatOpenAI:
    api_base = config.get(type, 'api_base')
    api_key = config.get(type, 'api_key')
    if type == 'azure':
        model =  AzureChatOpenAI(
            deployment_name    = 'gpt-35-turbo',
            openai_api_base    = api_base,
            openai_api_version = "2023-07-01-preview",
            openai_api_key     = api_key,
            temperature        = temperature,
            max_tokens         = max_tokens,
            n                  = n,
            cache              = cache
        )
    else:
        model = ChatOpenAI(
            model           = 'gpt-3.5-turbo',
            openai_api_base = api_base,
            openai_api_key  = api_key,
            temperature     = temperature,
            max_tokens      = max_tokens,
            n               = n,
            cache           = cache,
        )
            
    return model

def analysis(path: str, key: str):
    data = load_dataset(
        'json', data_files=path, split = 'train'
    )[key]
    
    return Counter(data)

    