import os
from PIL import Image
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
from typing import List, Dict
import json
from datasets import load_dataset, concatenate_datasets
import configparser
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
import random


def save_result(result: Dict, save_path: str) -> None:
    fp = open(save_path, 'a+')
    print(
        json.dumps(result), file=fp
    )

def switch_2_azure(config_path: str):
    temperature = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    config = configparser.ConfigParser()
    config.read(config_path)
    
    model = AzureChatOpenAI(
        deployment_name    = 'gpt-35-turbo',
        openai_api_base    = config.get('azure', 'api_base'),
        openai_api_version = "2023-07-01-preview",
        openai_api_key     = config.get('azure', 'api_key'),
        temperature        = temperature,
        max_tokens         = 300,
        n                  = 1,
    )
    return model

def switch_2_gpt4(config_path: str):
    temperature = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    config = configparser.ConfigParser()
    config.read(config_path)
    
    model = ChatOpenAI(
        model           = 'gpt-4-1106-preview',
        openai_api_base = config.get('oneapi', 'api_base'),
        openai_api_key  = config.get('oneapi', 'api_key'),
        temperature     = temperature,
        max_tokens      = 150,
        n               = 1,
    )
    return model

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

def preprocess(example):
    example['image'] = example['image'].replace("\\", "")
    return example

def convert_filter_dataset(
    file_names: List[str],
    output_name: str,
):
    """
    将过滤后的文本转化为新数据集
    """
    df = []
    for file in file_names:
        dataset = load_dataset(
            'json',
            data_files = file,
            split      = 'train',
        )
        dataset = dataset.filter(lambda x: x['suit'] == True)
        dataset = dataset.remove_columns(
            ['suit', 'suitable', 'unsuitable', 'unknown']
        )
        df.append(dataset)
    
    dataset = concatenate_datasets(df)
    print(dataset)
    dataset.to_csv(output_name, index=False)

# data analysis
def data_analysis(data_path: str):
    # load_data
    conversations = load_dataset('json', data_files=data_path, split='train')['conversation']
    tot_sticker_used = 0
    turns = 0
    sticker_list = []
    tot_data = len(conversations)
    tot_sentence_length = 0
    tot_user_sticker = 0
    tot_system_sticker = 0
    for conversation in conversations:
        turns += len(conversation)
        for conv in conversation:
            # TODO: other analysis data
            user_msg = conv['user_message']
            system_msg = conv['system_message']
            tot_sentence_length += len(user_msg.split(' ')) + len(system_msg.split(' ')) 
            # analysis sticker
            user_sticker = conv['user_sticker']
            system_sticker = conv['system_sticker']
            if user_sticker is not None:
                tot_sticker_used += 1
                tot_user_sticker += 1
                sticker_list.append(user_sticker['image'])
            if system_sticker is not None:
                tot_sticker_used += 1
                tot_system_sticker += 1
                sticker_list.append(system_sticker['image'])
    # total number 
    uni_sticker_used = len(list(set(sticker_list)))

    post = f"total number: {tot_data} - per conv turn: {turns/tot_data:.2f} - per sentence length: {tot_sentence_length/turns/2:.2f} - total sticker: {tot_sticker_used} - unique sticker: {uni_sticker_used} - per sticker turn: {tot_sticker_used/len(conversations) :.2f} - user sticker - {tot_user_sticker} - system_sticker - {tot_system_sticker}"

    return post
        
if __name__ == '__main__':
    print(
        data_analysis("../dataset/dialogue/new_method_test.json")
    )
