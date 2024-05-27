from datasets import load_dataset
import argparse
import openai
from tqdm.auto import tqdm
import concurrent.futures
import json
import os

SYSTEM_PROMPT = "You are an open-domain empathy dialog chatbot. You have been asked to small talk with humans."


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vicuna')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    return parser.parse_args()

def get_response(model: str, query: str, history: list = [], model_name: str = None) -> str:
    completion = openai.ChatCompletion.create(
        model       = model,
        messages    = make_messages(history=history, text = query, model_name=model_name),
        max_tokens  = 200,
        n           = 1,
        stop        = ["### Human", '### Assistant', 'ASSISTANT', 'USER'],
        temperature = 0.5
    )
    response = completion.choices[0].message.content
    return response.lstrip().strip()

def make_messages(history: list, text: str, model_name: str = None) -> list:
    result = []
    if "chatglm" in model_name:
        result.append({
            "role": "system", "content": SYSTEM_PROMPT
        })
    for (query, response) in history:
        result.append({
            "role": "user", "content": query.replace("<IMG>", "").strip()
        })
        result.append({
            "role": "assistant", "content": response.replace("<IMG>", "").strip()
        })
    result.append({
        "role": "user", "content": text.replace("<IMG>", "").strip()
    })
    return result

def process_data(idx, data, model_name):
    history = data['history']
    query = data['query']
    pr_response = get_response(model = 'vicuna', query = query, history = history, model_name=model_name)
    return {
        'idx'        : idx,
        'history'    : history,
        'query'      : query,
        'response_text': data['response_text'],
        'pred_text_response': pr_response
    }  

def get_baseline_response(
    api_base: str,
    api_key: str,
    model_name: str,
    input_path: str,
    output_path: str = "../outputs/baseline",
    max_workers: int = 4
):
    openai.api_base = api_base
    openai.api_key = api_key

    datas = load_dataset(
        "json", data_files=input_path, split = 'train'
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 futures 来保持结果的顺序
        futures = [executor.submit(process_data, idx, data, model_name) for idx, data in enumerate(datas)]
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(datas)):
            results.append(future.result())

    results = sorted(results, key = lambda x: x['idx'])

    with open(os.path.join(output_path, f'{model_name}_predict.json'), 'w') as w:
        json.dump(results, w, indent = 4)