from tqdm.auto import tqdm
from empathy import EmpathyEvaluator
from consistency import ConsistencyEvaluator
from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
import argparse
import concurrent.futures
import json
import os

parse = argparse.ArgumentParser(description="the parameters of llm-score")
parse.add_argument("--n", default=5, type =int)
parse.add_argument('--data_path', default="/datas/zyq/research/chat_meme/evaluates/outputs/baseline/vicuna_predict.json", type=str)
parse.add_argument("--save_path", default="./result", type=str)
args = parse.parse_args()

llm = ChatOpenAI(
    n     = args.n,
    model = "gpt-3.5-turbo"
)

verbose: bool = False

dataset = load_dataset(
    'json', data_files=args.data_path, split = 'train'
)

consistency_evaluater = ConsistencyEvaluator(
    llm     = llm,
    verbose = verbose
)
empathy_evaluater = EmpathyEvaluator(
    llm     = llm,
    verbose = verbose
)

def get_score(idx: int, data: dict):
    result = {"idx": idx}
    result.update(consistency_evaluater.compute_consistency_score(data, consider_user_sticker=False))
    result.update(empathy_evaluater.compute_empathy_score(data, consider_user_sticker=False))
    return result

results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    futures = [
        executor.submit(
            get_score,
            idx, 
            data
        ) for idx, data in enumerate(dataset)   
    ]
    for future in tqdm(concurrent.futures.as_completed(futures), total = len(dataset)):
        results.append(future.result())

results = sorted(results, key = lambda x: x['idx'])

empathy = []
empathy_txt = []
empathy_stc = []

consistency = []
consistency_txt = []
consistency_stc = []

for result in results:
    empathy.append(result['empathy_score_mean'])
    consistency.append(result['consistency_score_mean'])
    
    if result['empathy_score_text'] > 0:
        empathy_txt.append(result['empathy_score_text'])
    if result['empathy_score_sticker'] > 0:
        empathy_stc.append(result['empathy_score_sticker'])
    if result['consistency_score_text'] > 0:
        consistency_txt.append(result['consistency_score_text'])
    if result['consistency_score_sticker'] > 0:
        consistency_stc.append(result['consistency_score_sticker'])

empathy_txt_mean = sum(empathy_txt) / (len(empathy_txt) if len(empathy_txt) > 0 else 1)
empathy_stc_mean = sum(empathy_stc) / (len(empathy_stc) if len(empathy_stc) > 0 else 1)
consistency_txt_mean = sum(consistency_txt) / (len(empathy_txt) if len(empathy_txt) > 0 else 1)
consistency_stc_mean = sum(consistency_stc) / (len(empathy_stc) if len(empathy_stc) > 0 else 1)

results.insert(0, {
    "empathy_mean"             : sum(empathy)/len(empathy),
    "empathy_txt_mean"         : empathy_txt_mean,
    "empathy_stc_mean"         : empathy_stc_mean,
    "consistency_mean"         : sum(consistency) / len(consistency),
    "consistency_txt_mean"     : consistency_txt_mean,
    "consistency_stc_mean"     : consistency_stc_mean,
    "total_cost"               : consistency_evaluater.total_cost + empathy_evaluater.total_cost
})

with open(os.path.join(args.save_path, f"{args.data_path.split('/')[-1].split('.')[0]}_llm_score.json"), 'w') as w:
    json.dump(results, w, indent = 4)