import json
import argparse
import math
import os
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm
import pickle as pc

from transformers import AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from parlai.core.torch_classifier_agent import ConfusionMatrixMetric, WeightedF1Metric
from parlai.core.metrics import (
    AverageMetric, 
    InterDistinctMetric,
)
from modules.empathy_scorer import EmpathyScorer
from nltk import word_tokenize
import os
from datasets import load_dataset

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

def get_epitome_score(data, epitome_empathy_scorer):
    pred_IP_scores, pred_EX_scores, pred_ER_scores = [], [], []
    gt_IP_scores, gt_EX_scores, gt_ER_scores = [], [], []
    diff_IP_scores, diff_EX_scores, diff_ER_scores = [], [], []

    for example in tqdm(data):
        utter = example['utterance']
        pred = example['prediction']
        gt = example['gt']
        
        pred_epitome_score = epitome_empathy_scorer([utter], [pred])
        gt_epitome_score = epitome_empathy_scorer([utter], [gt])
        
        example['epitome-IP-pred'] = int(pred_epitome_score['IP'][0][0])
        example['epitome-EX-pred'] = int(pred_epitome_score['EX'][0][0])
        example['epitome-ER-pred'] = int(pred_epitome_score['ER'][0][0])

        example['epitome-IP-gt'] = int(gt_epitome_score['IP'][0][0])
        example['epitome-EX-gt'] = int(gt_epitome_score['EX'][0][0])
        example['epitome-ER-gt'] = int(gt_epitome_score['ER'][0][0])

        pred_IP_scores += pred_epitome_score['IP'][0]
        pred_EX_scores += pred_epitome_score['EX'][0]
        pred_ER_scores += pred_epitome_score['ER'][0]
        
        gt_IP_scores += gt_epitome_score['IP'][0]
        gt_EX_scores += gt_epitome_score['EX'][0]
        gt_ER_scores += gt_epitome_score['ER'][0]

        diff_IP_scores.append(math.pow(abs(pred_epitome_score['IP'][0][0] - gt_epitome_score['IP'][0][0]), 2))
        diff_EX_scores.append(math.pow(abs(pred_epitome_score['EX'][0][0] - gt_epitome_score['EX'][0][0]), 2))
        diff_ER_scores.append(math.pow(abs(pred_epitome_score['ER'][0][0] - gt_epitome_score['ER'][0][0]), 2))
        
    return data, pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores


def convert_input_file_to_tensor_dataset(lines,
                                         args,
                                         tokenizer,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    
    for line in lines:
        tokens = tokenizer.tokenize(line)
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_length - special_tokens_count:
            tokens = tokens[:(args.max_seq_length - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # for interpret functions
        ref_ids = [input_ids[0]] + [pad_token_id] * len(input_ids[1:-1]) + [input_ids[-1]]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

    #dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    dataset = {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'token_type_ids': all_token_type_ids,
    }
    
    return dataset

def get_intra_distinct(data):

    pred_dist_1, pred_dist_2 = [], []
    gold_dist_1, gold_dist_2 = [], []

    for example in data:
        pred = example['prediction']
        gold = example['gt']

        pred_m1 = InterDistinctMetric.compute(pred, 1)
        pred_m2 = InterDistinctMetric.compute(pred, 2)

        gold_m1 = InterDistinctMetric.compute(gold, 1)
        gold_m2 = InterDistinctMetric.compute(gold, 2)
        
        pred_dist_1.append(float(pred_m1))
        pred_dist_2.append(float(pred_m2))
        gold_dist_1.append(float(gold_m1))
        gold_dist_2.append(float(gold_m2))
    
    avg_pred_dist_1 = AverageMetric(sum(pred_dist_1), len(pred_dist_1))
    avg_pred_dist_2 = AverageMetric(sum(pred_dist_2), len(pred_dist_2))
    avg_gold_dist_1 = AverageMetric(sum(gold_dist_1), len(gold_dist_1))
    avg_gold_dist_2 = AverageMetric(sum(gold_dist_2), len(gold_dist_2))

    return avg_pred_dist_1, avg_pred_dist_2, avg_gold_dist_1, avg_gold_dist_2

def get_inter_distinct(data):

    pred_dist_1, pred_dist_2 = [], []
    gold_dist_1, gold_dist_2 = [], []

    for example in data:
        pred = example['prediction']
        gold = example['gt']

        pred_m1 = InterDistinctMetric.compute(pred, 1)
        pred_m2 = InterDistinctMetric.compute(pred, 2)

        gold_m1 = InterDistinctMetric.compute(gold, 1)
        gold_m2 = InterDistinctMetric.compute(gold, 2)
        
        pred_dist_1.append(float(pred_m1))
        pred_dist_2.append(float(pred_m2))
        gold_dist_1.append(float(gold_m1))
        gold_dist_2.append(float(gold_m2))
    
    avg_pred_dist_1 = AverageMetric(sum(pred_dist_1), len(pred_dist_1))
    avg_pred_dist_2 = AverageMetric(sum(pred_dist_2), len(pred_dist_2))
    avg_gold_dist_1 = AverageMetric(sum(gold_dist_1), len(gold_dist_1))
    avg_gold_dist_2 = AverageMetric(sum(gold_dist_2), len(gold_dist_2))

    return avg_pred_dist_1, avg_pred_dist_2, avg_gold_dist_1, avg_gold_dist_2

def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config['no_cuda'] else "cpu"

def get_length(data):
    length_results = [len(word_tokenize(sent)) for sent in data]
    return np.average(length_results)

def get_ngrams(resp, n):
    tokens = resp.split()
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]

def get_ngram_counter(resp, n):
    ngrams = get_ngrams(resp, n)
    counter = Counter()
    counter.update(ngrams)
    return counter

def _distinct_n(data, n):
    dist_results = []
    for sent in data:
        ngram_counter = get_ngram_counter(sent.strip().lower(), n)

        if sum(ngram_counter.values()) == 0:
            print("Warning: encountered a response with no {}-grams".format(n))
            print(sent.strip().lower())
            print("ngram_counter: ", ngram_counter)
            continue
            
        dist = len(ngram_counter) / sum(ngram_counter.values())
        dist_results.append(dist)
    
    return np.average(dist_results)

def evaluate_dist_n(data, n):
    return _distinct_n(data, n)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datatype', type=str, default='baseline')
    parser.add_argument('--data_dir', type=str, default='/datas/zyq/research/chat_meme/evaluate/vicuna_result.json')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--evaluation_save_dir', type=str, default='./result')
    parser.add_argument('--epitome_save_dir', type=str, default='/datas/zyq/research/EPITOME/epitome_model')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    
    # load generated result from EmpGPT-3
    # you can also evaluate the generated result from Blender
    
    # load data
    generation = load_dataset(
        'json', data_files=args.data_dir, split='train' 
    )
    
    results = []
    for example in generation:
        utter = example['query']
        pred_resp = example['pred_text_response'].replace('<IMG>', '').strip()
        gold_resp = example['response_text'].replace('<IMG>', '').strip()

        pred_resp = example['pred_text_response'].strip()
        gold_resp = example['response_text'].strip()
        
        result = {
            'utterance': utter.lower(),
            'prediction': pred_resp.lower(),
            'gt': gold_resp.lower(),
        }
        results.append(result)
        
    opt = {}
    opt['no_cuda'] = False
    
    device = "cuda"
    opt['epitome_save_dir'] = args.epitome_save_dir
    epitome_empathy_scorer = EmpathyScorer(opt, batch_size=args.batch_size, cuda_device=device, use_cuda=opt['no_cuda'])

    results, pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores = get_epitome_score(results, epitome_empathy_scorer)

    # save evaluation result
    result_save_dir = args.evaluation_save_dir
    os.makedirs(result_save_dir, exist_ok=True)
    with open(os.path.join(result_save_dir, f'{args.datatype}_results.pkl'), 'wb') as f:
        pc.dump(results, f)

    _report = {}
    
    _report['pred_IP'] = AverageMetric(sum(pred_IP_scores), len(pred_IP_scores))
    _report['pred_EX'] = AverageMetric(sum(pred_EX_scores), len(pred_EX_scores))
    _report['pred_ER'] = AverageMetric(sum(pred_ER_scores), len(pred_ER_scores))

    _report['gt_IP'] = AverageMetric(sum(gt_IP_scores), len(gt_IP_scores))
    _report['gt_EX'] = AverageMetric(sum(gt_EX_scores), len(gt_EX_scores))
    _report['gt_ER'] = AverageMetric(sum(gt_ER_scores), len(gt_ER_scores))

    _report['diff_IP'] = AverageMetric(sum(diff_IP_scores), len(diff_IP_scores))
    _report['diff_EX'] = AverageMetric(sum(diff_EX_scores), len(diff_EX_scores))
    _report['diff_ER'] = AverageMetric(sum(diff_ER_scores), len(diff_ER_scores))
    
    only_pred = [example['prediction'] for example in results if example['prediction'] != '']
    only_gold = [example['gt'] for example in results]
    
    resp_typ = {
        'pred': only_pred,
        'gold': only_gold,
    }
    for typ, data in resp_typ.items():
        # Distinct-n
        _report[f'{typ}-dist1'] = evaluate_dist_n(data, 1)
        _report[f'{typ}-dist2'] = evaluate_dist_n(data, 2)
        _report[f'{typ}-dist3'] = evaluate_dist_n(data, 3)

    f = open(os.path.join(result_save_dir, f'{args.datatype}_eval_stat.txt'), 'w')
    for k, v in _report.items():
        f.write(k + ' : ' + str(float(v)) + '\n')
    
    f.close()