from PIL import Image
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
import torch
import os
import json
import argparse

model_type="openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_type)
clip_tokenizer = CLIPTokenizer.from_pretrained(model_type)
clip_model = CLIPModel.from_pretrained(model_type)
vision_model = clip_model.vision_model
text_model = clip_model.text_model

def get_sticker_freq(dataset):
    count = 0
    for data in dataset:
        if len(data['pred_image']) > 0:
            count += 1
    return count / len(dataset) 

def get_visual_embeds(session_imgs):
    # process visual elements
    session_img_pil = [Image.open(img) for img in session_imgs]
    session_vision_inputs = processor(images=session_img_pil, return_tensors='pt')
    session_vision_outputs = vision_model(**session_vision_inputs)
    session_vision_embeds = session_vision_outputs[1]
    session_vision_embeds = clip_model.visual_projection(session_vision_embeds)
    session_vision_embeds = session_vision_embeds / session_vision_embeds.norm(p=2, dim=-1, keepdim=True)
    
    return session_vision_embeds

def get_text_embeds(session_utts):
    # process textual elements
    session_text_inputs = processor.tokenizer(session_utts, padding=True, truncation=True, return_tensors="pt")
    session_text_outputs = text_model(**session_text_inputs)
    session_text_embeds = session_text_outputs[1]
    session_text_embeds = clip_model.text_projection(session_text_embeds)
    session_text_embeds = session_text_embeds / session_text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    return session_text_embeds

def calculate_similarity(emb1, emb2, logit_scale):
    """计算两个嵌入向量的相似性得分"""
    if emb1 is not None and emb2 is not None:
        score = (torch.matmul(emb1.unsqueeze(0), emb2.unsqueeze(1)).squeeze() * logit_scale).item()
        return score
    return 0

def get_relevance_score(hyps: dict, refs: dict):
    logit_scale = clip_model.logit_scale.exp()
    text_emb_hyps = get_text_embeds(hyps['text']).squeeze()
    visual_emb_hyps = get_visual_embeds([hyps['image']]).squeeze() if hyps['image'] is not None else None
    
    text_emb_refs = get_text_embeds(refs['text']).squeeze()
    visual_emb_refs = get_visual_embeds([refs['image']]).squeeze() if refs['image'] is not None else None
    
    # 计算相似性得分
    text_text_similarity = calculate_similarity(text_emb_hyps, text_emb_refs, logit_scale)
    visual_visual_similarity = calculate_similarity(visual_emb_hyps, visual_emb_refs, logit_scale)
    text_visual_similarity = calculate_similarity(text_emb_hyps, visual_emb_refs, logit_scale)
    visual_text_similarity = calculate_similarity(visual_emb_hyps, text_emb_refs, logit_scale)
    
    clip_score = calculate_similarity(visual_emb_hyps, text_emb_hyps, logit_scale)
    clip_score_gt = calculate_similarity(visual_emb_refs, text_emb_refs, logit_scale)

    total_similarity = text_text_similarity + visual_visual_similarity
    optimized_similarity = total_similarity + text_visual_similarity + visual_text_similarity
    
    pmm = total_similarity / 2
    rmm = total_similarity / 2
    
    flag = 4
    if visual_emb_hyps is None:
        pmm = total_similarity
        flag = 2
    if visual_emb_refs is None:
        rmm = total_similarity
        flag = 2
    if visual_emb_refs is None and visual_emb_hyps is None:
        flag = 1
            
    f1_mm = 2*pmm*rmm / (pmm + rmm)
    
    return {
        "recall_mm_relevance"   : rmm,
        "precision_mm_relevance": pmm,
        "f1_mm_relevance"       : f1_mm,
        "optimized_mm_relevance": optimized_similarity/flag,
        "clip_score"            : clip_score,
        "clip_score_gt"         : clip_score_gt
    }
    
def process(data):
    gt_msg = data['response_text']
    gt_sticker = data['response_image']
    
    pr_msg = data['pred_text_response']
    pr_sticker = data['pred_image']
    if pr_sticker is None:
        pr_sticker = []
    # print(pr_sticker)
    
    refs = dict(
        text = gt_msg.replace("<IMG>", "").strip(),
        # image = '/datas/llm_datasets/SER_Dataset/Images/' + gt_sticker[0] if len(gt_sticker) > 0 else None,
        image = '/datas/llm_datasets/' + gt_sticker[0] if len(gt_sticker) > 0 else None
    )
    hyps = dict(
        text = pr_msg.replace("<IMG>", "").strip(),
        image = pr_sticker[0] if len(pr_sticker) > 0 else None
    )
    score = get_relevance_score(hyps = hyps, refs = refs)
    return score

if __name__ == "__main__":
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_type", type=str, default="pegs")
    args = parser.parse_args()
    
    # please refer to source codes of CLIP in HuggingFace
    logit_scale = clip_model.logit_scale.exp()

    from datasets import load_dataset
    import concurrent.futures
    from tqdm.auto import tqdm
    
    dataset = load_dataset(
        'json', data_files = args.data_path, split = 'train'
    )
    print(dataset)
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(
            process, data            
        ) for data in dataset]

        for future in tqdm(concurrent.futures.as_completed(futures), total = len(dataset)):
            results.append(future.result())
    
    precision_mm = []
    recall_mm = []
    f1_mm = []
    optim_mm = []
    clip_score = []
    clip_score_gt = []
    for result in results:
        precision_mm.append(result['precision_mm_relevance'])
        recall_mm.append(result['recall_mm_relevance'])
        f1_mm.append(result['f1_mm_relevance'])
        optim_mm.append(result['optimized_mm_relevance'])
        if result['clip_score'] > 0:
            clip_score.append(result['clip_score'])
        if result['clip_score_gt'] > 0:
            clip_score_gt.append(result['clip_score_gt'])
    
        
    with open(f'./result/{args.data_type}_multimodal_score.json', 'w') as w:
        json.dump({
            "frequency"    : get_sticker_freq(dataset),
            "precision_mm" : sum(precision_mm) / len(precision_mm),
            "recall_mm"    : sum(recall_mm) / len(recall_mm),
            "f1_mm"        : sum(f1_mm) / len(f1_mm),
            "optim_mm"     : sum(optim_mm) / len(optim_mm),
            "clip_score"   : sum(clip_score) / len(clip_score),
            "clip_score_gt": sum(clip_score_gt) / len(clip_score_gt),
        }, w, indent=4)