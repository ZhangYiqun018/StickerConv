import os
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.backends.cudnn as cudnn

from utils import load_yaml, merge_config, init_logger, get_rank
from eval.agent import AgentForEval
from datasets import load_dataset

MODEL_CONFIG = "/datas/llm_datasets/kfhcode/memechat/pagi/configs/pagi.yaml"
IMAGE_ROOT = "/datas/llm_datasets/SER_Dataset/Images"
SAVE_ROOT = "/datas/zyq/research/chat_meme/evaluates/outputs"
SAVE_IMAGE_ROOT = "/datas/zyq/research/chat_meme/evaluates/outputs/sticker"

os.makedirs(SAVE_IMAGE_ROOT, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,  default="/datas/llm_datasets/kfhcode/memechat/eval/eval_config.yaml")
    parser.add_argument("--data_path", type=str, default = "/datas/llm_datasets/kfhcode/memechat/eval_example.json")
    parser.add_argument("--output_path", type = str, default="/datas/zyq/research/chat_meme/evaluates/outputs/1_11_prediction_100.json")
    args = parser.parse_args()
    
    return args

def setup_seeds(config):
    seed = config.run.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main(args):
    # config
    eval_config = load_yaml(args.config)
    model_config = load_yaml(MODEL_CONFIG)
    config = merge_config([eval_config, model_config])
    
    # seed  [Required]
    setup_seeds(config)
    # logging
    init_logger(config)
    
    agent = AgentForEval(config)
    
    examples = load_dataset(
        'json', data_files=args.data_path, split = 'train'
    ).select(range(100))

    data = []
    for instance in tqdm(examples):
        images = []
        for image_path in instance["input_image"]:
            image =  Image.open(os.path.join(IMAGE_ROOT, image_path))
            images.append(image)
        
        outputs = agent.respond(
            context             = instance["input_text"],
            images              = images,
            do_sample           = False,
            top_p               = 1.0,
            temperature         = 1.0,
            num_inference_steps = 50,
            guidance_scale      = 7.5,
        )
        
        pred_text_reponse = outputs.text_response
        instance["pred_text_response"] = pred_text_reponse
        
        pred_image = outputs.image
        if pred_image is not None:
            pred_image_path = os.path.join(SAVE_IMAGE_ROOT, f"{len(os.listdir(SAVE_IMAGE_ROOT))}.jpg")
            pred_image.save(pred_image_path)
            instance["pred_image"] = [pred_image_path]
        else:
            instance["pred_image"] = []
        
        data.append(instance)
        
    with open(args.output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
       

if __name__ == "__main__":
    args = parse_args()

    main(args)

# CUDA_VISIBLE_DEVICES=2 python generate_response.py