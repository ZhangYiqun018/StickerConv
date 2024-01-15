from utils import load_image, load_pretrained_model
from llava_infer import infer
from filter_sticker import StickerDataset
import pandas as pd
from .llava.conversation import conv_templates
from tqdm.auto import tqdm
from datasets import load_dataset


def captionizer(model_path: str, vision_path: str, filter_path: str, prompt_path: str, save_path: str):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path  = model_path,
        vision_path = vision_path,
        load_8bit   = False,
    )

    labeled_querys = open(prompt_path, 'r').read().split('\n')

    image_set = load_dataset(
        'json',
        data_files = filter_path,
        split      = 'train'
    )
    
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    
    kwargs = {
        "do_sample"           : True,
        "temperature"         : 0.2,
        "top_p"               : 0.7,
        "max_new_tokens"      : 1024,
        "use_cache"           : True,
        "num_return_sequences": 1
    }
    
    for image in tqdm(image_set['image']):
        conv = conv_templates[conv_mode].copy()
        output = infer(
            model           = model,
            tokenizer       = tokenizer,
            image_processor = image_processor,
            querys          = labeled_querys.copy(),
            image           = image,
            conv            = conv,
            kwargs          = kwargs,
            verbose         = False,
            save_path       = save_path
        )
        
if __name__ == '__main__':
    model_path = "/datas/huggingface/llava-1.5-13b"
    vision_path = "/datas/huggingface/clip-vit-large-patch14-336"
    filter_path = "./dataset/captionizer/filter_sticker.json"
    prompt_path = './template/caption_queries.txt'
    save_path = "../dataset/captionizer/caption_sticker.json"
    captionizer(
        model_path, vision_path, filter_path, prompt_path, save_path
    )