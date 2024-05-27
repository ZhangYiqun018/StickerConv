from sticker_process.eval import get_image_knowledge
from datasets import load_dataset
import json
from langchain.schema.language_model import BaseLanguageModel
from tqdm.auto import tqdm
from sticker_process.utils import load_pretrained_model
# function: 给模型生成的图片打标签
# key： pred_image: list

MODEL_PATH = "/datas/huggingface/llava-1.5-13b"
VISION_PATH = "/datas/huggingface/clip-vit-large-patch14-336"
LLAVA_PROMPT_PATH = "/datas/zyq/research/chat_meme/sticker_process/template/caption_queries.txt"
GPT_PROMPT_PATH = "/datas/zyq/research/chat_meme/sticker_process/template/knowledge_queries.txt"


def process_sticker(input_path: str, output_path: str, llm: BaseLanguageModel, start: int, end: int):
    dataset = load_dataset(
        'json',
        data_files = input_path,
        split = 'train'
    ).select(range(start, end))
    print(dataset)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path  = MODEL_PATH,
        vision_path = VISION_PATH,
        load_8bit   = False,
    )
    
    results = []
    for data in tqdm(dataset):
        if len(data['pred_image']) > 0:
            knowledge = get_image_knowledge(
                model = model, tokenizer = tokenizer, image_processor=image_processor,
                llm = llm,
                image = data['pred_image'][0]
            )
            data['pred_image_metadata'] = knowledge
        else:
            data['pred_image_metadata'] = None
        results.append(data)
    
    # save result
    with open(output_path, 'w') as w:
        json.dump(results, w, indent = 4)