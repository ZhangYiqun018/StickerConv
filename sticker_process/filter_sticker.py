import json
from .utils import load_image, load_pretrained_model
from IPython import display
from typing import List, Dict
import glob
import os
from tqdm.auto import tqdm
from .llava.conversation import conv_templates
from .llava_infer import infer
import tenacity

# load origin sticker dataset
class StickerDataset:
    def __init__(self, root_path: str) -> None:
        self.sticker_files = self.get_files(root_path)
    
    def get_files(self, root_path) -> List:
        file_names = glob.glob(os.path.join(root_path, '*'))

        files = []
        for file_name in file_names:
            file = glob.glob(os.path.join(file_name, '*.jpg'))
            files.extend(file)
        return files

    def __len__(self) -> int:
        return len(self.sticker_files)
    
    def __getitem__(self, idx) -> str:
        return self.sticker_files[idx]

    def show_sticker(self, idx):
        sticker = self.sticker_files[idx]
        sticker = load_image(sticker)
        display(sticker) 

def get_result(idx: str, results: List[Dict]) -> Dict:
    suitable = 0
    unsuitable = 0
    unknown = 0
    for result in results:
        output = result["qa"][idx-1]["output"] 
        if postprocess(output) == 1:
            suitable += 1
        elif postprocess(output) == -1:
            unsuitable += 1
        else:
            unknown += 1
            
    return {
        "suit"      : suitable >= unsuitable,
        "suitable"  : suitable,
        "unsuitable": unsuitable,
        "unknown"   : unknown
    }

def postprocess(text: str) -> int:
    if text.startswith('yes') or text.startswith('Yes') or text.startswith('YES'):
        return 1
    elif text.startswith('no') or text.startswith('No') or text.startswith('NO'):
        return -1
    else:
        return 0

@tenacity.retry(tenacity.stop_after_attempt(10), tenacity.wait_exponential(multiplier=1, min=1, max=120))
def filter(model_path: str, vision_path: str, data_path: str, prompt_path: str, save_path: str, num_returns: int = 5):
    image_set = StickerDataset(root_path=data_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path  = model_path,
        vision_path = vision_path,
        load_8bit   = False,
    )
    
    conv_mode = "llava_v1"
    
    kwargs = {
        "do_sample"           : True,
        "temperature"         : 0.2,
        "top_p"               : 0.7,
        "max_new_tokens"      : 1024,
        "use_cache"           : True,
        "num_return_sequences": 1
    }

    filter_query = open(prompt_path, 'r').read().split("\n")

    num_returns = num_returns
    
    suit_count = 0
    total_count = 0
    pbar = tqdm(total=num_returns * len(image_set), desc="Filter images")
    for image in image_set:
        outputs = []
        for i in range(num_returns):
            conv = conv_templates[conv_mode].copy()
            try:
                output = infer(
                    model           = model,
                    tokenizer       = tokenizer,
                    image_processor = image_processor,
                    querys          = filter_query.copy(),
                    image           = image,
                    conv            = conv,
                    kwargs          = kwargs,
                    verbose         = False,
                    save_path       = ""
                )
                # assert output 
            except Exception as e:
                print(e)
                
            outputs.append(output)
            pbar.update(1)
            
        r = {"image": image}
        r.update(get_result(idx=3, results=outputs))

        if r["suit"]:
            suit_count += 1
        total_count += 1
        print(
            json.dumps(r), 
            file=open(save_path, 'a+')
        )
        # Update the progress bar
        pbar.set_postfix(suit_count=suit_count, total_count=total_count)

if __name__ == '__main__':
    model_path = "/datas/huggingface/llava-1.5-13b"
    vision_path = "/datas/huggingface/clip-vit-large-patch14-336"
    data_path = '/datas/llm_datasets/SER_Dataset/Images'
    prompt_path = './template/filter_queries.txt'
    save_path = "../dataset/filter_stickers_temp_test.json"
    filter(
        model_path, vision_path, data_path, prompt_path, save_path, num_returns=5
    )