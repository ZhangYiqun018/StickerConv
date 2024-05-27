from datasets import load_dataset
import json
import argparse

SYSTEM_PROMPT = "You are an open-domain empathy dialog chatbot. You have been asked to small talk with humans."

def make_prompt(input: str, image: str, history: str, image_token: str = "<IMG>", prefix: str = "USER: ", postfix: str = "ASSISTANT:"):
    if image is not None:
        input = f"{input}{image_token}"
    if history is None:
        return f"{prefix}{input}\n{postfix}"
    else:
        return f"{history}</s>\n{prefix}{input}\n{postfix}"

def convert_format(input_path: str, output_path: str, prefix, postfix) -> str:
    dialogue_data = load_dataset(
        'json', data_files=input_path, split='train'
    )
    
    full_data = []
    for data in dialogue_data:
        conversation = data['conversation']
        persona = data['user_persona']
        status = data['user_status']
        history = None
        self_history = []
        input_image = []
        for conv in conversation:
            usr_msg = conv['user_message']
            usr_img = conv['user_sticker']
            sys_msg = conv['system_message']
            sys_img = conv['system_sticker']
            
            if usr_img is not None:
                input_image.append(usr_img['image'].replace("/datas/llm_datasets/SER_Dataset/Images/", ""))
            
            prompt = make_prompt(
                input   = usr_msg,
                image   = usr_img,
                history = history,
                prefix  = prefix,
                postfix = postfix
            )
            # 顺序不能错
            if sys_img is not None:
                sys_msg = f"{sys_msg}<IMG>"
                
            history = prompt + sys_msg
            
            temp = {
                "input_text"             : prompt,
                "input_image"            : input_image.copy(),
                "response_text"          : sys_msg,
                "response_image"         : [] if sys_img is None else [sys_img['image'].replace("/datas/llm_datasets/SER_Dataset/Images/", "")],
                "user_persona"           : persona,
                "user_status"            : status,
                "input_image_metadata"   : usr_img,
                "response_image_metadata": sys_img,
                "query"                  : usr_msg,
                "history"                : self_history.copy()
            }
            self_history.append(
                [usr_msg, sys_msg]
            )
            full_data.append(temp)
            if sys_img is not None:
                input_image.append(sys_img['image'].replace("/datas/llm_datasets/SER_Dataset/Images/", ""))
    
    print(full_data[2]['input_text'])
    with open(output_path, 'w') as w:
        json.dump(full_data, w, indent = 4)
        
    return output_path

# main
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type = str, default="1.5")
parser.add_argument("--input_path", type = str, default="/datas/zyq/research/chat_meme/dataset/dialogue/test.json")
parser.add_argument("--output_path", type = str, default="/datas/zyq/research/chat_meme/evaluates/util/pegs_1.5_test_gen.json")
args = parser.parse_args()

if args.model_type == "v0":
    prefix = "### Human: "
    postfix = "### Assistant:"
elif args.model_type == "1.5":
    prefix = "USER: "
    postfix = "ASSISTANT:"
else:
    raise ValueError("check the model type.")

if __name__ == '__main__':
    convert_format(input_path=args.input_path, output_path=args.output_path, prefix = prefix, postfix=postfix)

