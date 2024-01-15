from datasets import load_dataset
import json

def make_prompt(input: str, image: str, history: str, image_token: str = "<IMG>", prefix: str = "### Human: ", postfix: str = "### Assistant:"):
    if image is not None:
        input = f"{input}\n{image_token}"
    if history is None:
        return f"{prefix}{input}\n{postfix}"
    else:
        return f"{history}\n{prefix}{input}\n{postfix}"

def convert_format(input_path: str, output_path: str) -> str:
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
                input = usr_msg,
                image = usr_img, 
                history = history,
            )
            # 顺序不能错
            if sys_img is not None:
                sys_msg = f"{sys_msg}\n<IMG>\n"
                
            history = prompt + ' ' + sys_msg
            
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
            
    with open(output_path, 'w') as w:
        json.dump(full_data, w, indent = 4)
        
    return output_path