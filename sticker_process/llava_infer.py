from typing import List, Dict
from .llava.conversation import Conversation
import json
from typing import List
from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .llava.conversation import conv_templates, SeparatorStyle, Conversation
from .llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
import torch
from .utils import load_image

def infer(
    model,
    tokenizer,
    image_processor,
    querys: List, 
    image: str, 
    conv: Conversation, 
    kwargs: Dict, 
    save_path: str, 
    verbose: bool = False
):
    if isinstance(querys, str):
        querys = [querys]
        
    if model.config.mm_use_im_start_end:
        querys[0] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + querys[0]
    else:
        querys[0] = DEFAULT_IMAGE_TOKEN + '\n' + querys[0]
    
    result = {"image": image}
    
    image = load_image(image)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    
    qa = []
    for idx, query in enumerate(querys):
        # init user message
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images               = image_tensor.half().cuda(),
                stopping_criteria    = [stopping_criteria],
                **kwargs,
            )
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        # update user & system message
        conv.messages.pop()
        conv.append_message(conv.roles[1], outputs)
        # verbose
        if verbose:
            print(f"[{idx+1}/{len(querys)}]")
            print(f"[QUERY]: \n{query}")
            print(f"[ANSWER]: \n{outputs}")
        # save result
        current_result = {
            "idx": idx + 1,
            "query": query,
            "output": outputs
        }
        qa.append(current_result)
    
    result.update({"qa": qa})
    
    if save_path is not None and len(save_path) > 0:
        print(
            json.dumps(result), file=open(save_path, 'a+')
        )
    
    return result