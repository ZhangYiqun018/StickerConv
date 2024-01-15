# process test image
# 1. get the qa pairs of generate sticker
# 2. get the knowledge of generate sticker and qa pairs.
from .utils import load_pretrained_model
from .llava_infer import infer
from .llava.conversation import conv_templates
from .knowledge_agent import KnowledgeAgent
import fire
from langchain.schema.language_model import BaseLanguageModel

MODEL_PATH = "/datas/huggingface/llava-1.5-13b"
VISION_PATH = "/datas/huggingface/clip-vit-large-patch14-336"
LLAVA_PROMPT_PATH = "/datas/zyq/research/chat_meme/sticker_process/template/caption_queries.txt"
GPT_PROMPT_PATH = "/datas/zyq/research/chat_meme/sticker_process/template/knowledge_queries.txt"


def process(model, tokenizer, image_processor, prompt_path: str, agent: KnowledgeAgent, image: str):
    queries = open(prompt_path, 'r').read().split('\n')
    
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
    # stage 1
    output = infer(
        model           = model,
        tokenizer       = tokenizer,
        image_processor = image_processor,
        querys          = queries.copy(),
        image           = image,
        conv            = conv,
        kwargs          = kwargs,
        verbose         = False,
        save_path       = None
    )
    # stage 2
    knowledge = agent.generate_sticker_knowledge(sticker=output)

    return knowledge

def get_image_knowledge(model, tokenizer, image_processor, llm: BaseLanguageModel, image: str = "/datas/zyq/research/chat_meme/evaluate/outputs/eval/generated/0.jpg"):
    knowledge_agent = KnowledgeAgent(
        llm         = llm,
        verbose     = False,
        prompt_path = GPT_PROMPT_PATH
    )

    result = process(
        model           = model,
        tokenizer       = tokenizer,
        image_processor = image_processor,
        prompt_path     = LLAVA_PROMPT_PATH,
        agent           = knowledge_agent,
        image           = image
    )
        
    return result


if __name__ == '__main__':
    fire.Fire(get_image_knowledge)