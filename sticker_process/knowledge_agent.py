from datasets import load_dataset
from typing import Dict
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .utils import save_result, get_llm
from langchain.pydantic_v1 import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser
import configparser
import tenacity
import concurrent.futures
import fire
from tqdm.auto import tqdm

# 把sticker的五条qa整理成有关于sticker的三条知识(description, emotion, recommender)
class KnowledgeAgent(BaseModel):
    llm: BaseLanguageModel
    verbose: bool = False
    prompt_path: str

    class Config:
        arbitrary_types_allowed = True
        
    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm     = self.llm,
            prompt  = prompt,
            verbose = self.verbose
        )
    
    def get_parser(self):
        response_schemas = [
            ResponseSchema(name = "description", type="string", description="a brief description of the sticker."),
            ResponseSchema(name = "emotion", type="string", description="a brief description of the sentiment, humor, irony and satire regarding stickers."),
            ResponseSchema(name = "recommendation", type="string", description="a brief recommendation for incorporating the sticker into casual conversations."),
        ]

        parser = StructuredOutputParser(response_schemas=response_schemas)
        autofix_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm)

        return parser, autofix_parser
    
    @tenacity.retry(
        wait = tenacity.wait_exponential(multiplier=1, min=60, max=200),
        stop = tenacity.stop_after_attempt(10),
        reraise = True
    )
    def generate_sticker_knowledge(self, sticker: Dict) -> str:
        parser, autofix_parser = self.get_parser()
        prompt = PromptTemplate.from_template(
            open(self.prompt_path, 'r').read()
        )

        sticker_info = ""
        for data in sticker['qa']:
            query = data['query'].split("\n")[-1]
            output = ' '.join(data['output'].split("\n"))

            sticker_info += f"QUESTION: {query}\nANSWER: {output}\n"

        response = self.chain(prompt = prompt).run(
            sticker_info = sticker_info,
            format_instructions = parser.get_format_instructions()
        )
        # auto fix
        response = autofix_parser.parse(response)
        
        return response

def run(sticker, agent: KnowledgeAgent):
    try:
        result = agent.generate_sticker_knowledge(sticker=sticker)
        sticker.update(result)
        return sticker
    except Exception as e:
        print(
            f"{sticker['image']}: {e}"
        )

def main(
    config_path: str = 'config_private.ini',
    prompt_path: str = './template/knowledge_queries.txt',
    data_path: str = "/datas/zyq/research/chat_meme/dataset/sticker/caption_sticker_example.json",
    save_path: str = "/datas/zyq/research/chat_meme/dataset/sticker/knowledge_sticker_example.json",
    temperature: float = 0.5,
    num_workers: int = 4,
):
    print("Knowledge Agent is started!"
        + "\nParameters: "
        + f"\nconfig path: {config_path}"
        + f"\nprompt path: {prompt_path}"
        + f"\ndata path: {data_path}"
        + f"\nsave path: {save_path}"
        + f"\ntemperature: {temperature}"
        + f"\napi num workers: {num_workers}"   
    )
    config = configparser.ConfigParser()
    config.read(config_path)

    llm = get_llm(config, temperature=temperature)

    knwoledge_agent = KnowledgeAgent(llm = llm, verbose = False, prompt_path=prompt_path)

    data = load_dataset(
        'json', data_files=data_path, split = 'train'
    )

    pbar = tqdm(total = len(data))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for sticker in data:
            future = executor.submit(
                run,
                sticker,
                knwoledge_agent
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                save_result(result, save_path = save_path)
                pbar.update(1)


if __name__ == '__main__':
    fire.Fire(main)
    


    
