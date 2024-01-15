# The `DataGenerator` class is responsible for generating diverse profiles by completing a series of
# prompts that include emotion, persona, and situation. It uses a language model to generate responses
# and applies post-processing to filter out profiles that have a high similarity to existing
# situations.
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
import random
import json
from rouge_score import rouge_scorer
from multiprocessing import Pool
from functools import partial
import re
from utils import save_result
import logging
import tenacity

logger = logging.getLogger()

class DataGenerator:
    def __init__(self, llm: BaseLanguageModel, seed_path: str, machine_generate_path: str, 
                 sample_number: int = 10, machine_sample_number: int = 2, verbose: bool = False, threshold: float = 0.7) -> None:
        self.llm = llm
        self.seed_path = seed_path
        self.generate_path = machine_generate_path
        self.sample_number = sample_number
        self.machine_sample_number = machine_sample_number
        self.verbose = verbose
        self.rouge_threshold = threshold
        # init 
        self.init_situation()
        
    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose,
        )
    
    def init_situation(self):
        seed_datas = open(f'{self.seed_path}', 'r').read().split("\n")[:-1]
        machine_datas = open(f'{self.generate_path}', 'r').read().split("\n")[:-1]
        datas = seed_datas + machine_datas
        self.situation = [json.loads(data)['situation'] for data in datas]
        
    def calculate_max_rouge_L(self, reference: str, hypotheses: list):
        scorer = rouge_scorer.RougeScorer(
            ['rougeL'], use_stemmer = False
        )
        with Pool(2) as p:
            rouge_scores = p.map(
                partial(scorer.score, reference), hypotheses
            )
        rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
        return max(rouge_scores)
    
    def postprocess(self, datas: str):
        datas = datas.split('\n')
        pattern = r"^\d+\.s*"
        result = []
        for data in datas:
            try:
                data = re.sub(pattern, "", data).lstrip()
                data = json.loads(data)
                score = self.calculate_max_rouge_L(
                    reference = data['situation'],
                    hypotheses = self.situation
                )
                if score < self.rouge_threshold:
                    result.append({
                        "emotion": data["emotion"],
                        "persona": data["persona"],
                        "situation": data["situation"]
                    })
                    self.situation.append(
                        data['situation']
                    )
                else:
                    raise ValueError(f"max rouge-L: {score:.2f}, over the threshold!")
            except Exception as e:
                logger.info(f"{e}, {data}")
                print(e)
            
        return result
        
    def _make_prompt(self):
        seed_datas = open(f'{self.seed_path}', 'r').read().split("\n")[:-1]
        machine_datas = open(f'{self.generate_path}', 'r').read().split("\n")[:-1]
        
        sample_seed_datas = random.sample(
            seed_datas, 
            k = min(self.sample_number, len(seed_datas))
        )
        sample_machine_datas = random.sample(
            machine_datas, 
            k = min(self.machine_sample_number, len(machine_datas))
        )
        sample_datas = sample_seed_datas + sample_machine_datas
        
        prompt = ""
        for idx, data in enumerate(sample_datas):
            prompt += f"{idx+1}. {data}\n"
        return prompt, idx+2

    @tenacity.retry(reraise=True, stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def _self_instruct(self):
        # prompt = PromptTemplate.from_template(
        #     "Generate diverse profile, include emotion, persona, situation. Do not repeat the content of template."
        #     "\n\nFormat Template: \n{template}"
        #     "\nOutput (do not repeat template):"
        # )
        template, next_idx = self._make_prompt()
        prompt = PromptTemplate.from_template(
            "Complete a series of profile, include emotion, persona, situation."
            "\n\n{template}"
            "{next_idx}. "
        )
        response = self.chain(prompt=prompt).run(
            template = template,
            next_idx = next_idx
        )
        return response
    
    def generate(self):
        responses = self._self_instruct()
        results = self.postprocess(datas = responses)
        for result in results:
            save_result(
                result = result,
                save_path= self.generate_path
            )
        return results

