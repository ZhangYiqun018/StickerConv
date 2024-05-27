from datasets import load_dataset
from typing import Any, Dict, List, Optional, Tuple
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain_experimental.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser
from langchain.callbacks import get_openai_callback
import tenacity

class ConsistencyEvaluator(BaseModel):
    llm: BaseLanguageModel
    verbose: bool = False
    consistency_prompt: str = './template/consis.txt'
    total_cost: float = 0
    
    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm = self.llm,
            prompt = prompt,
            verbose= self.verbose
        )
    
    # dialogue level evaluation
    @tenacity.retry(reraise = True, stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def _compute_consistency_score(self, history: str, response: str) -> str:
        response_schemas = [
            ResponseSchema(name="reason", type='string', description="Explain the reason for the score."),
            ResponseSchema(name="score", type="int", description="1 - 5 score only"),
        ]
        parser = StructuredOutputParser(response_schemas=response_schemas)
        outfix = OutputFixingParser.from_llm(llm = self.llm, parser=parser)
        kwargs: dict = dict(
            conversation       = history,
            response           = response,
            format_instruction = parser.get_format_instructions()
        )
        prompt = PromptTemplate.from_template(
            open(self.consistency_prompt, 'r').read()
        )

        responses = self.chain(prompt = prompt).generate(
            [kwargs]
        ).generations[0]
        
        result = []
        for generation in responses:
            result.append(
                outfix.parse(generation.text)['score'])
        return result
    
    def make_history_prompt(self, history: list, query: str, sticker: dict=None):
        prompt = ""
        for idx, msg in enumerate(history):
            if idx % 2 == 0:
                prompt += f"user: {msg}\n"
            else:
                prompt += f"system: {msg}\n"
        prompt += f"user: {query}\n"
        
        if sticker is not None:
            prompt += f"user send a sticker: user sticker's emotion: {sticker.get('emotion', None)}, user sticker's description: {sticker.get('description', None)}\n"
        
        return prompt
    
    def compute_consistency_score(self, data: dict, consider_user_sticker: bool = False) -> str:
        history = data['history']
        usr_msg = data['query']
        sys_msg = data['pred_text_response'].replace("<IMG>", "").strip()
        
        if "input_image_metadata" in data.keys():
            usr_stc = data['input_image_metadata']
        else:
            usr_stc = None
        
        history_prompt = self.make_history_prompt(history = history, query = usr_msg, sticker = usr_stc if consider_user_sticker else None)
        
        with get_openai_callback() as cb:
            result = self._compute_consistency_score(
                history  = history_prompt,
                response = sys_msg,
            )
            self.total_cost += cb.total_cost
            return {
                "consistency_score"        : result,
                "consistency_score_mean"   : sum(result) / len(result),
                "cost"                     : cb.total_cost
            }