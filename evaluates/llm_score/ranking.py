from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain_experimental.pydantic_v1 import BaseModel, Field
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser
import tenacity
import itertools
from tqdm.auto import tqdm
import json
import concurrent.futures

class RankingEvaluator(BaseModel):
    llm: BaseLanguageModel
    verbose: bool = False
    prompt_path: str = './template/ranking.txt'
    sticker_prompt_path: str = ""
    total_cost: float = 0
    
    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm     = self.llm,
            prompt  = prompt,
            verbose = self.verbose
        )
    @tenacity.retry(reraise = True, stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def _ranking(self, history: str, response_candidate: str) -> str:
        response_schemas = [
            ResponseSchema(name="reason", type='str', description="Provide a detailed justification for the ranking, focusing on the aspects of empathy and relevance to the conversation."),
            ResponseSchema(name="best", type="int", description="index of the best response, below in [1, 2, 3]"),
            ResponseSchema(name="medium", type='int', description="index of the medium quality response, below in [1, 2, 3]"),
            ResponseSchema(name="worst", type='int', description="index of the least effective response, below in [1, 2, 3]"),
        ]
        parser = StructuredOutputParser(response_schemas=response_schemas)
        outfix = OutputFixingParser.from_llm(
            llm = self.llm, parser = parser
        )
        kwargs = dict(
            conversation       = history,
            response_list      = response_candidate,
            format_instruction = parser.get_format_instructions()
        )
        prompt = PromptTemplate.from_template(
            open(self.prompt_path, 'r').read()
        )
        
        response = self.chain(prompt = prompt).run(**kwargs)
        response = outfix.parse(response)
        if self.verbose:
            print(response)
        result = {
            f"{response['best']}"  : "best",
            f"{response['medium']}": "medium",
            f"{response['worst']}" : "worst",
        }
        return result
    
    def ranking(self, d1, d2, d3):
        history = d1['history']
        query = d1['query']
        history_prompt = make_history_prompt(history=history, query=query)
        
        responses = {
            "vicuna"  : d1['pred_text_response'].strip(),
            "chatglm3": d2['pred_text_response'].strip(),
            "pegs"    : d3['pred_text_response'].replace("<IMG>", "").strip()
        }
        score = {'vicuna': 0, 'chatglm3': 0, 'pegs': 0}
        permutations = itertools.permutations(responses.keys())
        for perm in permutations:
            response_candidate = "\n".join([f"{i+1}. {responses[resp]}" for i, resp in enumerate(perm)])
            
            with get_openai_callback() as cb:
                ranking_result = rank_scorer._ranking(history=history_prompt, response_candidate=response_candidate)
                self.total_cost += cb.total_cost
            
            for idx, resp in enumerate(perm):
                try:
                    if ranking_result[str(idx + 1)] == 'best':
                        score[resp] += 1/6
                    elif ranking_result[str(idx + 1)] == 'medium':
                        score[resp] += 2/6
                    elif ranking_result[str(idx + 1)] == 'worst':
                        score[resp] += 3/6
                except Exception as e:
                    print(e)

        return score
        
    
def make_history_prompt(history: list, query: str):
    prompt = ""
    for idx, msg in enumerate(history):
        if idx % 2 == 0:
            prompt += f"user: {msg}\n"
        else:
            prompt += f"system: {msg}\n"
    prompt += f"user: {query}\n"
    return prompt

if __name__ == '__main__':
    from datasets import load_dataset
    from langchain.chat_models import ChatOpenAI
    
    vicuna = "../outputs/baseline/vicuna_predict.json"
    chatglm3 = "../outputs/baseline/chatglm3_predict.json"
    pegs = "../outputs/prediction_anno_full.json"
    
    
    vicuna_data = load_dataset(
        'json', data_files=vicuna, split = 'train'
    ).select(range(100))
    chatglm3_data = load_dataset(
        'json', data_files=chatglm3, split = 'train'
    ).select(range(100))
    pegs_data = load_dataset(
        'json', data_files=pegs, split = 'train'
    ).select(range(100))
    
    rank_scorer = RankingEvaluator(
        llm = ChatOpenAI(
            
        ),
        verbose=False
    )
    
    scores = []
    vicuna_score = 0
    chatglm3_score = 0
    pegs_score = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                rank_scorer.ranking,
                d1, d2, d3
            ) for d1, d2, d3 in zip(vicuna_data, chatglm3_data, pegs_data)
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pegs_data)):
            scores.append(future.result())

    for score in scores:
        vicuna_score += score['vicuna']
        chatglm3_score += score['chatglm3']
        pegs_score += score['pegs']
        
    scores.insert(0, {
        "vicuna_mean": vicuna_score / len(scores),
        "chatglm3_mean": chatglm3_score / len(scores),
        "pegs_mean": pegs_score / len(scores),
        "total_cost": rank_scorer.total_cost,
    })
    
    with open('./result/rank_result.json', 'w') as w:
        json.dump(scores, w, indent=4)