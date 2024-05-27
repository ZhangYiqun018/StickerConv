# %%
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
    def _ranking(self, history: str, response_candidate: str, candidate_numbers: int) -> str:
        response_schemas = [
            # ResponseSchema(name="reason", type='str', description="Provide a detailed justification for the ranking, focusing on the aspects of empathy and relevance to the conversation."),
            ResponseSchema(name="ranking", type='list', description=f"Start with the number 1 for the best response, followed by 2, 3, etc., in descending order of empathy and relevance.")
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
        
        response = self.chain(prompt = prompt).generate(
            [kwargs]
        ).generations[0]
        
        result = []
        for generation in response:
            parse_result = outfix.parse(generation.text)['ranking']
            try:
                assert len(parse_result) == candidate_numbers, 'check!'
            except:
                parse_result = [0] * candidate_numbers
                
            result.append(parse_result)
            
        if self.verbose:
            print(result)

        return result
    
    def ranking(self, candidate_list: list):
        # get history
        history = candidate_list[0].get('history')
        query = candidate_list[0].get('query')
        history_prompt = make_history_prompt(history=history, query=query)
        
        # get response
        response_prompt = make_candidate_template(candidate_list)
         
        with get_openai_callback() as cb:
            ranking_result = self._ranking(history=history_prompt, response_candidate=response_prompt, candidate_numbers=len(candidate_list))
            mean_result = [sum(t)/len(ranking_result) for t in zip(*ranking_result)]
            
            self.total_cost += cb.total_cost

        return {
            'rank': ranking_result,
            'rank_mean': mean_result,
            'total_cost': self.total_cost
        }
        
    
def make_history_prompt(history: list, query: str):
    prompt = ""
    for idx, (usr, sys) in enumerate(history):
        prompt += f"USER: {usr.replace('<IMG>', '').strip()}\nSYSTEM: {sys.replace('<IMG>', '').strip()}\n"
        
    prompt += f"USER: {query.replace('<IMG>', '').strip()}\n"
    return prompt

def make_sticker_prompt(sticker):
    description = sticker.get('description', None)
    emotion = sticker.get('emotion', None)
    
    prompt = f"send a sticker, sticker's emotion: {emotion}, sticker's description: {description}"
    return prompt

def make_candidate_template(candidate_list: list) -> tuple[str, int]:
    prompt = ""
    split_line = "="*10
    for idx, resp in enumerate(candidate_list):
        # 抽取文本回复
        try:
            pred_response = resp['pred_text_response']
        except:
            pred_response = resp['pred_text_reponse']
        # 抽取图片回复
        pred_sticker = resp.get('pred_image_metadata', None)
        if pred_sticker is None:
            prompt += f"Candidate {idx+1}\nSYSTEM: {pred_response}\n{split_line}\n"
        else:
            prompt += f"Candidate {idx+1}\nSYSTEM: {pred_response}\nSYSTEM Action: {make_sticker_prompt(pred_sticker)}\n{split_line}\n"
        
    return prompt
        
if __name__ == '__main__':
    from datasets import load_dataset
    from langchain.chat_models import ChatOpenAI
    
    dataset_paths = {
        'vicuna_text': "/datas/zyq/research/chat_meme/evaluates/outputs/baseline/vicuna_text_predict.json",
        'chatglm3_text': "/datas/zyq/research/chat_meme/evaluates/outputs/baseline/chatglm3_text_predict.json",
        'vicuna_tool': "/datas/zyq/research/chat_meme/evaluates/finetune_baseline/tool_finetune/vicuna/vicuna_tool_test.json",
        'chatglm3_tool': "/datas/zyq/research/chat_meme/evaluates/finetune_baseline/tool_finetune/chatglm/glm_tool_test2.json",
        'pegs_ret': "/datas/zyq/research/chat_meme/evaluates/outputs/baseline/pegs_v1.5_ret_full_anno.json",
        'pegs_gen': "/datas/zyq/research/chat_meme/evaluates/outputs/baseline/pegs_v15_gen_full_anno.json",
        "pegs_rag": "/datas/zyq/research/chat_meme/evaluates/outputs/baseline/pegs_v1.5_rag_full_anno.json"
    }
    
    with open('/datas/zyq/research/chat_meme/evaluates/human/idx.txt', 'r') as r:
        idx = r.read().split('\n')
        idx = [int(d) for d in idx]
    
    # idx = range(0, 10)
    datasets = {name: load_dataset('json', data_files=path, split='train').select(idx)
            for name, path in dataset_paths.items()}
    
    datasets_order = [
        'pegs_gen', 'pegs_ret', 'pegs_rag', 'vicuna_tool', 'chatglm3_tool', 'vicuna_text', 'chatglm3_text'
    ]
    
    rank_scorer = RankingEvaluator(
        llm = ChatOpenAI(
            base_url = "",
            api_key  = "",
            n        = 3,
        ),
        verbose=False
    )
    
    scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                rank_scorer.ranking,
                [datasets[name][i] for name in datasets_order]
            ) for i in range(len(idx))
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            scores.append(future.result())

    score_mean = [score['rank_mean'] for score in scores]
    rank_mean = [sum(t) / len(score_mean) for t in zip(*score_mean)]
    
    scores.insert(0, {
        "rank_mean": rank_mean,
        "distributed_rank": {name: score for name, score in zip(datasets_order, rank_mean)},
        "total_cost": rank_scorer.total_cost,
    })
    
    print(scores[0])
    
    with open('./result/rank_7.json', 'w') as w:
        json.dump(scores, w, indent=4)
# %%
 