from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_experimental.generative_agents import GenerativeAgentMemory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.docstore import InMemoryDocstore
from langchain.schema.language_model import BaseLanguageModel
from langchain_experimental.pydantic_v1 import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import faiss
import re
import tenacity
from utils import switch_2_gpt4
import math
import datetime
import logging

logger = logging.getLogger(__name__)

def relevance_score_fn(score: float) -> float:
    return 1 - score / math.sqrt(2)

def create_memory_retriever():
    embedding_model = OpenAIEmbeddings(
        base_url = "http://0.0.0.0:9099/v1",
        api_key  = "a6000"
    )
    embedding_size = 1024
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embedding_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn = relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore      = vectorstore,
        # other_score_keys = ["importance"]
    )
    
def get_user_memory(llm: BaseLanguageModel, reflection_threshold: float=2, verbose: bool = False):
    memory = GenerativeAgentMemory(
        llm                  = llm,
        memory_retriever     = create_memory_retriever(),
        verbose              = verbose,
        reflection_threshold = reflection_threshold,
        importance_weight    = 0.15
    )
    return memory

class UserAgent(BaseModel):
    name: str
    persona: str = "N/A"
    status: str
    # related self memory
    summary: str = ""
    memory: GenerativeAgentMemory
    llm: BaseLanguageModel
    verbose: bool = False

    class Config:
        arbitrary_types_allowed = True
        
    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory
        )
    
    def postprocess(self, text: str):
        text = re.sub(r'^.*?: ', '', text)
        if text.startswith('A:') or text.startswith('B:'):
            try:
                text = text.split(':')[1].lstrip()
            except Exception as e:
                print(f"{text} - - -> {e}")
        text = text.replace("Hey!", '')
        text = text.split('\n')[0]

        return text.lstrip()
    
    def _compute_agent_summary(self) -> str:
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name} said"])
            .strip()
        )

    def get_summary(self, force_refresh: bool = False) -> str:
        if force_refresh:
            self.summary = self._compute_agent_summary()
        return (
            f"You are {self.name}"
            f"\nInnate traits: {self.persona}"
            f"\n{self.summary}"
            f"\nYour status: {self.status}"
        )

    def init_chat(self):
        summary = self.get_summary()
        prompt = PromptTemplate.from_template(
            open('./template/init_chat_prompt.txt', 'r').read()
        )
        response = self.chain(prompt).run(
            name    = self.name,
            summary = summary
        ).strip()
        
        response = self.postprocess(response)
        
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} engage in a casual chat, and said {response}",
                self.memory.now_key: datetime.datetime.now()
            },
        )
        return response

    def generate_norepeat_response(self, history: str, observation: str, count: int, max_turn: int, current_turn: int):
        current_llm = self.llm
        self.llm = switch_2_gpt4(config_path="config.ini")
        response = self.generate_response(history=history, observation=observation, max_turn=max_turn, current_turn=current_turn)
        print(f"solve repeat response, count: {count+1}")
        logger.info(f"solve repeat response, count: {count+1}")
        self.llm = current_llm
        
        return response
    
    @tenacity.retry(reraise = True, stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def generate_response(self, history: str, observation: str, max_turn: int, current_turn: int):
        prompt = PromptTemplate.from_template(
            open('./template/chat_prompt.txt', 'r').read()
        )
        # 双数轮刷新summary
        summary = self.get_summary(
            force_refresh=True if current_turn % 2 == 0 else False
        )
        response = self.chain(prompt).run(
            summary      = summary,
            history      = history,
            observation  = observation,
            name         = self.name,
            max_turn     = max_turn,
            current_turn = current_turn
        ).strip()

        response = self.postprocess(response)

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} reply {response}",
                self.memory.now_key: datetime.datetime.now()
            },
        )

        return response

    # =============
    # plan module
    # intent -> query -> (retrieve) -> select
    # =============
    def _make_query_example(self) -> str:
        labels = ['Anger', 'Fear', 'Neutral', 'Sadness', 'Happiness', 'Disgust', 'Surprise']
        example = ""
        for idx, label in enumerate(labels):
            data = open(f'./template/query/{label}_example.txt', 'r').read().strip().split('\n')
            import random
            query = random.choice(data)
            example += f"{idx+1}. {query}\n"
        return example
    
    def _make_sticker_selection_prompt(self, documents: list) -> str:
        prompt = "="*10
        for idx, document in enumerate(documents):
            meta_data = document.metadata
            current_prompt = (
                f"\nidx: {idx}"
                f"\ndescription: {meta_data['description']}"
                f"\nemotion: {meta_data['emotion']}"
                f"\nrecommendation: {meta_data['recommendation']}\n"
            )
            prompt += current_prompt + "="*10
        return prompt
    
    @tenacity.retry(reraise=True, stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def _sticker_intent(self, message: str) -> str:
        response_schemas = [
            ResponseSchema(name = "intention", 
                           typing = 'string', 
                           description="select one of [Emotional Expression, Enhancing Emoathy, Reaction Intensity, Other, None]")
        ]
        prompt = PromptTemplate.from_template(
            open('./template/sticker_intent.txt', 'r').read()
        )
        parser = StructuredOutputParser(response_schemas=response_schemas)
        response = self.chain(prompt = prompt).run(
            name                = self.name,
            summary             = self.get_summary(),
            message             = message,
            format_instructions = parser.get_format_instructions()
        )
        
        response = parser.parse(response)
        return response
    
    @tenacity.retry(reraise=True, stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def _sticker_query(self, message: str, intent: str):
        prompt = PromptTemplate.from_template(
            open('./template/sticker_query.txt', 'r').read()
        )
        response = self.chain(prompt = prompt).run(
            summary = self.get_summary(),
            name    = self.name,
            message = message,
            intent  = intent,
            example = self._make_query_example()
        )
        if self.verbose:
            print(f"query --> {response}")
            logger.info(f"{self.name}'s message: {message}, intent: {intent}, query --> {response}")
        
        return response

    @tenacity.retry(reraise=True, stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def _sticker_select(self, message: str, intent: str, sticker_documents: list) -> str:
        sticker_prompt = self._make_sticker_selection_prompt(documents=sticker_documents)
        prompt = PromptTemplate.from_template(
            open('./template/sticker_select.txt', 'r').read()
        )
        response = self.chain(prompt = prompt).run(
            summary        = self.get_summary(),
            name           = self.name,
            message        = message,
            intent         = intent,
            sticker_prompt = sticker_prompt
        ).strip().split('\n')[0]
        try:
            response = int(response)
            return response
        except:
            return -1
    
    def _sticker_retriever(self, database, query: str, candidate_number: int=10):
        database.retriever.search_kwargs = {
            "k": candidate_number
        }
        documents = database.retriever.get_relevant_documents(query = query)
        return documents
        
    def sticker_process(self, meme_database, message: str, force: bool = False, candidate_number: int = 10):
        try:
            if not force:
                intent = self._sticker_intent(message = message)
                if intent['intention'] not in ["Emotional Expression", "Enhancing Empathy", "Reaction Intensity"]:
                    return None
                else:
                    query = self._sticker_query(message = message, intent=intent['intention'])
            else:
                query = self._sticker_query(message = message, intent = "Reaction Intensity and Emotional Expression")
            
            documents = self._sticker_retriever(database = meme_database, query = query, candidate_number=candidate_number)
            select = self._sticker_select(message = message, intent = intent, sticker_documents=documents)
            if select != -1:
                sticker = documents[select].metadata
                self.memory.save_context(
                    {},
                    {
                        self.memory.add_memory_key: f"{self.name} use a sticker, sticker's emotion: {sticker['emotion']}, sticker's description: {sticker['description']}",
                        self.memory.now_key: datetime.datetime.now()
                    },
                )
                return sticker
            return None
        except Exception as e:
            logger.error(f"{e}")
            return None