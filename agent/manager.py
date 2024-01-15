from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain_experimental.pydantic_v1 import BaseModel
from langchain.schema import BaseMemory
from langchain.output_parsers import ResponseSchema, OutputFixingParser, StructuredOutputParser
import random
from utils import save_result, switch_2_azure
from vector_base import MemeVector
from typing import Any, Dict, List, Optional, Tuple
import re
import tenacity
import configparser
import logging

logger = logging.getLogger(__name__)

# langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
class ConversationManager(object):
    def __init__(self, llm: BaseLanguageModel, memory: BaseMemory, database: MemeVector, max_turn: int,
                 UserAgent: BaseModel, SystemAgent: BaseModel, verbose: bool = False, save_path: str = None ):
        self.memory = memory
        self.UserAgent = UserAgent
        self.SystemAgent = SystemAgent
        self.database = database
        self.llm = llm
        self.verbose = verbose
        self.history = []
        self.user_msg_list = []
        self.system_msg_list = []
        self.max_turn = max_turn
        self.sticker_counter = 0
        self.sticker_stop = 0
        self.reset()
        self.save_path = save_path

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose,
        )
    
    def reset(self) -> None:
        self.history.clear()
        self.memory.clear()
        self.system_msg_list.clear()
        self.user_msg_list.clear()
        self.sticker_counter = 0
        self.sticker_stop = 0
        
    @tenacity.retry(reraise=True, stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def llm_reviewer(self, message: str, sticker: dict):
        response_schema = [
            ResponseSchema(
                name = "consistency", type="bool", description="Is there a clear inconsistency between the stickers and the conversation content?"
            )
        ]
        parser = StructuredOutputParser(response_schemas=response_schema)
        prompt = PromptTemplate.from_template(
            open('./template/llm_reviewer.txt', 'r').read()
        )
        response = self.chain(prompt = prompt).run(
            message = message, emotion = sticker['emotion']
        )
        response = parser.parse(response)
        return response['consistency']
    
    @tenacity.retry(reraise=True, stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def conv_closure(self, turn):
        response_schemas = [
            ResponseSchema(
                name = "output", type="string", description="FINI or CONT"
            )
        ]
        parser = StructuredOutputParser(response_schemas=response_schemas)
        autofix = OutputFixingParser.from_llm(parser = parser, llm = self.llm)

        prompt = PromptTemplate.from_template(
            open('./template/conv_closure.txt', 'r').read()
        )

        response = self.chain(prompt=prompt).run(
            history            = self.memory.load_memory_variables({})['chat_history'],
            format_instruction = parser.get_format_instructions()
        )
        response = autofix.parse(response)['output']

        logger.info(f"Turn: {turn}: Conv closure - - - - -> {response}")
        
        return True if response == "CONT" else False
    
    def save_history(self, turn: int, user_msg: str, system_msg: str, user_sticker: str, system_sticker: str):
        self.history.append({
            "turn"          : turn,
            "user_message"  : user_msg,
            "system_message": system_msg,
            "user_sticker"  : user_sticker,
            "system_sticker": system_sticker
        })
        self.memory.save_context(
            inputs  = {"input": user_msg},
            outputs = {"output": system_msg}
        )
        self.user_msg_list.append(user_msg)
        self.system_msg_list.append(system_msg)
        if user_sticker:
            self.sticker_counter += 1
            self.sticker_stop = -100
        if system_sticker:
            self.sticker_counter += 1
            self.sticker_stop = -100
        if user_sticker is None and system_sticker is None:
            self.sticker_stop += 1
        # self.memory.prune()
    
    def _generate_observation(self, speaker: str, sticker: Any) -> str:
        if sticker is None:
            observation = f"{speaker}'s action: Only speak."
        else:
            sticker_describe = f"{sticker['emotion']} {sticker['description']}"
            observation = f"\n{speaker}'s action: Speak and use a sticker {sticker_describe}"

        return observation.strip()

    # 处理复读
    def _once_generate(self, agent: BaseModel, history: str, observation: str, last_msg: str=None, retry_count: int=3, turn: int=1, max_turn: int=10):
        msg = agent.generate_response(
            history = history, observation = observation, max_turn=max_turn, current_turn=turn
        )
        # 判断是否复读
        if (msg in self.user_msg_list) or (msg in self.system_msg_list) or (msg == last_msg):
            count = 0
            while count < retry_count:
                msg = agent.generate_norepeat_response(history=history, observation = observation, 
                                                       count = count, max_turn=max_turn, current_turn=turn)
                if (msg in self.user_msg_list) or (msg in self.system_msg_list) or (msg == last_msg):
                    count += 1
                    if count == retry_count:
                        return None
                else:
                    return msg
        else:
            return msg
    
    def _once_chat(self, turn: int, last_user_msg: str=None, last_system_msg: str=None, 
                   last_user_sticker: str=None, last_system_sticker: str=None, candidate_number: int = 9,
                   user_token: str = "User", system_token: str = "System", retry_count: int = 3, interval: int = 5) -> Tuple[bool, str, str, Any, Any]:
        if turn == 1:
            user_msg = self.UserAgent.init_chat()
        else:
            # solve repeat problem
            user_msg = self._once_generate(
                agent       = self.UserAgent,
                history     = self.memory.load_memory_variables({})['chat_history'],
                observation = self._generate_observation(speaker=system_token, sticker=last_system_sticker),
                retry_count = retry_count,
                turn        = turn,
                max_turn    = self.max_turn
            )
            if user_msg is None:
                print(f"FINISH ---> finish reason: max retry {retry_count}, repeat from the past")
                return False, None, None, None, None
            
        if last_user_sticker is None and last_system_sticker is None:
            user_sticker = self.UserAgent.sticker_process(
                meme_database    = self.database,
                message          = user_msg,
                candidate_number = candidate_number,
            )
        else:
            user_sticker = None
        if user_sticker is not None:
            review_result = self.llm_reviewer(message = user_msg, sticker = user_sticker)
            if review_result is False:
                user_sticker = None
        # solve repeat problem
        system_msg = self._once_generate(
            agent       = self.SystemAgent,
            history     = self.memory.load_memory_variables({})['chat_history'] + f"\n{user_token}: {user_msg}",
            observation = self._generate_observation(speaker=user_token, sticker=user_sticker),
            last_msg    = user_msg,
            retry_count = retry_count,
            max_turn    = self.max_turn,
            turn        = turn
        )
        if system_msg is None:
            print(f"FINISH ---> finish reason: max retry {retry_count}, repeat from the past")
            return False, None, None, None, None

        if self.sticker_stop >= interval:
            system_sticker = self.SystemAgent.sticker_process(
                meme_database    = self.database,
                message          = system_msg,
                candidate_number = candidate_number,
            )
        else:
            # 不连续使用sticker
            if last_system_sticker is None:
                system_sticker = self.SystemAgent.sticker_process(
                    meme_database    = self.database,
                    message          = system_msg,
                    candidate_number = candidate_number,
                )
            else:
                system_sticker = None
                
        if system_sticker is not None:
            review_result = self.llm_reviewer(message = system_msg, sticker=system_sticker)
            if review_result is False:
                system_sticker = None
        # 开始复读，直接丢掉
        if (user_msg == system_msg) or (user_msg in self.user_msg_list) or (system_msg == self.system_msg_list) or (user_msg == self.system_msg_list) or (system_msg == self.user_msg_list):
            print("FINISH ---> finish reason: repeat from the past")
            return False, None, None, None, None
        
        self.save_history(
            turn           = turn,
            user_msg       = user_msg,
            system_msg     = system_msg,
            user_sticker   = user_sticker,
            system_sticker = system_sticker
        )

        if turn >= self.max_turn:
            print(f"FINISH ---> current turn: {turn}, finish reason: Get max turn!")
            return False, user_msg, system_msg, user_sticker, system_sticker
        
        if turn >= self.max_turn - 2 and turn % 2 == 0:
            conv_closure = self.conv_closure(turn=turn)
            if conv_closure == False:
                print(f"FINISH ---> current turn: {turn}, finish reason: Agent decision stop generate!")
            return conv_closure, user_msg, system_msg, user_sticker, system_sticker
        
        return True, user_msg, system_msg, user_sticker, system_sticker

    def chat_loop(self, user_token: str = 'User', system_token: str = "System", pbar = None, candidate_number: int = 10):
        turn = 0
        last_user_msg = None
        last_system_msg = None
        last_user_sticker = None
        last_system_sticker = None
        self.reset()
        
        minimum_sticker_interval = random.randint(a=1, b=3)
        while True:
            turn += 1
            if pbar is not None:
                desc = pbar.desc.split(' - ')[0]
                pbar.set_description_str(f"{desc} - Current Turn: {turn}")
            flag, last_user_msg, last_system_msg, last_user_sticker, last_system_sticker = self._once_chat(
                turn                = turn,
                last_user_msg       = last_user_msg,
                last_system_msg     = last_system_msg,
                last_user_sticker   = last_user_sticker,
                last_system_sticker = last_system_sticker,
                user_token          = user_token,
                system_token        = system_token,
                interval            = minimum_sticker_interval,
                candidate_number    = candidate_number
            )
            if not flag:
                break
        
        if self.sticker_counter > 0:
            save_result(result = {
                "conversation": self.history,
                "user_persona": self.UserAgent.persona,
                "user_status" : self.UserAgent.status
            }, save_path = self.save_path)
            return True
        else:
            save_result(result = {
                "conversation": self.history,
                "user_persona": self.UserAgent.persona,
                "user_status" : self.UserAgent.status
            }, save_path = "../dataset/backup/dialogue_backup.json")
            return False