from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import get_openai_callback
from manager import ConversationManager
from vector_base import MemeVector
from user_agent import UserAgent, get_user_memory
from datasets import load_dataset
from utils import get_llm, data_analysis
from tqdm.auto import tqdm, trange
import configparser
import logging
import datetime
import random
import argparse
import json

parser = argparse.ArgumentParser(description="Parameters of Sticker Agent")
parser.add_argument(
    '--llm_config', type = str, default='config_private.ini'
)
parser.add_argument(
    '--verbose', action = 'store_true'
)
parser.add_argument(
    '--mode', type = str, default='train', help = 'the mode of generated, train of test'
)
parser.add_argument(
    '--start_idx', type = int, default = 1, help = 'the start idx of user profile, must > 0'
)
parser.add_argument(
    '--end_idx', type = int, default = -1, help = 'the end idx of user profile, must > 0'
)
parser.add_argument(
    '--max_turn', type = int, default=6, help = 'the max turns of generated dialogues'
)
parser.add_argument(
    '--llm_type', type = str, default='azure', help = 'the default llm api type'
)
parser.add_argument(
    '--switch_llm', action = 'store_true', help = 'every agent use different api'
)
parser.add_argument(
    '--shuffle', action = 'store_true'
)
parser.add_argument(
    '--seed', type= int, default = 42
)
parser.add_argument(
    '--profile_path', type = str, default="../dataset/profile/user_profile.json"
)
parser.add_argument(
    '--candidate_number', type = int, default=10
)

args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.llm_config)

# init parameters
verbose = False
user_token = 'B'
system_token = 'A'
system_info_path = "./template/system_info.json"
user_profile_path = args.profile_path
default_llm_type = args.llm_type
use_multi_query = False
embedding_model = "../bge-large-en-v1.5"
max_turn = args.max_turn
generate_mode = args.mode
assert generate_mode in ["train", "test"], "generate model error!"
start_idx = args.start_idx
if args.end_idx == -1:
    end_idx = 21 if generate_mode == 'test' else 81
else:
    end_idx = args.end_idx
    assert end_idx > start_idx, f"end idx ({end_idx}) must > start idx ({start_idx}), please check it!"
    assert end_idx <= 21 if generate_mode == "test" else 81

save_path = f"../dataset/dialogue/normal/{generate_mode}_{start_idx}_{end_idx}.json"

system_info = json.loads(open(system_info_path, 'r').read().strip())

user_profile = load_dataset(
    'json',
    data_files = user_profile_path,
    split      = 'train'
)
if args.shuffle:
    user_profile.shuffle(seed=args.seed)
    
windows = int(len(user_profile) / 20) if generate_mode == 'test' else int(len(user_profile) / 80)
candidate_number = args.candidate_number

logging.basicConfig(filename=f'./log/{generate_mode}_{start_idx}_{end_idx}_{datetime.datetime.now().strftime("%y-%m-%d_%H:%M")}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"Parameters:")
logging.info(f"Generate Mode = {generate_mode}")
logging.info(f"start idx = {start_idx}, end idx = {end_idx}, windows = {windows}")
logging.info(f"api config = {args.llm_config}")
logging.info(f"verbose = {verbose}")
logging.info(f"user_token = {user_token}, system_token = {system_token}")
logging.info(f"default_llm_type = {default_llm_type}, use_llm_switch = {args.switch_llm}")
logging.info(f"use_multi_query = {use_multi_query}")
logging.info(f"save_path = {save_path}")
logging.info(f"embedding_model = {embedding_model}")
logging.info(f"system info = {system_info}")
logging.info(f"user profile = {user_profile_path}")
logging.info(f"max turn = {max_turn}")
logging.info(f"user profile number: {len(user_profile)}")
logging.info(f"shuffle = {args.shuffle}, seed = {args.seed}")

memory = ConversationSummaryBufferMemory(
    human_prefix    = user_token,
    ai_prefix       = system_token,
    llm             = get_llm(config=config, type=default_llm_type, temperature=0, max_tokens=500),
    memory_key      = "chat_history",
    max_token_limit = 2000
)

user = UserAgent(
    name        = user_token,
    persona     = "",
    status      = "",
    llm         = get_llm(config=config, type=default_llm_type, temperature=0.7, max_tokens=150),
    verbose     = verbose,
    memory = get_user_memory(
        llm                  = get_llm(config=config, type=default_llm_type, temperature=0.1, max_tokens=150),
        reflection_threshold = 2.0
    )
)

system = UserAgent(
    name        = system_token,
    persona     = system_info['persona'],
    status      = system_info['status'],
    llm         = get_llm(config=config, type=default_llm_type, temperature=0.5, max_tokens=150),
    verbose     = verbose,
    memory = get_user_memory(
        llm                  = get_llm(config=config, type=default_llm_type, temperature=0.1, max_tokens=150),
        reflection_threshold = 2.0
    )
)

manager = ConversationManager(
    memory      = memory,
    UserAgent   = user,
    SystemAgent = system,
    verbose     = verbose,
    database    = None,
    llm         = get_llm(config=config, type=default_llm_type, temperature= 0.0, max_tokens=200),
    save_path   = save_path,
    max_turn    = max_turn
)

# compute the cost of generate data
with get_openai_callback() as cb:
    pbar = tqdm(total = windows * (end_idx-start_idx), leave=True)
    for i in trange(start_idx, end_idx):
        pbar.set_description_str(f"current split {i}")
        sticker_database = MemeVector(
            llm             = get_llm(config, temperature=0, type='local', max_tokens=200),
            db_path         = f"../dataset/vectorstore/{generate_mode}/split_{i}",
            model_name      = embedding_model,
            data_path       = f"../dataset/labeled/{generate_mode}/sticker_{i}.json",
            force_create    = False,
            use_multi_query = use_multi_query
        )
        manager.database = sticker_database
        current_user = user_profile[(i-1)*windows : i*windows]
        for idx, (p, s) in  enumerate(zip(current_user['persona'], current_user['situation'])):
            # init persona and situation
            user.persona = p
            user.status = s
            # init memory 
            user.memory = get_user_memory(
                llm = get_llm(config=config, type=default_llm_type, temperature=0.1, max_tokens=150),
                reflection_threshold = 2.0, verbose = True
            )
            system.memory = get_user_memory(
                llm = get_llm(config=config, type=default_llm_type, temperature=0.1, max_tokens=150),
                reflection_threshold = 2.0, verbose = True  
            )
            # 更换api
            flag = manager.chat_loop(user_token=user_token, system_token=system_token, pbar=pbar, candidate_number=candidate_number)
            if flag:
                pbar.update(1)
            post = data_analysis(data_path=manager.save_path)
            pbar.set_postfix_str(post)
            logging.info(pbar.__str__())
            logging.info(f"current cost: {cb.total_cost:.4f}$")
    logging.info(cb)