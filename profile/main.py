from generator import DataGenerator
from utils import get_llm, analysis
import configparser
from tqdm.auto import tqdm
from langchain.callbacks import get_openai_callback
import logging
import datetime
import argparse

logging.basicConfig(filename=f'./log/profile_{datetime.datetime.now().strftime("%m-%d_%H:%M")}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Run Data Generator')
parser.add_argument('--config_path', type=str, default='../agent/config_private.ini', help='Path to the config file')
parser.add_argument('--sample_number', type=int, default=5, help='Number of samples to generate')
parser.add_argument('--machine_sample_number', type=int, default=2, help='Number of machine samples to generate')
parser.add_argument('--generator_number', type=int, default=20, help='Total number of generations')
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
parser.add_argument('--threshold', type=float, default=0.7)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_path)

llm = get_llm(type='test', config=config, max_tokens=500, temperature=0.7)

generator = DataGenerator(
    llm                   = llm,
    seed_path             = 'seeds.json',
    machine_generate_path = 'machine_generate.json',
    sample_number         = args.sample_number,
    machine_sample_number = args.machine_sample_number,
    verbose               = args.verbose,
    threshold             = args.threshold,
)

pbar = tqdm(total = args.generator_number)
count = 0

with get_openai_callback() as cb:
    while count < args.generator_number:
        result = generator.generate()
        pbar.update(len(result))
        count += len(result)
        logging.info(pbar.__str__())
    logging.info(cb)
    logging.info(analysis(
        path = 'machine_generate.json', key='emotion')
    )