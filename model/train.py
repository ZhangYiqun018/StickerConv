import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import load_yaml, merge_config, init_distributed_mode, init_logger, get_rank
from pegs import BaseRunner


MODEL_CONFIG = "pegs/configs/common/pegs.yaml"
DATASETS_CONFIG = "pegs/configs/common/datasets.yaml"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    
    return args


def setup_seeds(config):
    seed = config.run.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main(args):
    # config
    train_config = load_yaml(args.config)
    model_config = load_yaml(MODEL_CONFIG)
    datasets_config = load_yaml(DATASETS_CONFIG)
    config = merge_config([train_config, model_config, datasets_config])

    # distributed
    init_distributed_mode(config.run)
    # seed  [Required]
    setup_seeds(config)
    # logging
    init_logger(config)
    
    runner = BaseRunner(config)
    runner.train()
    

if __name__ == "__main__":
    args = parse_args()

    main(args)
