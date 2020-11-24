import torch
import numpy as np
import random

from typing import List,Tuple
import os

def seed_everything(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def random_sample_selection(population:List, select_n:int) -> List:
    return random.sample(population, k=select_n)

def get_best_checkpoint_path(checkpoint_dir:str, by:str='val_ap', mode:str='max') -> Tuple[float,str]:
    checkpoints = []
    for file_name in os.listdir(checkpoint_dir):
        pure_name,ext = os.path.splitext(file_name)
        for part in pure_name.split('-'):
            if '=' not in part or by not in part:
                continue
            _,v = part.split("=")
            v = float(v)
            checkpoints.append((v,file_name))
            break
    return eval(mode)(checkpoints)
