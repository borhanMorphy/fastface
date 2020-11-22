import torch
import numpy as np
import random

from typing import List

def seed_everything(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def random_sample_selection(population:List, select_n:int) -> List:
    return random.sample(population, k=select_n)