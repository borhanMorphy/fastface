import torch
import numpy as np
import random

def seed_everything(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)