import torch
import numpy as np


def seed_everything(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)