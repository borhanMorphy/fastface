import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LFFDTinyPart(nn.Module):
    def __init__(self):
        super(LFFDTinyPart,self).__init__()

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        pass
