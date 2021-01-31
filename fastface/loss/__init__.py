__all__ = [
    "get_loss_by_name", "list_losses",
    "BinaryCrossEntropy",
    "L2Loss"
]

from .BCE import BinaryCrossEntropy
from .MSE import L2Loss
import torch.nn as nn
from typing import List

__loss_mapper__ = {
    'BCE':{
        'cls': BinaryCrossEntropy,
        'kwargs': {}
    },
    'MSE':{
        'cls': L2Loss,
        'kwargs': {}
    }
}

# TODO add *args
def get_loss_by_name(loss:str, **kwargs) -> nn.Module:
    assert loss in __loss_mapper__, f"given loss {loss} is not defined"
    cls = __loss_mapper__[loss]['cls']
    ckwargs = __loss_mapper__[loss]['kwargs'].copy()
    ckwargs.update(kwargs)
    return cls(*ckwargs)

def list_losses() -> List[str]: return list(__loss_mapper__.keys())