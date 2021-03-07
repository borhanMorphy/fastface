__all__ = [
    "get_loss_by_name", "list_losses",
    "BinaryCrossEntropyLoss",
    "BinaryFocalLoss",

    "L2Loss",
    "DIoULoss"
]

from .BCE import BinaryCrossEntropyLoss
from .BFL import BinaryFocalLoss

from .MSE import L2Loss
from .DIoU import DIoULoss

import torch.nn as nn
from typing import List

__loss_mapper__ = {
    'BCE':{
        'cls': BinaryCrossEntropyLoss,
        'kwargs': {}
    },
    'BFL':{
        'cls': BinaryFocalLoss,
        'kwargs': {}
    },
    'MSE':{
        'cls': L2Loss,
        'kwargs': {}
    },
    'DIoU':{
        'cls': DIoULoss,
        'kwargs': {}
    }
}

# TODO add *args
def get_loss_by_name(loss:str, **kwargs) -> nn.Module:
    """Returns loss instance that matches with the given name

    Args:
        loss (str): name of the loss

    Returns:
        nn.Module: loss module as torch.nn.Module instance
    """
    assert loss in __loss_mapper__, f"given loss {loss} is not defined"
    cls = __loss_mapper__[loss]['cls']
    ckwargs = __loss_mapper__[loss]['kwargs'].copy()
    ckwargs.update(kwargs)
    return cls(**ckwargs)

def list_losses() -> List[str]:
    """Returns list of available loss names

    Returns:
        List: available loss names as list of strings
    """
    return list(__loss_mapper__.keys())