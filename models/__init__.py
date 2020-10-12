from .lffd import LFFD
import torch.nn as nn
from typing import List

__model_mapper__ = {
    "lffd":{
        "cls":LFFD
    }
}

def get_detector_by_name(model_name:str, *args, **kwargs) -> nn.Module:
    assert model_name in __model_mapper__.keys(),"model name not found"
    return __model_mapper__[model_name]['cls'](*args,**kwargs)

def get_available_detectors() -> List:
    global __model_mapper__
    return list(__model_mapper__.keys())