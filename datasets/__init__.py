from .widerface import WiderFace

from typing import List

__ds_mapper__ = {
    'widerface': WiderFace
}

def get_dataset(dataset_name:str, *args, **kwargs):
    assert dataset_name in __ds_mapper__.keys(),f"given dataset name must be one of the {list(__ds_mapper__.keys())}"
    return __ds_mapper__[dataset_name](*args,**kwargs)

def get_available_datasets() -> List:
    return list(__ds_mapper__.keys())