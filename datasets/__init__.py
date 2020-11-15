from .widerface import WiderFace

from typing import List

__ds_mapper__ = {
    'widerface': {
        'cls': WiderFace,
        'args': (),
        'kwargs': {}
    },
    'widerface-easy':{
        'cls': WiderFace,
        'args': (),
        'kwargs': {'partitions':['easy']}
    },
    'widerface-medium':{
        'cls': WiderFace,
        'args': (),
        'kwargs': {'partitions':['medium']}
    },
    'widerface-hard':{
        'cls': WiderFace,
        'args': (),
        'kwargs': {'partitions':['hard']}
    }
}

def get_dataset(dataset_name:str, *args, **kwargs):
    assert dataset_name in __ds_mapper__.keys(),f"given dataset name must be one of the {list(__ds_mapper__.keys())}"
    ds = __ds_mapper__[dataset_name]
    args += ds['args']
    kwargs.update(ds['kwargs'])
    return ds['cls'](*args,**kwargs)

def get_available_datasets() -> List:
    return list(__ds_mapper__.keys())