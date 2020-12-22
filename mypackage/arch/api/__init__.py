import torch.nn as nn
from typing import List,Dict
import importlib

__all__ = [
    'get_arch_by_name', 'get_available_archs',
    'get_arch_config_by_name', 'get_available_arch_configs'
]

__ARCHS__ = ('lffd',)

def get_arch_by_name(arch_name:str, *args, config:Dict={}, **kwargs) -> nn.Module:
    assert arch_name in __ARCHS__,f"architecture not found: {arch_name}"
    arch_api = importlib.import_module(f'archs.{arch_name}')
    return arch_api.arch_cls(*args, config=config, **kwargs)

def get_available_archs() -> List[str]:
    global __ARCHS__
    return list(__ARCHS__)

def get_arch_config_by_name(arch_name:str, config:str='') -> Dict:
    assert arch_name in __ARCHS__,f"architecture not found: {arch_name}"
    arch_api = importlib.import_module(f'archs.{arch_name}')
    assert config in arch_api.arch_cls.__CONFIGS__,f"{arch_name} does not have configuration: {config}"
    return arch_api.arch_cls.__CONFIGS__[config].copy()

def get_available_arch_configs(arch_name:str) -> List[str]:
    assert arch_name in __ARCHS__,f"architecture not found: {arch_name}"
    arch_api = importlib.import_module(f'archs.{arch_name}')
    return list(arch_api.arch_cls.__CONFIGS__.keys())