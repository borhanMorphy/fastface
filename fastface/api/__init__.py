__all__ = [
    "list_pretrained_models", "download_pretrained_model",
    "list_archs", "list_arch_configs", "get_arch_config"
]

from typing import List,Dict
import os

from ..utils.config import (
    discover_archs,
    get_arch_cls,
    get_registry
)
from ..utils.cache import get_model_cache_path

from ..adapter import download_object

"""
- list_pretrained_models() -> List[str]
- download_pretrained_model(model:str, target_path:str=None) -> str
- list_archs() -> List[str]
- list_arch_configs(arch:str) -> List[str]
- get_arch_config(arch:str, config:str) -> Dict

"""

def list_pretrained_models() -> List[str]:
    """returns available pretrained model names

    Returns:
        List[str]: list of pretrained model names
    """
    return list(get_registry().keys())

def download_pretrained_model(model:str, target_path:str=None) -> str:
    """Downloads pretrained model to given target path,
    if target path is None, it will use model cache path.
    If model already exists in the given target path than it will do notting.

    Args:
        model (str): pretrained model name to download
        target_path (str, optional): target directory to download model. Defaults to None.

    Returns:
        str: file path of the model
    """
    if target_path is None:
        target_path = get_model_cache_path()
    registry = get_registry()
    assert model in registry,f"given model: {model} is not in the registry"
    assert os.path.exists(target_path),f"given target path: {target_path} does not exists"
    assert os.path.isdir(target_path),f"given target path must be directory not a file"

    adapter = registry[model]["adapter"]
    file_name = registry[model]["adapter"]["kwargs"]["file_name"]
    model_path = os.path.join(target_path,file_name)

    if not os.path.isfile(model_path):
        # download if model not exists
        download_object(adapter['type'],
            dest_path=target_path, **adapter['kwargs'])
    return model_path

def list_archs() -> List[str]:
    """returns available architecture names

    Returns:
        List[str]: list of arch names
    """
    return [arch for arch,_ in discover_archs()]

def list_arch_configs(arch:str) -> List[str]:
    """returns available architecture configurations as list

    Args:
        arch (str): architecture name

    Returns:
        List[str]: list of arch config names
    """
    return list(get_arch_cls(arch).__CONFIGS__.keys())

def get_arch_config(arch:str, config:str) -> Dict:
    """returns configuration dictionary for given arch and config names

    Args:
        arch (str): architecture name
        config (str): configuration name

    Returns:
        Dict: configuration details as dictionary
    """
    arch_cls = get_arch_cls(arch)
    return arch_cls.__CONFIGS__[config].copy()