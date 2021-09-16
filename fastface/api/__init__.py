import os
from typing import Dict, List

from ..adapter import download_object
from ..utils.cache import get_model_cache_dir
from ..utils.config import discover_archs, get_arch_cls, get_registry


def list_pretrained_models() -> List[str]:
    """Returns available pretrained model names

    Returns:
        List[str]: list of pretrained model names

    >>> import fastface as ff
    >>> ff.list_pretrained_models()
    ['lffd_original', 'lffd_slim']
    """
    return list(get_registry().keys())


def download_pretrained_model(model: str, target_path: str = None) -> str:
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
        target_path = get_model_cache_dir()
    registry = get_registry()
    assert model in registry, f"given model: {model} is not in the registry"
    assert os.path.exists(
        target_path
    ), f"given target path: {target_path} does not exists"
    assert os.path.isdir(target_path), "given target path must be directory not a file"

    adapter = registry[model]["adapter"]
    file_name = registry[model]["adapter"]["kwargs"]["file_name"]
    model_path = os.path.join(target_path, file_name)

    if not os.path.isfile(model_path):
        # download if model not exists
        download_object(adapter["type"], dest_path=target_path, **adapter["kwargs"])
    return model_path


def list_archs() -> List[str]:
    """Returns available architecture names

    Returns:
        List[str]: list of arch names

    >>> import fastface as ff
    >>> ff.list_archs()
    ['lffd']

    """
    return [arch for arch, _ in discover_archs()]


def list_arch_configs(arch: str) -> List[str]:
    """Returns available architecture configurations as list

    Args:
        arch (str): architecture name

    Returns:
        List[str]: list of arch config names

    >>> import fastface as ff
    >>> ff.list_arch_configs('lffd')
    ['original', 'slim']

    """
    return list(get_arch_cls(arch).__CONFIGS__.keys())


def get_arch_config(arch: str, config: str) -> Dict:
    """Returns configuration dictionary for given arch and config names

    Args:
        arch (str): architecture name
        config (str): configuration name

    Returns:
        Dict: configuration details as dictionary

    >>> import fastface as ff
    >>> ff.get_arch_config('lffd', 'slim')
    {'input_shape': (-1, 3, 480, 480), 'backbone_name': 'lffd-v2', 'head_infeatures': [64, 64, 64, 128, 128], 'head_outfeatures': [128, 128, 128, 128, 128], 'rf_sizes': [20, 40, 80, 160, 320], 'rf_start_offsets': [3, 7, 15, 31, 63], 'rf_strides': [4, 8, 16, 32, 64], 'scales': [(10, 20), (20, 40), (40, 80), (80, 160), (160, 320)]}

    """
    arch_cls = get_arch_cls(arch)
    return arch_cls.__CONFIGS__[config].copy()
