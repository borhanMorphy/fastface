import os
from typing import List, Union

from ..arch.base import ArchInterface
from ..config import ArchConfig
from ..adapter import download_object
from ..utils import discover_versions
from ..utils.cache import get_model_cache_dir
from ..utils.config import get_registry
from ..factory import _Factory


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
    registry = get_registry()
    assert model in registry, f"given model: {model} is not in the registry"
    adapter = registry[model]["adapter"]
    file_name = registry[model]["adapter"]["kwargs"]["file_name"]

    if target_path is None:
        for version in discover_versions(include_current_version=True):
            target_path = get_model_cache_dir(version=version)
            model_path = os.path.join(target_path, file_name)
            if os.path.isfile(model_path):
                break
            else:
                target_path = None

    target_path = target_path or get_model_cache_dir()

    model_path = os.path.join(target_path, file_name)

    assert os.path.exists(
        target_path
    ), f"given target path: {target_path} does not exists"
    assert os.path.isdir(target_path), "given target path must be directory not a file"

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
    ['centerface', 'lffd']

    """
    return _Factory.get_arch_names()


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
    return _Factory.get_arch_config_names(arch)


def get_arch_config(arch: str, config: str) -> ArchConfig:
    """Returns configuration object for given arch's named config

    Args:
        arch (str): architecture name
        config (str): configuration name

    Returns:
        ArchConfig: configuration details as `ArchConfig` object

    >>> import fastface as ff
    >>> ff.get_arch_config('lffd', 'slim')
    {'input_shape': (-1, 3, 480, 480), 'backbone_name': 'lffd-v2', 'head_infeatures': [64, 64, 64, 128, 128], 'head_outfeatures': [128, 128, 128, 128, 128], 'rf_sizes': [20, 40, 80, 160, 320], 'rf_start_offsets': [3, 7, 15, 31, 63], 'rf_strides': [4, 8, 16, 32, 64], 'scales': [(10, 20), (20, 40), (40, 80), (80, 160), (160, 320)]}

    """
    return _Factory.get_arch_config(arch, config)

def build_arch(arch: str, config: Union[str, ArchConfig]) -> ArchInterface:
    """Returns configuration object for given arch's named config

    Args:
        arch (str): architecture name
        config (str): configuration name as string or `ArchConfig`

    Returns:
        ArchInterface: torch.nn.Module with `ArchInterface` functionality

    """
    return _Factory.build_arch(arch, config)
