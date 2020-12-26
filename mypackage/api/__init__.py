from typing import List,Dict

from ..utils.config import discover_archs,get_arch_cls

__all__ = [
    "list_archs", "list_arch_configs", "get_arch_config"]

"""
- list_archs() -> List[str]
- list_arch_configs(arch:str) -> List[str]
- get_arch_config(arch:str, config:str) -> Dict

- list_models() -> List[str] # TODO
"""

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