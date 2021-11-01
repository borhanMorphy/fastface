import os
from functools import wraps

from ..version import __version__


def ensure_path(fun):
    @wraps(fun)
    def more_fun(*args, **kwargs):
        p = fun(*args, **kwargs)
        if os.path.isfile(p):
            return p
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)
        return p

    return more_fun


@ensure_path
def get_root_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "fastface")


@ensure_path
def get_cache_dir(version: str = __version__) -> str:
    return os.path.join(get_root_cache_dir(), version)


@ensure_path
def get_model_cache_dir(suffix: str = "", version: str = __version__) -> str:
    return os.path.join(get_cache_dir(version=version), "model", suffix)


@ensure_path
def get_data_cache_dir(suffix: str = "", version: str = __version__) -> str:
    return os.path.join(get_cache_dir(version=version), "data", suffix)


@ensure_path
def get_checkpoint_cache_dir(suffix: str = "", version: str = __version__) -> str:
    return os.path.join(get_cache_dir(version=version), "checkpoints", suffix)
