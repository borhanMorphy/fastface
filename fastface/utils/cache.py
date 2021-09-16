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
def get_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "fastface", __version__)


@ensure_path
def get_model_cache_dir(suffix: str = "") -> str:
    return os.path.join(get_cache_dir(), "model", suffix)


@ensure_path
def get_data_cache_dir(suffix: str = "") -> str:
    return os.path.join(get_cache_dir(), "data", suffix)


@ensure_path
def get_checkpoint_cache_dir(suffix: str = "") -> str:
    return os.path.join(get_cache_dir(), "checkpoints", suffix)
