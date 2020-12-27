import importlib
from ..utils.config import get_pkg_root_path
from .gdrive import GoogleDriveAdapter
from .http import HttpAdapter

__all__ = ['download_object']

__adapters__ = {
    'gdrive': GoogleDriveAdapter,
    'http': HttpAdapter
}

def download_object(adapter:str, dest_path:str=None, **kwargs):
    assert adapter in __adapters__.keys(),f"given adapter {adapter} not defined"
    return __adapters__[adapter].download(dest_path, **kwargs)