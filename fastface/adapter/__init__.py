import logging

from .gdrive import GoogleDriveAdapter
from .http import HttpAdapter

logger = logging.getLogger("fastface.adapter")

__adapters__ = {"gdrive": GoogleDriveAdapter, "http": HttpAdapter}


def download_object(adapter: str, dest_path: str = None, **kwargs):
    assert adapter in __adapters__.keys(), "given adapter {} is not defined".format(
        adapter
    )
    logger.info("Downloading object to {} with {} adapter".format(dest_path, adapter))
    return __adapters__[adapter].download(dest_path, **kwargs)


__all__ = ["download_object"]
