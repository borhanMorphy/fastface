from typing import List
import os
from packaging.version import (
    Version,
    InvalidVersion,
)

from . import box, cache, cluster, config, geo, kernel, preprocess, random, vis
from ..version import __version__

def discover_versions(include_current_version: bool = True) -> List[str]:
    """Returns all versions stored in the `cache` directory

    Args:
        include_current_version (bool, optional): include current version or not. Defaults to True.

    Returns:
        List[str]: list of string versions sorted latest to oldest
    """
    versions = os.listdir(cache.get_root_cache_dir())
    versions.append(__version__)

    for i, version in enumerate(list(set(versions))):
        try:
            versions[i] = Version(version)
        except InvalidVersion:
            pass

    versions = filter(
        lambda version: isinstance(version, Version),
        versions
    )

    return [
        str(version)
        for version in sorted(versions, reverse=True)
        if include_current_version or version != Version(__version__)
    ]

__all__ = [
    "box",
    "cache",
    "cluster",
    "config",
    "geo",
    "kernel",
    "preprocess",
    "random",
    "vis",
    "discover_versions",
]
