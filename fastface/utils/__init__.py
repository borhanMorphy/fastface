import os
from typing import List

from packaging.version import InvalidVersion, Version

from ..version import __version__
from . import box, cache, config, process, vis


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

    versions = filter(lambda version: isinstance(version, Version), versions)

    return [
        str(version)
        for version in sorted(versions, reverse=True)
        if include_current_version or version != Version(__version__)
    ]


__all__ = [
    "box",
    "cache",
    "config",
    "process",
    "vis",
    "discover_versions",
]
