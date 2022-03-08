import os
from typing import Dict

import yaml

__all__ = [
    "get_pkg_root_path",
    "get_pkg_arch_path",
    "get_registry_path",
    "get_registry",
]

__ROOT_PATH__ = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2])


def get_pkg_root_path() -> str:
    global __ROOT_PATH__
    return __ROOT_PATH__


def get_pkg_arch_path() -> str:
    root_path = get_pkg_root_path()
    return os.path.join(root_path, "arch")


def get_registry_path() -> str:
    root_path = get_pkg_root_path()
    return os.path.join(root_path, "registry.yaml")


def get_registry() -> Dict:
    registry_path = get_registry_path()
    with open(registry_path, "r") as foo:
        registry = yaml.load(foo, Loader=yaml.FullLoader)
    return registry
