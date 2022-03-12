from typing import List, Union
from .config import ArchConfig
from .arch.base import ArchInterface

class _Factory:
    """Factory for different types of face detection architectures"""

    archs = dict()

    def __init__(self) -> None:
        raise AssertionError("Do not create instance of this class")

    @staticmethod
    def register(cls = None, configs: List[ArchConfig] = None):
        assert len(configs) > 0, "configuration list is empty"

        arch_names = set([config.arch for config in configs])
        assert len(arch_names) == 1, "arch names does not match with given configs"

        arch_name, = arch_names

        if arch_name not in _Factory.archs:
            _Factory.archs[arch_name] = dict(
                cls=cls,
                configs=list()
            )

        _Factory.archs[arch_name]["configs"] += configs

    @staticmethod
    def get_arch_names() -> List[str]:
        return sorted(list(_Factory.archs.keys()))

    @staticmethod
    def get_arch_config_names(arch: str) -> List[str]:
        assert arch in _Factory.archs, "given architecture {} is not found in the registry".format(arch)
        return sorted(
            list(
                set(
                    map(lambda config: config.name, _Factory.archs[arch]["configs"])
                )
            )
        )

    @staticmethod
    def get_arch_config(arch: str, config: str) -> ArchConfig:
        assert config in _Factory.get_arch_config_names(arch), "given configuration {} is not found in the {}".format(config, arch)
        for arch_config in _Factory.archs[arch]["configs"]:
            if arch_config.name == config:
                return arch_config.copy()

        raise AssertionError("configuration not found")

    @staticmethod
    def build_arch(arch: str, config: Union[str, ArchConfig]) -> ArchInterface:
        config = _Factory.get_arch_config(arch, config) if isinstance(config, str) else config

        return _Factory.archs[arch]["cls"](config)
