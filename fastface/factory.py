from typing import List, Dict, Union
from .config import ArchConfig
from .arch.base import ArchInterface

class _Factory:
    """Factory for different types of face detection architectures"""

    archs = dict()

    def __init__(self) -> None:
        raise AssertionError("Do not create instance of this class")

    @staticmethod
    def register(name: str = None, cls = None, configs: Dict[str, ArchConfig] = None):
        _Factory.archs[name] = dict(
            cls=cls,
            configs=configs or dict()
        )

    @staticmethod
    def get_arch_names() -> List[str]:
        return sorted(list(_Factory.archs.keys()))

    @staticmethod
    def get_arch_config_names(arch: str) -> List[str]:
        assert arch in _Factory.archs, "given architecture {} is not found in the registry".format(arch)
        return sorted(list(_Factory.archs[arch]["configs"].keys()))

    @staticmethod
    def get_arch_config(arch: str, config: str) -> ArchConfig:
        assert arch in _Factory.archs, "given architecture {} is not found in the registry".format(arch)
        assert config in _Factory.archs[arch]["configs"], "given configuration {} is not found in the {}".format(config, arch)
        return _Factory.archs[arch]["configs"][config].copy()

    @staticmethod
    def build_arch(arch: str, config: Union[str, ArchConfig]) -> ArchInterface:
        config = _Factory.get_arch_config(arch, config) if isinstance(config, str) else config

        assert arch in _Factory.archs, "given architecture {} is not found in the registry".format(arch)

        return _Factory.archs[arch]["cls"](config)
