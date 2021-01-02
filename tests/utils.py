import fastface
from typing import Tuple

def build_module_args() -> Tuple:
    for arch in fastface.list_archs():
        for config in fastface.list_arch_configs(arch):
            yield (arch,config)