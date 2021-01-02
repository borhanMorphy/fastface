import mypackage
from typing import Tuple

def build_module_args() -> Tuple:
    for arch in mypackage.list_archs():
        for config in mypackage.list_arch_configs(arch):
            yield (arch,config)