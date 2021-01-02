import pytest
import mypackage
import pytorch_lightning as pl
from .utils import build_module_args

@pytest.mark.parametrize("api",
    ["build","from_checkpoint","from_pretrained"])
def test_api_exists(api):
    assert api in dir(mypackage.module),f"{api} not found in the mypackage.module"

@pytest.mark.parametrize("arch,config", list(build_module_args()))
def test_module_build(arch:str, config:str):
    module = mypackage.module.build(arch, config)
    assert isinstance(module, pl.LightningModule),f"module must be instance of pl.LightningModule but found:{type(module)}"
    config = mypackage.get_arch_config(arch,config)
    module = mypackage.module.build(arch, config)
    assert isinstance(module, pl.LightningModule),f"module must be instance of pl.LightningModule but found:{type(module)}"