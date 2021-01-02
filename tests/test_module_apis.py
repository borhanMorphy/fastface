import pytest
import fastface
import pytorch_lightning as pl
from .utils import build_module_args

@pytest.mark.parametrize("api",
    ["build","from_checkpoint","from_pretrained"])
def test_api_exists(api):
    assert api in dir(fastface.module),f"{api} not found in the fastface.module"

@pytest.mark.parametrize("arch,config", list(build_module_args()))
def test_module_build(arch:str, config:str):
    module = fastface.module.build(arch, config)
    assert isinstance(module, pl.LightningModule),f"module must be instance of pl.LightningModule but found:{type(module)}"
    config = fastface.get_arch_config(arch,config)
    module = fastface.module.build(arch, config)
    assert isinstance(module, pl.LightningModule),f"module must be instance of pl.LightningModule but found:{type(module)}"