import fastface as ff
import pytest
import pytorch_lightning as pl
from .utils import build_module_args

@pytest.mark.parametrize("api",
    ["build","from_checkpoint","from_pretrained"])
def test_api_exists(api):
    assert api in dir(ff.module),f"{api} not found in the fastface.module"

@pytest.mark.parametrize("arch,config", list(build_module_args()))
def test_module_build(arch:str, config:str):
    module = ff.module.build(arch, config)
    assert isinstance(module, pl.LightningModule),f"module must be instance of pl.LightningModule but found:{type(module)}"
    config = ff.get_arch_config(arch,config)
    module = ff.module.build(arch, config)
    assert isinstance(module, pl.LightningModule),f"module must be instance of pl.LightningModule but found:{type(module)}"