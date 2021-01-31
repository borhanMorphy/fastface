import fastface as ff
import pytest
from typing import List,Dict

@pytest.mark.parametrize("api",
    [
        "list_pretrained_models","download_pretrained_model",
        "list_archs","list_arch_configs","get_arch_config"
    ]
)
def test_api_exists(api):
    assert api in dir(ff),f"{api} not found in the fastface"

def test_list_pretrained_models():
    models = ff.list_pretrained_models()
    assert isinstance(models,List),f"returned value must be list but found:{type(models)}"
    for model in models:
        assert isinstance(model,str),f"pretrained model must contain name as string but found:{type(model)}"

def test_list_archs():
    archs = ff.list_archs()
    assert isinstance(archs,List),f"returned value must be list but found:{type(archs)}"
    for arch in archs:
        assert isinstance(arch,str),f"architecture must contain name as string but found:{type(arch)}"

@pytest.mark.parametrize("arch", ff.list_archs())
def test_list_arch_configs(arch:str):
    arch_configs = ff.list_arch_configs(arch)
    assert isinstance(arch_configs,List),f"returned value must be list but found:{type(arch_configs)}"
    for arch_config in arch_configs:
        assert isinstance(arch_config,str),f"architecture config must contain string but found:{type(arch_config)}"

@pytest.mark.parametrize("arch", ff.list_archs())
def test_get_arch_config(arch:str):
    arch_configs = ff.list_arch_configs(arch)
    for arch_config in arch_configs:
        config = ff.get_arch_config(arch, arch_config)
        assert isinstance(config,Dict),f"{arch}.{arch_config} must be dictionary but found: {type(config)}"