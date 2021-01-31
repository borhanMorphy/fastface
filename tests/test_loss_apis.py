import fastface as ff
import pytest
from typing import List,Dict
import torch.nn as nn

@pytest.mark.parametrize("api",
    [
        "list_losses","get_loss_by_name"
    ]
)
def test_api_exists(api):
    assert api in dir(ff.loss),f"{api} not found in the fastface.loss"

def test_get_available_losses():
    losses = ff.loss.list_losses()
    assert isinstance(losses,List),f"returned value must be list but found:{type(losses)}"
    for loss in losses:
        assert isinstance(loss,str),f"loss must contain name as string but found:{type(loss)}"

@pytest.mark.parametrize("loss_name", ff.loss.list_losses())
def test_list_arch_configs(loss_name:str):
    loss = ff.loss.get_loss_by_name(loss_name)
    assert isinstance(loss, nn.Module),f"returned value must be loss but found:{type(loss)}"