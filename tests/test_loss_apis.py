import fastface as ff
import pytest

@pytest.mark.parametrize("api",
    [
        "list_losses",
        "get_loss_by_name"
    ]
)
def test_api_exists(api):
    assert api in dir(ff.loss),f"{api} not found in the fastface.loss"

def test_get_available_losses():
    losses = ff.loss.list_losses()
    assert isinstance(losses, list),f"returned value must be list but found:{type(losses)}"
    for loss in losses:
        assert isinstance(loss,str),f"loss must contain name as string but found:{type(loss)}"

@pytest.mark.parametrize("loss_name", ff.loss.list_losses())
def test_loss_build(loss_name:str):
    loss = ff.loss.get_loss_by_name(loss_name)
    assert loss is not None,f"returned value must be object but found None"