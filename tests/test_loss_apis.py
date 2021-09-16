import pytest
import torch

import fastface as ff

# TODO expand this


@pytest.mark.parametrize("loss_name", ["DIoULoss", "BinaryFocalLoss"])
def test_loss_api_exists(loss_name: str):
    assert hasattr(ff.loss, loss_name), "{} not found in the fastface.loss".format(
        loss_name
    )


@pytest.mark.parametrize("loss_name", ["DIoULoss", "BinaryFocalLoss"])
def test_loss_build(loss_name: str):
    loss_cls = getattr(ff.loss, loss_name)
    loss_fn = loss_cls()
    assert isinstance(
        loss_fn, torch.nn.Module
    ), "loss must contain name as string but found:{}".format(type(loss_fn))
