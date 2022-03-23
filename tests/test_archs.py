import pytest
import torch
import torch.nn as nn
import fastface as ff

from . import utils


@pytest.mark.parametrize("arch_name, config_name", utils.build_module_args())
def test_arch_build(arch_name: str, config_name: str):
    model = ff.build_arch(arch_name, config_name)

    assert isinstance(model, nn.Module)


@pytest.mark.parametrize("arch_name, config_name", utils.build_module_args())
def test_arch_input_shape_prop(arch_name: str, config_name: str):
    model = ff.build_arch(arch_name, config_name)

    assert len(model.input_shape) == 3, "expected length of `input_shape` to be 3 but found {}".format(model.input_shape)

    channel, _, _ = model.input_shape

    assert channel == 1 or channel == 3, "input channel can only be `1` or `3` but found {}".format(channel)


@pytest.mark.parametrize("arch_name, config_name", utils.build_module_args())
def test_arch_forward(arch_name: str, config_name: str):
    model = ff.build_arch(arch_name, config_name)

    batch = utils.generate_batch(model.input_shape)

    model.forward(batch)
    assert True

@pytest.mark.parametrize("arch_name, config_name", utils.build_module_args())
def test_arch_config(arch_name: str, config_name: str):
    model = ff.build_arch(arch_name, config_name)
    assert hasattr(model, "config")


@pytest.mark.parametrize("arch_name, config_name", utils.build_module_args())
def test_arch_compute_preds(arch_name: str, config_name: str):
    model = ff.build_arch(arch_name, config_name)
    batch = utils.generate_batch(model.input_shape)

    logits = model.forward(batch)

    preds = model.compute_preds(logits)
    # preds: B x N x (5 + 2*l)

    assert isinstance(preds, torch.Tensor), "prediction expected to be `torch.Tensor` but found {}".format(type(preds))
    assert len(preds.shape) == 3, "prediction shape expected to be B x N x (5 + 2*l) but found {}".format(preds.shape)
    assert preds.shape[0] == batch.shape[0], "prediction first dimension must be equal to batch size but found {}".format(preds.shape[0])
    assert preds.shape[2] >= 5, "prediction third dimension must be equal or greater then 5 but found {}".format(preds.shape[2])


def test_arch_build_targets():
    # check build_targets
    """
    check if after giving raw targets, make sure it builds correct shape 
    """
    pass # TODO


def test_arch_compute_loss():
    # check compute_loss
    """
    check if after giving `forward` output (logits) and `build_targets` output (target_logits) it computes the loss
    """
    pass # TODO

