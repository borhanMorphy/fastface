import tempfile

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

import fastface as ff

from . import utils


@pytest.mark.parametrize(
    "api", ["build", "from_checkpoint", "from_pretrained"]
)
def test_api_exists(api):
    assert api in dir(
        ff.FaceDetector
    ), "{} not found in the fastface.FaceDetector".format(api)


@pytest.mark.parametrize("arch,config", list(utils.build_module_args()))
def test_module_build(arch: str, config: str):
    module = ff.FaceDetector.build(arch, config)
    assert isinstance(
        module, pl.LightningModule
    ), "module must be instance of pl.LightningModule but found:{}".format(type(module))
    config = ff.get_arch_config(arch, config)
    module = ff.FaceDetector.build(arch, config)
    assert isinstance(
        module, pl.LightningModule
    ), "module must be instance of pl.LightningModule but found:{}".format(type(module))


@pytest.mark.parametrize("arch,config", list(utils.build_module_args()))
def test_module_export_to_torchscript(arch: str, config: str):
    module = ff.FaceDetector.build(arch, config)
    module.eval()

    sc_module = module.to_torchscript()
    assert isinstance(
        sc_module, torch.jit.ScriptModule
    ), "build failed \
        for {} with config {}".format(
        arch, config
    )

    dummy_input = torch.rand(2, *module.arch.input_shape[1:])

    output = module.forward(dummy_input)

    sc_output = sc_module.forward(dummy_input)

    assert (
        output == sc_output
    ).all(), "module output and exported module output \
        does not match for {} with config {}".format(
        arch, config
    )


@pytest.mark.parametrize("arch,config", list(utils.build_module_args()))
def test_module_export_to_onnx(arch: str, config: str):
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("skipping test, onnxruntime is not installed")

    module = ff.FaceDetector.build(arch, config)
    module.eval()

    opset_version = 11

    dynamic_axes = {
        "input_data": {0: "batch", 2: "height", 3: "width"},  # write axis names
        "preds": {0: "batch"},
    }

    input_names = ["input_data"]

    output_names = ["preds"]

    input_sample = torch.rand(1, *module.arch.input_shape[1:])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmpfile:

        module.to_onnx(
            tmpfile.name,
            input_sample=input_sample,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )

        sess = ort.InferenceSession(tmpfile.name)


    dummy_input = np.random.rand(2, *module.arch.input_shape[1:]).astype(np.float32)
    del module
    input_name = sess.get_inputs()[0].name
    (ort_output,) = sess.run(None, {input_name: dummy_input})

    assert (
        len(ort_output.shape) == 2
    ), "shape of the output must be length of 2 but found {}".format(
        len(ort_output.shape)
    )
    assert (
        ort_output.shape[1] == 6
    ), "shape of output must be N,6 but found N,{}".format(ort_output.shape[1])
