import os
import tempfile

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

import fastface as ff

from . import utils


@pytest.mark.parametrize(
    "api", ["build", "build_from_yaml", "from_checkpoint", "from_pretrained"]
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


@pytest.mark.parametrize("model_name", ff.list_pretrained_models())
def test_module_from_pretrained(model_name: str):
    module = ff.FaceDetector.from_pretrained(model_name)
    assert isinstance(
        module, pl.LightningModule
    ), "module must be instance of pl.LightningModule but found:{}".format(type(module))


@pytest.mark.parametrize("model_name", ff.list_pretrained_models())
def test_module_from_checkpoint(model_name: str):
    cache_path = ff.utils.cache.get_model_cache_dir()

    model_path = os.path.join(cache_path, model_name)

    if not os.path.exists(model_path):
        # download the model
        model_path = ff.download_pretrained_model(model_name, target_path=cache_path)

    module = ff.FaceDetector.from_checkpoint(model_path)
    assert isinstance(
        module, pl.LightningModule
    ), "module must be instance of pl.LightningModule but found:{}".format(type(module))


@pytest.mark.parametrize(
    "yaml_file_path", ["config_zoo/lffd_original.yaml", "config_zoo/lffd_slim.yaml"]
)
def test_module_build_from_yaml(yaml_file_path: str):
    module = ff.FaceDetector.build_from_yaml(yaml_file_path)
    assert isinstance(
        module, pl.LightningModule
    ), "module must be instance of pl.LightningModule but found:{}".format(type(module))


@pytest.mark.parametrize(
    "model_name, img_file_path",
    utils.mixup_arguments(ff.list_pretrained_models(), utils.get_img_paths()),
)
def test_module_predict(model_name: str, img_file_path: str):
    module = ff.FaceDetector.from_pretrained(model_name)
    module.eval()
    img = utils.load_image(img_file_path)
    preds = module.predict(img)
    assert isinstance(
        preds, list
    ), "prediction result must be list but found {}".format(type(preds))
    assert len(preds) == 1, "lenght of predictions must be 1 but found {}".format(
        len(preds)
    )


@pytest.mark.parametrize(
    "model_name, img_file_path",
    utils.mixup_arguments(ff.list_pretrained_models(), utils.get_img_paths()),
)
def test_module_forward(model_name: str, img_file_path: str):
    module = ff.FaceDetector.from_pretrained(model_name)
    module.eval()
    data = utils.load_image_as_tensor(img_file_path)
    preds = module.forward(data)
    # preds: N,6
    assert isinstance(
        preds, torch.Tensor
    ), "predictions must be tensor but found {}".format(type(preds))
    assert (
        len(preds.shape) == 2
    ), "prediction shape length must be 2 but found {}".format(len(preds.shape))
    assert (
        preds.shape[1] == 6
    ), "prediction shape index 1 must be 6 but found {}".format(preds.shape[1])
    assert (
        (preds[:, 4] >= 0) & (preds[:, 4] <= 1)
    ).all(), "predicton scores must be between 0 and 1"
    assert (preds[:, 5] == 0).all(), "batch dimension of all predictions must be 0"
    assert (preds[:, :4] >= 0).all(), "box dimensions of predictions must be positive"
    assert (
        (preds[:, [2, 3]] - preds[:, [0, 1]]) >= 0
    ).all(), "predicted box height and width must be greater than 0"


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

    dummy_input = torch.rand(2, 3, 480, 360)

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

    del module

    dummy_input = np.random.rand(2, 3, 200, 200).astype(np.float32)
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
