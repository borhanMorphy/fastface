import os

import pytest
import torch
import pytorch_lightning as pl

import fastface as ff

from . import utils

@pytest.mark.parametrize("api",
    [
        "build", "build_from_yaml", "from_checkpoint", "from_pretrained"
    ]
)
def test_api_exists(api):
    assert api in dir(ff.FaceDetector), "{} not found in the fastface.FaceDetector".format(api)

@pytest.mark.parametrize("arch,config", list(utils.build_module_args()))
def test_module_build(arch: str, config: str):
    module = ff.FaceDetector.build(arch, config)
    assert isinstance(module, pl.LightningModule), "module must be instance of pl.LightningModule but found:{}".format(type(module))
    config = ff.get_arch_config(arch, config)
    module = ff.FaceDetector.build(arch, config)
    assert isinstance(module, pl.LightningModule), "module must be instance of pl.LightningModule but found:{}".format(type(module))

@pytest.mark.parametrize("model_name", ff.list_pretrained_models())
def test_module_from_pretrained(model_name: str):
    module = ff.FaceDetector.from_pretrained(model_name)
    assert isinstance(module, pl.LightningModule), "module must be instance of pl.LightningModule but found:{}".format(type(module))

@pytest.mark.parametrize("model_name", ff.list_pretrained_models())
def test_module_from_checkpoint(model_name: str):
    cache_path = ff.utils.cache.get_model_cache_dir()

    model_path = os.path.join(cache_path, model_name)

    if not os.path.exists(model_path):
        # download the model
        model_path = ff.download_pretrained_model(model_name, target_path=cache_path)

    module = ff.FaceDetector.from_checkpoint(model_path)
    assert isinstance(module, pl.LightningModule), "module must be instance of pl.LightningModule but found:{}".format(type(module))

@pytest.mark.parametrize("yaml_file_path", [
    "config_zoo/lffd_original.yaml",
    "config_zoo/lffd_slim.yaml"
])
def test_module_build_from_yaml(yaml_file_path: str):
    module = ff.FaceDetector.build_from_yaml(yaml_file_path)
    assert isinstance(module, pl.LightningModule), "module must be instance of pl.LightningModule but found:{}".format(type(module))

@pytest.mark.parametrize("model_name, img_file_path",
    utils.mixup_arguments(ff.list_pretrained_models(), utils.get_img_paths())
)
def test_module_predict(model_name: str, img_file_path: str):
    module = ff.FaceDetector.from_pretrained(model_name)
    module.eval()
    img = utils.load_image(img_file_path)
    preds = module.predict(img)
    assert isinstance(preds, list), "prediction result must be list but found {}".format(type(preds))
    assert len(preds) == 1, "lenght of predictions must be 1 but found {}".format(len(preds))

@pytest.mark.parametrize("model_name, img_file_path",
    utils.mixup_arguments(ff.list_pretrained_models(), utils.get_img_paths())
)
def test_module_forward(model_name: str, img_file_path: str):
    module = ff.FaceDetector.from_pretrained(model_name)
    module.eval()
    data = utils.load_image_as_tensor(img_file_path)
    preds = module.forward(data)
    # preds: N,6
    assert isinstance(preds, torch.Tensor), "predictions must be tensor but found {}".format(type(preds))
    assert len(preds.shape) == 2, "prediction shape length must be 2 but found {}".format(len(preds.shape))
    assert preds.shape[1] == 6, "prediction shape index 1 must be 6 but found {}".format(preds.shape[1])
    assert ((preds[:, 4] >= 0) & (preds[:, 4] <= 1)).all(), "predicton scores must be between 0 and 1"
    assert (preds[:, 5] == 0).all(), "batch dimension of all predictions must be 0"
    assert (preds[:, :4] >= 0).all(), "box dimensions of predictions must be positive"
    assert ((preds[:, [2, 3]] - preds[:, [0, 1]]) >= 0).all(), "predicted box height and width must be greater than 0"
