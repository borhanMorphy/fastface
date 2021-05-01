import fastface as ff
import pytest
import pytorch_lightning as pl
from .utils import build_module_args

@pytest.mark.parametrize("api",
    [
        "build", "from_checkpoint", "from_pretrained",
        "to_tensor", "to_json"
    ]
)
def test_api_exists(api):
    assert api in dir(ff.FaceDetector), "{} not found in the fastface.FaceDetector".format(api)

@pytest.mark.parametrize("arch,config", list(build_module_args()))
def test_module_build(arch: str, config: str):
    module = ff.FaceDetector.build(arch, config)
    assert isinstance(module, pl.LightningModule), "module must be instance of pl.LightningModule but found:{}".format(type(module))
    config = ff.get_arch_config(arch, config)
    module = ff.FaceDetector.build(arch, config)
    assert isinstance(module, pl.LightningModule), "module must be instance of pl.LightningModule but found:{}".format(type(module))