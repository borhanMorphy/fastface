import pytest
import torch
import fastface as ff
from . import utils

# TODO expand this

@pytest.mark.parametrize("preprocess_name", [
    "Interpolate", "DummyInterpolate",
    "Pad", "DummyPad",
    "Normalize",
    "Preprocess" # TODO
])
def test_api_exists(preprocess_name: str):
    assert hasattr(ff.preprocess, preprocess_name), "{} not found in the \
        fastface.preprocess".format(preprocess_name)

@pytest.mark.parametrize("img_file_path", utils.get_img_paths())
def test_interpolate_forward(img_file_path: str):
    data = utils.load_image_as_tensor(img_file_path)
    max_size = 480
    interpolate = ff.preprocess.Interpolate(max_size=max_size)
    assert isinstance(interpolate, torch.nn.Module), "interpolate op must be torch.nn.Module but found {}".format(type(interpolate))
    p_data, scale_factor = interpolate(data)
    assert isinstance(scale_factor, torch.Tensor), "scale factor must be tensor but found {}".format(type(scale_factor))
    assert len(scale_factor.shape) == 1, "scale factor must be shape of 1 but found {}".format(len(scale_factor.shape))
    assert max(p_data.shape) == max_size, "expected max dim to be {} but found {}".format(max_size, max(p_data.shape))

@pytest.mark.parametrize("img_file_path", utils.get_img_paths())
def test_dummy_interpolate_forward(img_file_path: str):
    data = utils.load_image_as_tensor(img_file_path)
    dummy_interpolate = ff.preprocess.DummyInterpolate()
    assert isinstance(dummy_interpolate, torch.nn.Module), "dummy_interpolate op must be torch.nn.Module but found {}".format(type(dummy_interpolate))
    p_data, scale_factor = dummy_interpolate(data)
    assert isinstance(scale_factor, torch.Tensor), "scale factor must be tensor but found {}".format(type(scale_factor))
    assert len(scale_factor.shape) == 1, "scale factor must be shape of 1 but found {}".format(len(scale_factor.shape))
    assert (p_data == data).all(), "output must be equal to input for `DummyInterpolate`"

@pytest.mark.parametrize("img_file_path", utils.get_img_paths())
def test_pad_forward(img_file_path: str):
    data = utils.load_image_as_tensor(img_file_path)
    target_size = (max(data.shape), max(data.shape))
    pad = ff.preprocess.Pad(target_size)
    assert isinstance(pad, torch.nn.Module), "Pad op must be torch.nn.Module but found {}".format(type(pad))
    p_data, paddings = pad(data)
    assert isinstance(paddings, torch.Tensor), "paddings must be tensor but found {}".format(type(paddings))
    assert len(paddings.shape) == 1, "paddings must be shape of 1 but found {}".format(len(paddings.shape))
    assert tuple(p_data.shape[2:]) == target_size, "expected shape to be {} but found {}".format(target_size, tuple(p_data.shape[2:]))

@pytest.mark.parametrize("img_file_path", utils.get_img_paths())
def test_dummy_pad_forward(img_file_path: str):
    data = utils.load_image_as_tensor(img_file_path)
    dummy_pad = ff.preprocess.DummyPad()
    assert isinstance(dummy_pad, torch.nn.Module), "dummy_pad op must be torch.nn.Module but found {}".format(type(dummy_pad))
    p_data, paddings = dummy_pad(data)
    assert isinstance(paddings, torch.Tensor), "paddings must be tensor but found {}".format(type(paddings))
    assert len(paddings.shape) == 1, "paddings must be shape of 1 but found {}".format(len(paddings.shape))
    assert (p_data == data).all(), "output must be equal to input for `DummyPad`"

@pytest.mark.parametrize("img_file_path", utils.get_img_paths())
def test_normalize_forward(img_file_path: str):
    data = utils.load_image_as_tensor(img_file_path)
    normalize = ff.preprocess.Normalize(mean=0, std=255)
    assert isinstance(normalize, torch.nn.Module), "normalize op must be torch.nn.Module but found {}".format(type(normalize))
    p_data = normalize(data)
    assert torch.max(p_data) <= 1, "expected image max value is to be lower than 1 but found {}".format(torch.max(p_data))
