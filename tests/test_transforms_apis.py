import pytest

import fastface as ff

from . import utils


@pytest.mark.parametrize(
    "transform_name",
    [
        # regular
        "Compose",
        "FaceDiscarder",
        "ConditionalInterpolate",
        "Interpolate",
        "Padding",
        "Rotate",
        # augmentations
        "RandomGaussianBlur",
        "ColorJitter",
        "LFFDRandomSample",
        "RandomHorizontalFlip",
        "RandomRotate",
    ],
)
def test_api_exists(transform_name: str):
    assert hasattr(
        ff.transforms, transform_name
    ), "{} not found in the fastface.transforms".format(transform_name)


@pytest.mark.parametrize("img_file_path", utils.get_img_paths())
def test_interpolate_call(img_file_path: str):
    img = utils.load_image(img_file_path)
    target_size = 480
    interpolate = ff.transforms.Interpolate(target_size=target_size)
    result_img, _ = interpolate(img)
    assert (
        max(result_img.shape) == target_size
    ), "expected max dim to be {} but found {}".format(
        target_size, max(result_img.shape)
    )


@pytest.mark.parametrize("img_file_path", utils.get_img_paths())
def test_padding_call(img_file_path: str):
    img = utils.load_image(img_file_path)
    target_size = max(img.shape)
    target_size = (target_size, target_size)
    padding = ff.transforms.Padding(target_size=target_size)
    result_img, _ = padding(img)
    assert (
        result_img.shape[:2] == target_size
    ), "expected image shape to \
        be {} but found {}".format(
        target_size, max(result_img.shape[:2])
    )
