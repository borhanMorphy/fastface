import itertools
import os
from typing import List, Tuple
import random

from cv2 import cv2
import numpy as np
import torch

import fastface as ff

__VALID_IMG_EXTS__ = (".jpeg", ".jpg", ".png")

extract_ext = lambda file_name: os.path.splitext(file_name.lower())[1]


def build_module_args() -> Tuple:
    for arch in ff.list_archs():
        for config in ff.list_arch_configs(arch):
            yield (arch, config)


def mixup_arguments(*args) -> List:
    """mixups given arguments
    [argument_1_1, argument_1_2], [argument_2_1] =>
    [(argument_1_1, argument_2_1), (argument_1_2, argument_2_1)]

    Returns:
        List: [(arg1, arg2), ...]
    """
    return list(itertools.product(*args))


def get_img_paths() -> List:
    return [
        os.path.join("tests/data/", file_name)
        for file_name in os.listdir("tests/data/")
        if extract_ext(file_name) in __VALID_IMG_EXTS__
    ]


def load_image(img_file_path: str) -> np.ndarray:
    return cv2.imread(img_file_path)[..., [2, 1, 0]]


def load_image_as_tensor(img_file_path: str) -> torch.Tensor:
    img = load_image(img_file_path)
    return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous()


def generate_batch(shape: Tuple[int, int, int]) -> torch.Tensor:
    batch_size = random.randint(1, 5)
    return torch.rand(*(batch_size, *shape), dtype=torch.float32)
