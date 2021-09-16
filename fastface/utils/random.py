from typing import Tuple

import numpy as np
import torch

from .box import cxcywh2xyxy


def generate_uniform_boxes(
    center_range: Tuple[float, float] = (0.1, 0.9),
    wh_range: Tuple[float, float] = (0.2, 0.8),
    n: int = 100,
):

    # TODO pydoc

    cxcy = np.random.uniform(low=center_range[0], high=center_range[1], size=(n, 2))
    wh = np.random.uniform(low=wh_range[0], high=wh_range[1], size=(n, 2))

    boxes = np.concatenate([cxcy, wh], axis=1).astype(np.float32)

    boxes = cxcywh2xyxy(torch.from_numpy(boxes))

    return boxes.clamp(min=0, max=1)
