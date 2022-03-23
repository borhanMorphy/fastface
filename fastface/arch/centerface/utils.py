from typing import Tuple

import numpy as np


def gaussian_radius(det_size: Tuple, min_overlap: float = 0.7) -> float:
    # code ref: https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/utils/image.py#L95
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def get_gaussian_kernel(
    kernel_width: int,
    kernel_height: int,
    sigma: float = 1.0,
    center_point: Tuple[float, float] = None,
) -> np.ndarray:
    """Generates gaussian kernel using 2D gaussian distribution

    Args:
        kernel_width (int): width of the kernel size as int
        kernel_height (int): height of the kernel size as int
        sigma (float, optional): sigma value of the gaussian kernel. Defaults to 1.0.
        center_point (Tuple[float, float], optional): mean data point of the distribution as x,y order. Defaults to None.

    Returns:
        np.ndarray: 2D kernel with shape kernel_height x kernel_width
    """
    ax = np.arange(kernel_width)
    ay = np.arange(kernel_height)
    xx, yy = np.meshgrid(ax, ay)

    if center_point is None:
        center_point = ((kernel_width - 1) / 2, (kernel_height - 1) / 2)
    center_point_x, center_point_y = center_point
    return np.exp(
        -0.5
        * (np.square(xx - center_point_x) + np.square(yy - center_point_y))
        / np.square(sigma)
    )
