from typing import Tuple
import numpy as np

from ..transforms import functional as F


def apply_conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    assert len(kernel.shape) == 2, "kernel shape must be 2D but found {}".format(
        len(kernel.shape)
    )
    img_h, img_w = img.shape[:2]
    nimg = []
    s = kernel.shape + tuple(np.subtract(img.shape[:2], kernel.shape) + 1)
    if len(img.shape) == 2:
        img = img[..., np.newaxis]

    for ch in range(img.shape[2]):
        subM = np.lib.stride_tricks.as_strided(
            img[..., ch], shape=s, strides=img[..., ch].strides * 2
        )
        nimg.append(np.einsum("ij,ijkl->kl", kernel, subM))

    nimg = np.stack(nimg, axis=2).astype(np.uint8)

    nimg, _ = F.pad(nimg, (img_w, img_h), pad_value=0)

    return nimg


def get_gaussian_kernel(kernel_size: int, sigma: float = 1.0,
        center_point: Tuple[float, float] = None, normalize: bool = True) -> np.ndarray:
    """Generates gaussian kernel using 2D gaussian distribution

    Args:
        kernel_size (int): kernel size
        sigma (float, optional): sigma value of the gaussian kernel. Defaults to 1.0.
        center_point (Tuple[float, float], optional): mean data point of the distribution as x,y order. Defaults to None.
        normalize (bool, optional): whether to normalize kernel or not. Defaults to True.

    Returns:
        np.ndarray: 2D kernel with shape kernel_size x kernel_size
    """
    ax = np.arange(kernel_size)
    xx, yy = np.meshgrid(ax, ax)

    if center_point is None:
        center_point = ((kernel_size - 1) / 2, (kernel_size - 1) / 2)
    center_point_x, center_point_y = center_point
    kernel = np.exp(-0.5 * (np.square(xx - center_point_x) + np.square(yy - center_point_y)) / np.square(sigma))
    if normalize:
        kernel = kernel / (np.pi * 2 * sigma ** 2)
        kernel /= kernel.sum()
    return kernel
