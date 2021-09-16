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


def get_gaussian_kernel(kernel_size: int, sigma: float = 1.0) -> np.ndarray:
    """Generates kernel using 2d gaussian distribution

    Args:
        kernel_size (int): size of the window
        sigma (float, optional): standard deviation of the distribution. Defaults to 1.0.

    Returns:
        np.ndarray: 2D kernel as (kernel_size x kernel_size)
    """
    ax = np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel = kernel / (np.pi * 2 * sigma ** 2)

    return kernel / kernel.sum()
