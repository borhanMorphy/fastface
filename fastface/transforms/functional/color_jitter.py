import numpy as np
from PIL import Image, ImageEnhance


def adjust_brightness(img: np.ndarray, factor: float = 0.0) -> np.ndarray:
    # between [-1, 1] 0 is same image -1 is darken, 1 is brighten
    factor = min(factor, 1)
    factor = max(factor, -1)

    pimg = ImageEnhance.Brightness(Image.fromarray(img)).enhance(factor + 1)
    return np.array(pimg)


def adjust_contrast(img: np.ndarray, factor: float = 0.0) -> np.ndarray:
    # between [-1, 1] 0 is same image -1 is lower, 1 is higher
    factor = min(factor, 1)
    factor = max(factor, -1)

    pimg = ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor + 1)
    return np.array(pimg)


def adjust_saturation(img: np.ndarray, factor: float = 0.0) -> np.ndarray:
    # between [-1, 1] 0 is same image -1 is lower, 1 is higher
    factor = min(factor, 1)
    factor = max(factor, -1)

    pimg = ImageEnhance.Color(Image.fromarray(img)).enhance(factor + 1)
    return np.array(pimg)
