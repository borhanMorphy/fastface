import random
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageColor, ImageDraw


def render_predictions(
    img: np.ndarray, preds: Dict, color: Tuple[int, int, int] = None
) -> Image:
    """Returns Rendered PIL Image using given predictions
    Args:
        img (np.ndarray): 3 channeled image
        preds (Dict): predictions as {'boxes':[[x1,y1,x2,y2], ...], 'scores':[<float>, ..]}
        color (Tuple[int,int,int], optional): color of the boundaries. if None that it will be random color.

    Returns:
        Image: 3 channeled pil image
    """
    if color is None:
        color = random.choice(list(ImageColor.colormap.keys()))
    pil_img = Image.fromarray(img)

    # TODO use score
    for (x1, y1, x2, y2), score in zip(preds["boxes"], preds["scores"]):
        ImageDraw.Draw(pil_img).rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
    return pil_img


def render_targets(
    img: np.ndarray, targets: Dict, color: Tuple[int, int, int] = None
) -> Image:
    """Returns Rendered PIL Image using given targets
    Args:
        img (np.ndarray): 3 channeled image
        targets (Dict): {'target_boxes':[[x1,y1,x2,y2], ...], ...}
        color (Tuple[int,int,int], optional): color of the boundaries. if None that it will be random color.

    Returns:
        Image: 3 channeled pil image
    """
    if color is None:
        color = random.choice(list(ImageColor.colormap.keys()))
    pil_img = Image.fromarray(img)
    for x1, y1, x2, y2 in targets["target_boxes"].astype(np.int32):
        ImageDraw.Draw(pil_img).rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
    return pil_img


def draw_rects(
    img: np.ndarray, boxes: np.ndarray, color: Tuple[int, int, int] = None
) -> np.ndarray:
    """Returns Rendered numpy image using given boxes
    Args:
        img (np.ndarray): 3 channeled image
        boxes (np.ndarray): with shape N,4 as xmin,ymin,xmax,ymax
        color (Tuple[int,int,int], optional): color of the boundaries. if None that it will be random color.

    Returns:
        Image: 3 channeled pil image
    """
    if color is None:
        color = random.choice(list(ImageColor.colormap.keys()))
    pil_img = Image.fromarray(img)
    for x1, y1, x2, y2 in boxes.astype(np.int32):
        ImageDraw.Draw(pil_img).rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
    return np.array(pil_img)
