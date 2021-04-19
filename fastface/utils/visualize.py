from PIL import Image, ImageDraw, ImageColor
import random
import numpy as np
from typing import Tuple, Dict

def prettify_detections(img:np.ndarray, preds:Dict,
        color:Tuple[int,int,int]=None) -> Image:
    """Returns Rendered PIL Image using given predictions 
    Args:
        img (np.ndarray): 3 channeled image
        preds (Dict): predictions as {'boxes':[[x1,y1,x2,y2], ...], 'scores':[<float>, ..]}
        color (Tuple[int,int,int], optional): color of the boundaries. if None that it will be random color.

    Returns:
        Image: 3 channeled pil image
    """
    color = random.choice(list(ImageColor.colormap.keys()))
    pil_img = Image.fromarray(img)
    for (x1,y1,x2,y2), score in zip(preds['boxes'],preds['scores']):
        ImageDraw.Draw(pil_img).rectangle([(x1,y1),(x2,y2)], outline=color, width=3)
    return pil_img