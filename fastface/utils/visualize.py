from cv2 import cv2
import numpy as np
from random import randint
from typing import Tuple,List,Dict

def prettify_detections(img:np.ndarray, preds:List[Dict],
        color:Tuple[int,int,int]=None) -> np.ndarray:
    """
    Args:
        img (np.ndarray): 3 channeled image
        preds (Dict): predictions as [{'box':[x1,y1,x2,y2], 'score':<float>}, ...]
        color (Tuple[int,int,int], optional): color of the boundaries. if None that it will be random color.

    Returns:
        np.ndarray: 3 channeled image
    """
    color = tuple((randint(0,255) for _ in range(3))) if isinstance(color,type(None)) else color
    rimg = img.copy()
    for pred in preds:
        x1,y1,x2,y2 = pred['box']
        rimg = cv2.rectangle(rimg, (x1,y1), (x2,y2), color, thickness=2, lineType=cv2.LINE_AA)
    return rimg