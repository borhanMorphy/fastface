from cv2 import cv2
import numpy as np
from random import randint
from typing import Tuple

def prettify_detections(img:np.ndarray, pred_boxes:np.ndarray,
        color:Tuple[int,int,int]=None) -> np.ndarray:
    """
    Args:
        img (np.ndarray): 3 channeled image
        pred_boxes (np.ndarray): prediction boxes np.ndarray(N,5) as x1,y1,x2,y2,score
        color (Tuple[int,int,int], optional): color of the boundaries. if None that it will be random color.

    Returns:
        np.ndarray: 3 channeled image
    """
    color = tuple((randint(0,255) for _ in range(3))) if isinstance(color,type(None)) else color
    rimg = img.copy()
    for *box,_ in pred_boxes:
        x1,y1,x2,y2 = (int(b) for b in box)

        rimg = cv2.rectangle(rimg, (x1,y1), (x2,y2), color, thickness=2, lineType=cv2.LINE_AA)
    return rimg