from models import get_detector_by_name
from datasets import get_dataset
from transforms import Interpolate,Padding
from utils.utils import seed_everything
from typing import Tuple

from tqdm import tqdm
from cv2 import cv2
import numpy as np
import torch

class Transforms():
    def __init__(self, *ts):
        self.ts = ts

    def __call__(self, img:np.ndarray, gt_boxes:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        for t in self.ts:
            img,gt_boxes = t(img,gt_boxes)
        return img,gt_boxes

if __name__ == "__main__":
    target_size = (640,640)

    model = get_detector_by_name("lffd").train()

    seed_everything(42)

    transforms = Transforms(
        Interpolate(max_dim=target_size[0]),
        Padding(target_size=target_size, pad_value=0)
    )

    ds = get_dataset("widerface", phase='val', partitions=['easy'], transforms=transforms)

    for img,boxes in tqdm(ds):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for x1,y1,x2,y2 in boxes.astype(np.int32):
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255))
        cv2.imshow("",img)
        if cv2.waitKey(0) == 27:
            break