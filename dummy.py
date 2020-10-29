from cv2 import cv2
from models import get_detector_by_name
import torch
import numpy as np


if __name__ == "__main__":
    target_size = 640
    model = get_detector_by_name("lffd")

    canvas = np.ones((target_size,target_size,3), dtype=np.uint8)

    dummy = torch.rand((1,3,target_size,target_size), dtype=torch.float32)

    preds = model(dummy)

    for index,(cls_pred,reg_pred) in enumerate(preds):
        if index <= 5:
            continue
        _,_,fmaph,fmapw = cls_pred.shape
        rfs = model.heads[index].gen_rf_centers(fmaph,fmapw)
        _,j_,i_ = rfs.shape
        t_canvas = canvas.copy()
        for x,y in rfs.long().permute(1,2,0).reshape(-1,2):
            t_canvas = cv2.circle(t_canvas, (x,y), 5, (0,0,255))
            cv2.imshow("rfs", t_canvas)
            cv2.waitKey(0)


        


