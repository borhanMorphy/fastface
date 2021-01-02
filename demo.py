import torch
import argparse
from cv2 import cv2
import numpy as np
from typing import List

import mypackage
from mypackage.utils.visualize import prettify_detections
from mypackage.transform import (
    Compose,
    Interpolate,
    Padding,
    Normalize,
    ToTensor
)

def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument('--input', '-i', type=str,
        required=True, help='image file path')

    ap.add_argument('--det-threshold', '-dt', type=float,
        default=.8, help='detection score threshold')

    ap.add_argument('--iou-threshold', '-it', type=float,
        default=.4, help='iou score threshold')

    return ap.parse_args()

def load_image(img_path:str) -> np.ndarray:
    return cv2.imread(img_path)

def main(img_path:str, det_threshold:float, iou_threshold:float):
    # load image
    img = load_image(img_path)

    # get pretrained model
    model = mypackage.module.from_pretrained(model="original_lffd_560_25L_8S")

    # build required transforms
    transforms = Compose(
        Interpolate(max_dim=640),
        Padding(target_size=(640,640)),
        Normalize(mean=127.5, std=127.5),
        ToTensor()
    )

    # enable tracking to perform postprocess after inference 
    transforms.enable_tracking()

    # reset queue
    transforms.flush()

    # set model to eval mode
    model.eval()

    # freeze model in order to disable gradient tracking
    model.freeze()

    # apply transforms
    batch = transforms(img)

    # model feed forward
    preds = model.predict(batch, det_threshold=det_threshold,
        iou_threshold=iou_threshold)[0].cpu().numpy()

    # postprocess to adjust predictions
    preds = transforms.adjust(preds)

    # reset queue
    transforms.flush()

    # disable tracking
    transforms.disable_tracking()

    # visualize predictions
    pretty_img = prettify_detections(img, preds)
    cv2.imshow("demo", pretty_img)

    if cv2.waitKey(0) == 27: pass

if __name__ == '__main__':
    args = get_arguments()
    main(args.input, args.det_threshold, args.iou_threshold)