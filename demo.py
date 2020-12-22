import torch
import argparse
from cv2 import cv2
import numpy as np
from typing import List

from utils.visualize import prettify_detections
from detector import FaceDetector
from transforms import (
    Compose,
    Interpolate,
    Padding,
    Normalize,
    ToTensor
)
import archs

def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument('--arch', '-a', type=str, choices=archs.get_available_archs(),
        default='lffd', help='architecture to perform face detection')

    ap.add_argument('--config', '-c', type=str,
        default='560_25L_8S', help='architecture configuration')

    ap.add_argument('--input', '-i', type=str,
        required=True, help='image file path')

    ap.add_argument('--weights', '-w', type=str,
        help='model weights file path', default='./models/original_lffd_560_25L_8S.pt')

    ap.add_argument('--det-threshold', '-dt', type=float,
        default=.8, help='detection score threshold')

    ap.add_argument('--iou-threshold', '-it', type=float,
        default=.4, help='iou score threshold')

    return ap.parse_args()

def load_image(img_path:str):
    return cv2.imread(img_path)

def main(img_path:str, arch_name:str, config:str, weights:str,
        det_threshold:float, iou_threshold:float):
    img = load_image(img_path)
    model = FaceDetector.from_pretrained(arch_name, weights, config=config)

    transforms = Compose(
        Interpolate(max_dim=640),
        Padding(target_size=(640,640)),
        Normalize(mean=127.5, std=127.5),
        ToTensor()
    )

    transforms.enable_tracking()
    transforms.flush()

    model.eval()

    batch = transforms(img)

    preds = model.predict(batch, det_threshold=det_threshold,
        iou_threshold=iou_threshold)[0].cpu().numpy()

    preds = transforms.adjust(preds)

    transforms.flush()
    transforms.disable_tracking()

    pretty_img = prettify_detections(img, preds)
    cv2.imshow("",pretty_img)
    if cv2.waitKey(0) == 27:
        pass


if __name__ == '__main__':
    args = get_arguments()
    main(args.input, args.arch, args.config, args.weights,
        args.det_threshold, args.iou_threshold)