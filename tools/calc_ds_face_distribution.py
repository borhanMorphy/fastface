import sys;sys.path.append('./')
from datasets import get_dataset, get_available_datasets
from utils.utils import seed_everything

from transforms import (
    Compose,
    Interpolate,
    Padding,
    FaceDiscarder,
    Normalize,
    ToTensor,
    LFFDRandomSample,
    RandomHorizontalFlip
)

import torch
from cv2 import cv2
import numpy as np
from tqdm import tqdm
from typing import List,Tuple
import math

class FaceScaleDistributionHandler():
    def __init__(self, scales:List[Tuple[int,int]]):
        self.scales = scales
        self.branch_faces = {i:0 for i in range(len(scales))}
        self.gray_scale_for_each_branch = {idx: self.get_gray_scale_range(l,u) for idx,(l,u) in enumerate(scales)}
        self.gray_scale_counter = {i:0 for i in range(len(scales))}
        self.lower_out_of_bounds = 0
        self.upper_out_of_bounds = 0
        self.min_scale = scales[0][0]
        self.max_scale = scales[-1][1]
        self.total_faces = 0

    @staticmethod
    def get_gray_scale_range(sl:int, su:int) -> Tuple[Tuple[int,int],Tuple[int,int]]:
        sl_range = (int(math.floor(sl * 0.9)), sl)
        su_range = (su, int(math.ceil(su * 1.1)))
        return sl_range,su_range

    def __call__(self, boxes:np.ndarray):
        for x1,y1,x2,y2 in boxes:
            self.total_faces += 1
            face_scale = max(y2-y1,x2-x1)
            if face_scale < self.min_scale:
                self.lower_out_of_bounds += 1
                continue
            elif face_scale >= self.max_scale:
                self.upper_out_of_bounds += 1
                continue

            for scale_idx,(lower_bound,upper_bound) in enumerate(self.scales):
                #[lower, upper)
                gl,gu = self.gray_scale_for_each_branch[scale_idx]
                if (gl[0] <= face_scale <= gl[0]) or (gu[0] <= face_scale <= gu[1]):
                    self.gray_scale_counter[scale_idx] += 1

                if lower_bound < face_scale < upper_bound:
                    self.branch_faces[scale_idx] += 1
                    break


    def __str__(self):
        results = []
        for scale_idx,(lower_bound,upper_bound) in enumerate(self.scales):
            num = self.branch_faces[scale_idx]
            results.append(f"\tbranch {scale_idx+1} low:{lower_bound} \
                hight:{upper_bound} number of faces: {num} %{int((num*100)/self.total_faces)} gray scales: {self.gray_scale_counter[scale_idx]}")

        results.append(f"\tout of bounds with lower: {self.min_scale} \
            number of faces: {self.lower_out_of_bounds} %{int((self.lower_out_of_bounds*100)/self.total_faces)}")

        results.append(f"\tout of bounds with higher: {self.max_scale} \
            number of faces: {self.upper_out_of_bounds} %{int((self.upper_out_of_bounds*100)/self.total_faces)}")

        return "\n".join(results)

if __name__ == '__main__':

    seed_everything(42)

    scales = [
        (10,15),(15,20),(20,40),(40,70),
        (70,110),(110,250),(250,400),(400,560)
    ]

    # TODO add gray scales
    fsd_handler = FaceScaleDistributionHandler(scales)

    transforms = Compose(
        FaceDiscarder(min_face_scale=2),
        LFFDRandomSample(scales, target_size=(640,640)),
        FaceDiscarder(min_face_scale=8)
    )

    ds = get_dataset("widerface", phase="train", transforms=transforms)

    for img,boxes in tqdm(ds):
        """
        img = img[:,:,[2,1,0]]
        img = cv2.UMat(img)
        for x1,y1,x2,y2 in boxes.astype(np.int32):
            img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("",img)
        if cv2.waitKey(0) == 27: break
        """
        assert img.shape[0] == img.shape[1]
        assert img.shape[0] == 640
        assert img.shape[1] == 640
        assert img.shape[2] == 3
        fsd_handler(boxes)

    print(fsd_handler)