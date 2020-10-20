import numpy as np
from typing import List,Tuple
import random
from cv2 import cv2

class LFFDRandomSample():
    def __init__(self, scales:List[List], target_size:Tuple[int,int]=(640,640), min_dim:int=10):
        self.scales = np.array(scales, dtype=np.float32) # N,2
        self.target_size = target_size # W,H
        self.min_dim = min_dim

    def __call__(self, img:np.ndarray, boxes:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly samples faces using given scales. All scales represents branches and
        for each branch selection probability is same.

        Args:
            img (np.ndarray): H,W,C
            boxes (np.ndarray): N,4 as xmin,ymin,xmax,ymax

        Returns:
            Tuple[np.ndarray, np.ndarray]: transformed image and transformed boxes
        """

        if boxes.shape[0] == 0: return img,boxes

        mask = np.bitwise_and((boxes[:, 2] - boxes[:, 0] >= self.min_dim), (boxes[:, 3] - boxes[:, 1] >= self.min_dim))
        boxes = boxes[mask]

        num_faces = boxes.shape[0]
        if num_faces == 0: return img,boxes

        # select one face

        selected_face_idx = random.randint(0, num_faces-1)

        selected_face_scale_idx = random.randint(0, self.scales.shape[0]-1)
        scale_lower,scale_higher = self.scales[selected_face_scale_idx]
        scale_size = random.randint(scale_lower,scale_higher)
        sf = scale_size / (boxes[selected_face_idx][2]-boxes[selected_face_idx][0])
        nimg = cv2.resize(img, None, fx=sf, fy=sf)
        nboxes = boxes * sf

        center_x = int((nboxes[selected_face_idx][2] + nboxes[selected_face_idx][0]) / 2)
        center_y = int((nboxes[selected_face_idx][3] + nboxes[selected_face_idx][1]) / 2)

        timg = np.zeros((*self.target_size,3), dtype=np.uint8)
        crop_xmin = max((center_x-self.target_size[0])//2,0)
        crop_ymin = max((center_y-self.target_size[1])//2,0)
        crop_xmax = min((center_x+self.target_size[0])//2,nimg.shape[1])
        crop_ymax = min((center_y+self.target_size[1])//2,nimg.shape[0])

        leftover_x = self.target_size[0]-(crop_xmax-crop_xmin)

        offset_x_left = leftover_x // 2 + leftover_x % 2
        offset_x_right = leftover_x // 2

        leftover_y = self.target_size[1]-(crop_ymax-crop_ymin)
        offset_y_up = leftover_y // 2 + leftover_y % 2
        offset_y_down = leftover_y // 2

        timg[offset_y_up:self.target_size[1]-offset_y_down, offset_x_left:self.target_size[0]-offset_x_right] = \
            nimg[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]

        center_shift_x = center_x - nimg.shape[1] // 2 
        center_shift_y = center_y - nimg.shape[0] // 2

        nboxes[:,[0,2]] = nboxes[:,[0,2]] + center_shift_x
        nboxes[:,[1,3]] = nboxes[:,[1,3]] + center_shift_y

        nboxes[:,[0,2]] = np.clip(nboxes[:,[0,2]], 0, self.target_size[0])
        nboxes[:,[1,3]] = np.clip(nboxes[:,[1,3]], 0, self.target_size[1])
        wh = np.zeros((nboxes.shape[0], 2), dtype=nboxes.dtype)

        wh[:,0] = nboxes[:, 2] - nboxes[:, 0]
        wh[:,1] = nboxes[:, 3] - nboxes[:, 1]
        min_dims = wh.min(axis=1)
        
        nboxes = nboxes[min_dims >= self.min_dim]

        return timg,nboxes