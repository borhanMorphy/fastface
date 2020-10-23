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

        x1,y1,x2,y2 = boxes[selected_face_idx]
        h,w = img.shape[:2]

        aimg = img.copy()
        cx,cy = w//2, h//2
        tcx,tcy = (x1+x2) // 2, (y1+y2) // 2
        delta_x = (tcx - cx) * 2
        delta_y = (tcy - cy) * 2

        if delta_x < 0:
            # pad left
            aimg = pad_image(aimg, int(abs(delta_x)), axis=1, first=True)
        else:
            # pad right
            aimg = pad_image(aimg, int(abs(delta_x)), axis=1, first=False)
        if delta_y < 0:
            # pad up
            aimg = pad_image(aimg, int(abs(delta_y)), axis=0, first=True)
        else:
            # pad down
            aimg = pad_image(aimg, int(abs(delta_y)), axis=0, first=False)

        new_h,new_w = aimg.shape[:2]

        boxes[:,[0,2]] = boxes[:,[0,2]] + max(-1*delta_x,0)
        boxes[:,[1,3]] = boxes[:,[1,3]] + max(-1*delta_y,0)

        boxes[:, [0,2]] = boxes[:, [0,2]].clip(0,new_w)
        boxes[:, [1,3]] = boxes[:, [1,3]].clip(0,new_h)

        x1,y1,x2,y2 = boxes[selected_face_idx].astype(np.int32)

        target_w,target_h = self.target_size

        clip_left = (new_w - target_w) // 2 + (new_w - target_w) % 2
        clip_right = clip_left - (new_w - target_w) % 2

        clip_up = (new_h - target_h) // 2 + (new_h - target_h) % 2
        clip_down = clip_up - (new_h - target_h) % 2

        aimg  = aimg[clip_up: new_h-clip_down, clip_left: new_w-clip_right]

        boxes[:, [0,2]] = boxes[:, [0,2]] - clip_left
        boxes[:, [1,3]] = boxes[:, [1,3]] - clip_up

        boxes[:, [0,2]] = boxes[:, [0,2]].clip(0,target_w)
        boxes[:, [1,3]] = boxes[:, [1,3]].clip(0,target_h)

        # TODO this might be not needed
        mask = np.bitwise_and((boxes[:, 2] - boxes[:, 0] >= self.min_dim), (boxes[:, 3] - boxes[:, 1] >= self.min_dim))
        boxes = boxes[mask]

        return aimg,boxes


def pad_image(img:np.ndarray, pad_size:int, axis:int, pad_value:int=0, first:bool=True):
    h,w,c = img.shape
    if axis == 0:
        # pad up or down
        val = np.ones((pad_size,w,c), dtype=np.uint8)
    else:
        # pad left or right
        val = np.ones((h,pad_size,c), dtype=np.uint8)
    
    val *= pad_value

    if first:
        nimg = np.concatenate([val,img], axis=axis)
    else:
        nimg = np.concatenate([img,val], axis=axis)

    return nimg



