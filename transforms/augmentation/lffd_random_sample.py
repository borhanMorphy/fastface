import numpy as np
from typing import List,Tuple
import random
import math
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

        # TODO move this to dataset
        mask = np.bitwise_and((boxes[:, 2] - boxes[:, 0] >= self.min_dim), (boxes[:, 3] - boxes[:, 1] >= self.min_dim))
        boxes = boxes[mask]

        num_faces = boxes.shape[0]
        if num_faces == 0:
            h,w = img.shape[:2]
            target_w,target_h = self.target_size
            cx,cy = w//2,h//2
            offset_x = int(w//2 - target_w//2)
            offset_y = int(h//2 - target_h//2)
            aimg = img[offset_y:offset_y+target_h, offset_x:offset_x+target_w]
            return aimg,boxes

        # select one face
        selected_face_idx = random.randint(0, num_faces-1)

        selected_face_scale_idx = random.randint(0, self.scales.shape[0]-1)
        scale_lower,scale_higher = self.scales[selected_face_scale_idx]
        scale_size = random.randint(scale_lower,scale_higher)

        x1,y1,x2,y2 = boxes[selected_face_idx].astype(np.int32)
        h,w = img.shape[:2]

        sf = scale_size / (math.sqrt((y2-y1) * (x2-x1)) + 1e-16)

        aboxes = boxes * sf
        sx1,sy1,sx2,sy2 = aboxes[selected_face_idx].astype(np.int32)

        offset_w_1 = (self.target_size[0] - (sx2-sx1)) // 2
        offset_w_2 = offset_w_1 + (self.target_size[0] - (sx2-sx1)) % 2

        offset_w_1 //= sf
        offset_w_2 //= sf

        offset_h_1 = (self.target_size[1] - (sy2-sy1)) // 2
        offset_h_2 = offset_h_1 + (self.target_size[1] - (sy2-sy1)) % 2

        offset_h_1 //= sf
        offset_h_2 //= sf

        offset_w_1 = int(min(x1,offset_w_1))
        offset_w_2 = int(min(w-x2,offset_w_2))

        offset_h_1 = int(min(y1,offset_h_1))
        offset_h_2 = int(min(h-y2,offset_h_2))

        # select faces that center's lie between cropped region
        low_h,high_h = y1-offset_h_1,y2+offset_h_2
        low_w,high_w = x1-offset_w_1,x2+offset_w_2
        cboxes_x = (boxes[:, 0] + boxes[:, 2]) // 2
        cboxes_y = (boxes[:, 1] + boxes[:, 3]) // 2

        center_mask = np.bitwise_and(
            np.bitwise_and(cboxes_x > low_w, cboxes_x < high_w),
            np.bitwise_and(cboxes_y > low_h, cboxes_y < high_h))

        aimg = img[y1-offset_h_1:y2+offset_h_2, x1-offset_w_1:x2+offset_w_2]
        aimg = cv2.resize(aimg,None,fx=sf,fy=sf)
        aimg = aimg[:self.target_size[1], : self.target_size[0]]

        boxes[:, [0,2]] = boxes[:, [0,2]] - (x1 - offset_w_1)
        boxes[:, [1,3]] = boxes[:, [1,3]] - (y1 - offset_h_1)
        boxes *= sf

        x1,y1,x2,y2 = boxes[selected_face_idx].astype(np.int32)
        cx = (x1+x2) // 2
        cy = (y1+y2) // 2

        img = np.zeros((self.target_size[1],self.target_size[0],3), dtype=np.uint8)
        tcx = img.shape[1] // 2
        tcy = img.shape[0] // 2

        offset_x = int(tcx - cx)
        offset_y = int(tcy - cy)

        if offset_x >= 0:
            # pad left
            left_index_x = offset_x
            right_index_x = offset_x+aimg.shape[1]
        else:
            # pad_right
            left_index_x = 0
            right_index_x = aimg.shape[1]
        if offset_y >= 0:
            # pad up
            up_index_y = offset_y
            down_index_y = offset_y+aimg.shape[0]
        else:
            # pad down
            up_index_y = 0
            down_index_y = aimg.shape[0]

        target_h,target_w = img[up_index_y:down_index_y, left_index_x:right_index_x].shape[:2]
        source_h,source_w = aimg.shape[:2]

        up_index_y = up_index_y + target_h - source_h
        down_index_y = down_index_y + target_h - source_h
        left_index_x = left_index_x + target_w - source_w
        right_index_x = right_index_x + target_w - source_w

        img[up_index_y:down_index_y, left_index_x:right_index_x] = aimg

        boxes[:, [0,2]] += left_index_x
        boxes[:, [1,3]] += up_index_y

        mask = np.bitwise_and((boxes[:, 2] - boxes[:, 0] >= self.min_dim), (boxes[:, 3] - boxes[:, 1] >= self.min_dim))
        boxes = boxes[np.bitwise_and(mask,center_mask)]
        boxes[:, 0] = boxes[:, 0].clip(0,self.target_size[0])
        boxes[:, 1] = boxes[:, 1].clip(0,self.target_size[1])
        boxes[:, 2] = boxes[:, 2].clip(0,self.target_size[0])
        boxes[:, 3] = boxes[:, 3].clip(0,self.target_size[1])

        return img,boxes
