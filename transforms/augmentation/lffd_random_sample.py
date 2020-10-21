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
        target_center_x,target_center_y = (x2+x1) // 2, (y2+y1) // 2
        current_center_x,current_center_y = img.shape[1] // 2, img.shape[0] // 2

        delta_x = int(current_center_x - target_center_x)
        delta_y = int(current_center_y - target_center_y)

        boxes[:, [0,2]] = boxes[:, [0,2]] - delta_x
        boxes[:, [1,3]] = boxes[:, [1,3]] - delta_y

        new_img = np.zeros((img.shape[0]+abs(delta_y), img.shape[1]+abs(delta_x), 3), dtype=np.uint8)
        if delta_x >= 0:
            new_img[delta_y:, delta_x:] = img
        else:
            new_img[delta_y:, :delta_x] = img
        return new_img,boxes

        sf = scale_size / (boxes[selected_face_idx][2]-boxes[selected_face_idx][0])
        nimg = cv2.resize(img, None, fx=sf, fy=sf)
        nboxes = boxes * sf

        center_x = int((nboxes[selected_face_idx][2] + nboxes[selected_face_idx][0]) / 2)
        center_y = int((nboxes[selected_face_idx][3] + nboxes[selected_face_idx][1]) / 2)

        timg = np.zeros((*self.target_size,3), dtype=np.uint8)

        left_x = center_x - self.target_size[0] // 2
        right_x = center_x + self.target_size[0] // 2
        up_y = center_y - self.target_size[1] // 2
        down_y = center_y + self.target_size[1] // 2

        crop_left = max(0,left_x)
        crop_right = min(nimg.shape[1],right_x)
        crop_up = max(0,up_y)
        crop_down = min(nimg.shape[0],down_y)

        offset_left =  self.target_size[0] // 2 - ((crop_right - crop_left) // 2 + (crop_right - crop_left) % 2)
        offset_right = self.target_size[0] // 2 - ((crop_right - crop_left) // 2)
        offset_up = self.target_size[1] // 2 - ((crop_down - crop_up) // 2 + (crop_down - crop_up) % 2)
        offset_down = self.target_size[1] // 2 - ((crop_down - crop_up) // 2)
        print("shift: ",left_x,up_y)
        print("offset: ",offset_left,offset_up)
        nboxes[:,[0,2]] = nboxes[:,[0,2]] - max(left_x,0) + offset_left
        nboxes[:,[1,3]] = nboxes[:,[1,3]] - max(up_y,0) + offset_up

        
        print(timg[offset_up:self.target_size[1]-offset_down, offset_left:self.target_size[0]-offset_right].shape)
        print(nimg[crop_up:crop_down,crop_left:crop_right].shape)
        timg[offset_up:self.target_size[1]-offset_down, offset_left:self.target_size[0]-offset_right] = nimg[crop_up:crop_down,crop_left:crop_right]

        return timg,nboxes