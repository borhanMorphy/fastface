from models import get_detector_by_name
from datasets import get_dataset
from transforms import Interpolate,Padding
from utils.utils import seed_everything
from typing import Tuple,List

from tqdm import tqdm
from cv2 import cv2
import numpy as np
import torch

def prep_batch(img:np.ndarray, gt_boxes:np.ndarray) -> Tuple[torch.Tensor,torch.Tensor]:
    batch = torch.from_numpy(img.astype(np.float32) / 255)
    batch = batch.permute(2,0,1).unsqueeze(0).contiguous()

    return batch,torch.from_numpy(gt_boxes)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ds = get_dataset("widerface", phase='val', partitions=['easy'], transforms=transforms)
    batch_size = 4
    batch_counter = 0
    img_batch = []
    gt_boxes_batch = []
    optimizer.zero_grad()

    val_holder = []
    verbose = 4
    accumulation = 4
    accumulation_counter = 0

    for i,(img,boxes) in enumerate(tqdm(ds)):

        batch,gt_boxes = prep_batch(img,boxes)

        if batch_counter < batch_size:
            img_batch.append(batch)
            gt_boxes_batch.append(gt_boxes)
            batch_counter += 1
            continue
        else:
            batch_counter = 0
            batch = torch.cat(img_batch, dim=0).contiguous()
            img_batch = []
            gt_boxes = gt_boxes_batch
            gt_boxes_batch = []

        loss = model.training_step((batch,gt_boxes),i)


        loss /= accumulation
        loss.backward()

        val_holder.append(loss.item())

        if len(val_holder) == verbose:
            print(f"epoch [{1}/{1}] loss: {sum(val_holder)/verbose}")
            val_holder = []

        accumulation_counter += 1

        if accumulation_counter == accumulation:
            optimizer.step()
            optimizer.zero_grad()
            accumulation_counter = 0
        else:
            accumulation_counter += 1