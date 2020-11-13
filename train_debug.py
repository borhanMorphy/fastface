from models import get_detector_by_name
from datasets import get_dataset
from transforms import Interpolate,Padding,LFFDRandomSample
from utils.utils import seed_everything
from typing import Tuple,List

from tqdm import tqdm
from cv2 import cv2
import numpy as np
import torch

def prep_batch(img:np.ndarray, gt_boxes:np.ndarray) -> Tuple[torch.Tensor,torch.Tensor]:
    batch = (torch.from_numpy(img.astype(np.float32)) - 127.5) / 127.5
    batch = batch.permute(2,0,1).unsqueeze(0).contiguous()

    return batch,torch.from_numpy(gt_boxes)

class Transforms():
    def __init__(self, *ts):
        self.ts = ts

    def __call__(self, img:np.ndarray, gt_boxes:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        for t in self.ts:
            img,gt_boxes = t(img,gt_boxes)
        return img,gt_boxes

def collate_fn(data):
    imgs = []
    gt_boxes = []
    for img,gt_box in data:
        img,gt_box = prep_batch(img,gt_box)
        imgs.append(img)
        gt_boxes.append(gt_box)

    batch = torch.cat(imgs, dim=0)
    return batch,gt_boxes

if __name__ == "__main__":
    target_size = (640,640)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_detector_by_name("lffd").train()
    model.to(device)

    seed_everything(42)

    val_transforms = Transforms(
        Interpolate(max_dim=target_size[0]),
        Padding(target_size=target_size, pad_value=0)
    )

    train_transforms = Transforms(
        LFFDRandomSample([(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)], target_size=target_size, min_dim=10)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

    ds = get_dataset("widerface", phase='train', partitions=['easy','medium','hard'], transforms=val_transforms)
    #val_ds = get_dataset("widerface", phase='val', partitions=['easy'], transforms=val_transforms)

    batch_size = 16

    val_holder = []
    verbose = 16
    epochs = 50
    accumulation = 2
    accumulation_counter = 0

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=4, collate_fn=collate_fn)

    #val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
    #    num_workers=2, collate_fn=collate_fn)
    optimizer.zero_grad()
    for epoch in range(epochs):
        for i,(batch,gt_boxes) in enumerate(tqdm(dl)):

            loss = model.training_step((batch.to(device),[box.to(device) for box in gt_boxes]),i)
            (loss / accumulation).backward()

            accumulation_counter += 1

            val_holder.append(loss.item())

            if accumulation_counter == accumulation:
                optimizer.step()
                optimizer.zero_grad()

            if len(val_holder) == verbose:
                print(f"epoch [{epoch+1}/{epochs}] loss: {sum(val_holder)/verbose}")
                val_holder = []

