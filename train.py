from models.lffd import LFFD
from datasets import get_dataset
from transforms import Interpolate,Padding,FaceDiscarder
from utils.utils import seed_everything
from typing import Tuple,List

from tqdm import tqdm
from cv2 import cv2
import numpy as np
import torch
import argparse

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', '-bs', type=int, default=32)
    ap.add_argument('--epochs', '-e', type=int, default=50)
    ap.add_argument('--verbose', '-vb', type=int, default=8)
    ap.add_argument('--seed', '-s', type=int, default=42)
    ap.add_argument('--learning-rate', '-lr', type=float, default=1e-1)
    ap.add_argument('--momentum', '-m', type=float, default=.9)
    ap.add_argument('--weight-decay', '-wd', type=float, default=0)
    ap.add_argument('--debug','-d',action='store_true')

    return ap.parse_args()

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
    args = parse_arguments()

    debug = args.debug
    target_size = (640,640)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LFFD(debug=debug)
    model.to(device)

    if args.seed != -1:
        seed_everything(args.seed)

    val_transforms = Transforms(
        Interpolate(max_dim=target_size[0]),
        Padding(target_size=target_size, pad_value=0),
        FaceDiscarder(min_face_scale=10)
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    ds = get_dataset("widerface", phase='val', partitions=['easy','medium','hard'], transforms=val_transforms)

    batch_size = args.batch_size

    val_holder = []
    verbose = args.verbose
    epochs = args.epochs

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=4, collate_fn=collate_fn)

    for epoch in range(epochs):
        for i,(batch,gt_boxes) in enumerate(tqdm(dl)):

            loss = model.training_step((batch.to(device),[box.to(device) for box in gt_boxes]),i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_holder.append(loss.item())

            if len(val_holder) == verbose:
                print(f"epoch [{epoch+1}/{epochs}] loss: {sum(val_holder)/verbose}")
                val_holder = []