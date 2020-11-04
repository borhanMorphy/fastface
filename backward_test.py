from models import get_detector_by_name
from datasets import get_dataset
from transforms import LFFDRandomSample
from utils.utils import seed_everything

from tqdm import tqdm
from cv2 import cv2
import numpy as np
import torch

if __name__ == "__main__":
    # TODO concat head results

    model = get_detector_by_name("lffd").train()

    seed_everything(42)

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scales = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]

    transforms = LFFDRandomSample(scales)

    ds = get_dataset("widerface", phase='val', partitions=['easy'], transforms=transforms)

    batch_size = 8
    batch_counter = 0
    verbose = 20
    val_holder = []

    img_batch = []
    gt_boxes_batch = []

    epochs = 50
    accumulation = 4
    accumulation_counter = 0
    optimizer.zero_grad()

    loop_over = False
    cached_batch = None
    cached_gt_boxes = None

    for i in range(epochs):
        for img,boxes in tqdm(ds):

            batch = (torch.from_numpy(img).float() / 255).permute(2,0,1).unsqueeze(0).cuda().contiguous()
            gt_boxes = torch.from_numpy(boxes).cuda()

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

            if isinstance(cached_batch,type(None)):
                cached_batch = batch.clone()
                cached_gt_boxes = gt_boxes
            else:
                pass#batch = cached_batch
                pass#gt_boxes = cached_gt_boxes

            loss = model.training_step((batch,gt_boxes),0)

            loss /= accumulation
            loss.backward()

            val_holder.append(loss.item())

            if len(val_holder) == verbose:
                print(f"epoch [{i+1}/{epochs}] loss: {sum(val_holder)/verbose}")
                val_holder = []

            accumulation_counter += 1

            if accumulation_counter == accumulation:
                optimizer.step()
                optimizer.zero_grad()
                accumulation_counter = 0
            else:
                accumulation_counter += 1