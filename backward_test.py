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

    model = get_detector_by_name("lffd")

    seed_everything(42)

    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    scales = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]

    transforms = LFFDRandomSample(scales)

    ds = get_dataset("widerface", phase='val', partitions=['easy'], transforms=transforms)

    batch_size = 8
    batch_counter = 0

    img_batch = []
    gt_boxes_batch = []


    for img,boxes in tqdm(ds):

        batch = (torch.from_numpy(img).float() / 255).permute(2,0,1).unsqueeze(0).cuda()
        gt_boxes = torch.from_numpy(boxes).cuda()

        if batch_counter < batch_size:
            img_batch.append(batch)
            gt_boxes_batch.append(gt_boxes)
            batch_counter += 1
            continue
        else:
            batch_counter = 0
            batch = torch.cat(img_batch, dim=0)
            img_batch = []
            gt_boxes = gt_boxes_batch
            gt_boxes_batch = []
        optimizer.zero_grad()
        loss = model.training_step((batch,gt_boxes),0)
        loss.backward()
        optimizer.step()
        print(loss)

