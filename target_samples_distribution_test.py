from models import get_detector_by_name
from datasets import get_dataset
from transforms import LFFDRandomSample
from utils.utils import seed_everything

from tqdm import tqdm
from cv2 import cv2
import numpy as np
import torch

if __name__ == "__main__":
    # TODO make sure this works correctly

    model = get_detector_by_name("lffd")

    seed_everything(42)

    model.cuda()

    scales = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]

    transforms = LFFDRandomSample(scales)

    ds = get_dataset("widerface", phase='val', partitions=['easy'], transforms=transforms)

    batch_size = 8
    batch_counter = 0

    img_batch = []
    gt_boxes_batch = []

    head_pos_counter = [0,0,0,0,0,0,0,0]

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

        logits = model(batch)

        for head_index in range(len(logits)):

            head = model.heads[head_index]

            cls_logits, reg_logits, cls_targets, reg_targets, debug_info = head.build_targets(logits[head_index], gt_boxes, debug=True)
            head_pos_counter[head_index] += reg_targets.size(0)

    for idx,pos_count in enumerate(head_pos_counter):
        print(f"head {idx+1} total positive sample signals: ",pos_count)