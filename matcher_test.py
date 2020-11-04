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

    scales = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]

    transforms = LFFDRandomSample(scales)

    ds = get_dataset("widerface", phase='val', partitions=['easy'], transforms=transforms)

    batch_size = 8
    batch_counter = 0

    img_batch = []
    gt_boxes_batch = []
    imgs = []

    for img,boxes in tqdm(ds):

        batch = (torch.from_numpy(img).float() / 255).permute(2,0,1).unsqueeze(0).cuda()
        gt_boxes = torch.from_numpy(boxes).cuda()

        if batch_counter < batch_size:
            img_batch.append(batch)
            gt_boxes_batch.append(gt_boxes)
            imgs.append(img)
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

            for batch_index in range(batch_size):
                for selected_rfs,(x1,y1,x2,y2) in debug_info[batch_index]:
                    timg = imgs[batch_index].copy()
                    timg = cv2.cvtColor(timg,cv2.COLOR_RGB2BGR)
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    timg = cv2.rectangle(timg,(x1,y1),(x2,y2),(255,0,0))

                    for x1,y1,x2,y2 in selected_rfs:
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        timg = cv2.rectangle(timg, (x1,y1), (x2,y2), (0,255,0))

                    cv2.imshow("",timg)
                    if cv2.waitKey(0) == 27: exit(0)
        imgs = []
