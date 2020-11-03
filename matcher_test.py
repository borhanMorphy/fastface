from models import get_detector_by_name
from datasets import get_dataset
from transforms import LFFDRandomSample
from utils.utils import seed_everything

from tqdm import tqdm
from cv2 import cv2
import numpy as np
import torch

if __name__ == "__main__":
    model = get_detector_by_name("lffd")

    seed_everything(42)

    model.cuda()

    scales = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]

    transforms = LFFDRandomSample(scales)

    ds = get_dataset("widerface", phase='val', partitions=['easy'], transforms=transforms)

    for img,boxes in tqdm(ds):

        batch = (torch.from_numpy(img).float() / 255).permute(2,0,1).unsqueeze(0).cuda()
        gt_boxes = torch.from_numpy(boxes).cuda()

        logits = model(batch)


        for head_index in range(len(logits)):
            timg = img.copy()

            print("processing head: ",head_index)

            head = model.heads[head_index]

            cls_logits, reg_logits, cls_targets, reg_targets = head.build_targets(logits[head_index], [gt_boxes])
            print("cls_logits: ",cls_logits.shape)
            print("reg_logits: ",reg_logits.shape)
            print("cls_targets: ",cls_targets.shape)
            print("reg_targets: ",reg_targets.shape)
            """
                cls_logits  : N x 1
                reg_logits  : N x 4
                cls_targets : N x 1
                reg_targets : N x 4

            timg = cv2.cvtColor(timg,cv2.COLOR_RGB2BGR)

            for x1,y1,x2,y2 in boxes.astype(np.int32):
                timg = cv2.rectangle(timg,(x1,y1),(x2,y2),(255,0,0))

            for x,y in rfs[cls_mask >= 0]:
                timg = cv2.circle(timg, (x,y), 5, (0,255,0))

            for x,y in rfs[cls_mask == -1]:
                timg = cv2.circle(timg, (x,y), 5, (0,0,255))

            cv2.imshow("",timg)
            """
            if cv2.waitKey(0) == 27: exit(0)
        exit(0)