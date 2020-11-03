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

            cls_logits, reg_logits, cls_targets, reg_targets, debug_info = head.build_targets(logits[head_index], [gt_boxes], debug=True)


            for selected_rfs,(x1,y1,x2,y2) in debug_info:
                iimg = timg.copy()
                iimg = cv2.cvtColor(iimg,cv2.COLOR_RGB2BGR)
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                iimg = cv2.rectangle(iimg,(x1,y1),(x2,y2),(255,0,0))

                for x,y in selected_rfs:
                    x = int(x)
                    y = int(y)
                    iimg = cv2.circle(iimg, (x,y), 5, (0,255,0))

                cv2.imshow("",iimg)
                if cv2.waitKey(0) == 27: exit(0)