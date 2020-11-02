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

        positive_rf_centers = []
        negative_rf_centers = []


        for head_index in range(len(logits)):
            timg = img.copy()

            print("processing head: ",head_index)

            cls_logits,reg_logits = logits[head_index]
            _,_,fh,fw = cls_logits.shape

            head = model.heads[head_index]

            rfs = head.gen_rf_centers(fh,fw,gt_boxes.device)
            # torch.Tensor: rf centers as fh x fw x 2 (x, y order)

            cls_mask,reg_mask = head.matcher(rfs,gt_boxes)

            timg = cv2.cvtColor(timg,cv2.COLOR_RGB2BGR)

            for x1,y1,x2,y2 in boxes.astype(np.int32):
                timg = cv2.rectangle(timg,(x1,y1),(x2,y2),(255,0,0))

            for x,y in rfs[cls_mask >= 0]:
                timg = cv2.circle(timg, (x,y), 5, (0,255,0))

            for x,y in rfs[cls_mask == -1]:
                timg = cv2.circle(timg, (x,y), 5, (0,0,255))

            cv2.imshow("",timg)
            if cv2.waitKey(0) == 27: exit(0)