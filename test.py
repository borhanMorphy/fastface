
from datasets import get_dataset
from transforms import LFFDRandomSample
from tqdm import tqdm
from detector import build_detector

if __name__ == "__main__":
    from cv2 import cv2
    import numpy as np

    detector = build_detector("lffd")

    scales = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]

    transforms = LFFDRandomSample(scales)

    ds = get_dataset("widerface", phase='val', partitions=['easy'], transforms=transforms)

    for img,boxes in tqdm(ds):
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for x1,y1,x2,y2 in boxes.astype(np.int32):
            img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255))

        cv2.imshow("",img)
        if cv2.waitKey(0) == 27: break