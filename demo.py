from detector import LightFaceDetector
from transforms import (
    Compose,
    Interpolate,
    Padding,
    Normalize,
    ToTensor
)

import torch
import argparse
from cv2 import cv2
import numpy as np

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', type=str, required=True)
    ap.add_argument('--model-path', '-mp', type=str, required=True)
    return ap.parse_args()

def load_image(img_path:str):
    return cv2.imread(img_path)

def re_transform(batch:torch.Tensor):
    batch = (batch * 127.5) + 127.5
    img = batch.permute(1,2,0).cpu().numpy()
    img = img.astype(np.uint8)
    return img

def main(img_path:str, model_path:str, target_size:int=640):
    img = load_image(img_path)
    model = LightFaceDetector.from_pretrained("lffd", model_path)

    transforms = Compose(
        #Interpolate(max_dim=target_size),
        #Padding(target_size=(target_size,target_size), pad_value=0),
        Normalize(mean=127.5, std=127.5),
        ToTensor()
    )

    model.eval()

    batch = transforms(img)

    preds = model.predict(batch)[0]
    print(preds)
    img = re_transform(batch)

    for pred in preds:
        pred = pred.cpu().numpy()
        x1,y1,x2,y2 = pred[:4].astype(np.int32)
        print(x1,y1,x2,y2)
        score = pred[4]
        print(score)
        img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow("",img)
    if cv2.waitKey(0) == 27:
        pass


if __name__ == '__main__':
    args = get_arguments()
    main(args.input, args.model_path)