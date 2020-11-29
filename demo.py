from detector import LightFaceDetector
from transforms import (
    Compose,
    Interpolate,
    Padding,
    Normalize,
    ToTensor
)

import argparse
from cv2 import cv2

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', type=str, required=True)
    ap.add_argument('--model-path', '-mp', type=str, required=True)
    return ap.parse_args()

def load_image(img_path:str):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

def main(img_path:str, model_path:str, target_size:int=640):
    img = load_image(img_path)
    model = LightFaceDetector.from_pretrained("lffd",model_path)

    transforms = Compose(
        Interpolate(max_dim=target_size),
        Padding(target_size=(target_size,target_size), pad_value=0),
        Normalize(mean=127.5, std=127.5),
        ToTensor()
    )

    batch = transforms(img)

    model.eval()

    cls_logits,reg_logits = model(batch.unsqueeze(0))

    print(cls_logits[0].shape)


if __name__ == '__main__':
    args = get_arguments()
    main(args.input, args.model_path)