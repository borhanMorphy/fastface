from detector import LightFaceDetector
from datasets import get_dataset, get_available_datasets
from utils.utils import seed_everything, get_best_checkpoint_path
import metrics
import archs

from transforms import (
    Compose,
    Interpolate,
    Padding,
    Normalize,
    ToTensor
)

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import argparse

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device','-d',type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu','cuda'])

    ap.add_argument('--ds', '-ds', type=str,
        default="widerface-easy", choices=get_available_datasets())

    ap.add_argument('--arch', '-a', type=str, choices=archs.get_available_archs(),
        default='lffd', help='architecture to perform face detection')

    ap.add_argument('--config', '-c', type=str,
        default='560_25L_8S', help='architecture configuration')

    ap.add_argument('--checkpoint-path', '-ckpt', type=str, default="./checkpoints/", help='checkpoint dir path')
    ap.add_argument('--model-path', '-mp', type=str, help='model path')

    ap.add_argument('--metric', '-m', type=str, choices=metrics.get_available_metrics(), default='widerface_ap')

    return ap.parse_args()

def generate_dl(dataset_name:str, phase:str, batch_size:int, transforms=None, **kwargs):
    ds = get_dataset(dataset_name, phase=phase, transforms=transforms, **kwargs)

    def collate_fn(data):
        imgs,gt_boxes = zip(*data)
        batch = torch.stack(imgs, dim=0)
        return batch,gt_boxes

    num_workers = max(int(batch_size / 4),2)

    return DataLoader(ds, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers, collate_fn=collate_fn)

if __name__ == '__main__':
    args = parse_arguments()

    transforms = Compose(
        Normalize(mean=127.5, std=127.5),
        ToTensor()
    )

    metrics = {
        args.metric : metrics.get_metric(args.metric)
    }

    print(f"testing on {args.ds} dataset")

    if args.model_path:
        print("loading using model path")
        model_path = args.model_path
    else:
        print(f"Best checkpoint, using: {args.checkpoint_path}")
        best_ap_score,model_path = get_best_checkpoint_path(
                args.checkpoint_path, by='ap', mode='max')

    detector = LightFaceDetector.from_pretrained(args.arch, model_path, config=args.config, metrics=metrics)

    trainer = pl.Trainer(gpus=1 if args.device=='cuda' else 0)

    dl = generate_dl(args.ds, "val", 1, transforms=transforms)

    trainer.test(detector, test_dataloaders=dl)