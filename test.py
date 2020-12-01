from detector import LightFaceDetector
from datasets import get_dataset, get_available_datasets
from utils.utils import seed_everything, get_best_checkpoint_path

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
    ap.add_argument('--batch-size', '-bs', type=int, default=32)
    ap.add_argument('--verbose', '-vb', type=int, default=8)
    ap.add_argument('--seed', '-s', type=int, default=-1)

    ap.add_argument('--target-size', '-t', type=int, default=640)
    ap.add_argument('--device','-d',type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu','cuda'])

    ap.add_argument('--ds', '-ds',type=str,
        default="widerface-easy", choices=get_available_datasets())

    ap.add_argument('--checkpoint-path', '-ckpt', type=str, default="./checkpoints/", help='checkpoint dir path')
    ap.add_argument('--debug', action='store_true')

    return ap.parse_args()

def generate_dl(dataset_name:str, phase:str, batch_size:int, transforms=None, **kwargs):
    ds = get_dataset(dataset_name, phase=phase, transforms=transforms, **kwargs)

    def collate_fn(data):
        imgs,gt_boxes = zip(*data)
        batch = torch.stack(imgs, dim=0)
        return batch,gt_boxes

    num_workers = max(int(batch_size / 4),1)

    return DataLoader(ds, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers, collate_fn=collate_fn)

if __name__ == '__main__':
    args = parse_arguments()

    if args.seed != -1: seed_everything(args.seed)

    transforms = Compose(
        Interpolate(max_dim=args.target_size),
        Padding(target_size=(args.target_size,args.target_size), pad_value=0),
        Normalize(mean=127.5, std=127.5),
        ToTensor()
    )

    print(f"Best checkpoint, using: {args.checkpoint_path}")
    best_ap_score,ckpt_path = get_best_checkpoint_path(
            args.checkpoint_path, by='val_ap', mode='max')

    detector = LightFaceDetector.from_pretrained("lffd",ckpt_path)

    trainer = pl.Trainer(gpus=1 if args.device=='cuda' else 0)

    dl = generate_dl(args.ds, "val",
        args.batch_size, transforms=transforms)

    trainer.test(detector, test_dataloaders=dl)