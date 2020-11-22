from detector import LightFaceDetector
from datasets import get_dataset, get_available_datasets
from utils.utils import seed_everything

from transforms import (
    Compose,
    Interpolate,
    Padding,
    FaceDiscarder,
    Normalize,
    ToTensor,
    LFFDRandomSample,
    RandomHorizontalFlip
)

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import argparse

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', '-bs', type=int, default=32)
    ap.add_argument('--accumulation', '-ac', type=int, default=1)
    ap.add_argument('--epochs', '-e', type=int, default=50)
    ap.add_argument('--verbose', '-vb', type=int, default=8)
    ap.add_argument('--seed', '-s', type=int, default=-1)
    ap.add_argument('--learning-rate', '-lr', type=float, default=1e-1)
    ap.add_argument('--momentum', '-m', type=float, default=.9)
    ap.add_argument('--weight-decay', '-wd', type=float, default=0)

    ap.add_argument('--target-size', '-t', type=int, default=640)
    ap.add_argument('--device','-d',type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu','cuda'])

    ap.add_argument('--train-ds', '-tds',type=str,
        default="widerface", choices=get_available_datasets())

    ap.add_argument('--val-ds', '-vds', type=str,
        default="widerface-easy", choices=get_available_datasets())

    ap.add_argument('--debug', action='store_true')

    return ap.parse_args()

def generate_dl(dataset_name:str, phase:str, batch_size:int, transforms=None, **kwargs):
    ds = get_dataset(dataset_name, phase=phase, transforms=transforms, **kwargs)

    def collate_fn(data):
        imgs,gt_boxes = zip(*data)
        batch = torch.stack(imgs, dim=0)
        return batch,gt_boxes

    num_workers = max(int(batch_size / 4),1)

    return DataLoader(ds, batch_size=batch_size, shuffle=phase=='train', pin_memory=True,
        num_workers=num_workers if phase=='train' else 1,
        collate_fn=collate_fn)

if __name__ == '__main__':
    args = parse_arguments()

    if args.seed != -1: seed_everything(args.seed)

    val_transforms = Compose(
        Interpolate(max_dim=args.target_size),
        Padding(target_size=(args.target_size,args.target_size), pad_value=0),
        Normalize(mean=127.5, std=127.5),
        ToTensor()
    )

    train_transforms = Compose(
        FaceDiscarder(min_face_scale=10),
        LFFDRandomSample(
            [
                (10,15),(15,20),(20,40),(40,70),
                (70,110),(110,250),(250,400),(400,560)
            ], target_size=(args.target_size,args.target_size)),
        RandomHorizontalFlip(p=0.5),
        Normalize(mean=127.5, std=127.5),
        ToTensor()
    )

    hyp = {
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay
    }

    trainer = pl.Trainer(gpus=1 if args.device=='cuda' else 0,
        accumulate_grad_batches=args.accumulation)

    detector = LightFaceDetector.build("lffd", metric_names=['ap'], hyp=hyp, debug=args.debug)

    train_dl = generate_dl(args.train_ds, "train",
        args.batch_size, transforms=train_transforms)

    val_dl = generate_dl(args.val_ds, "val",
        args.batch_size, transforms=val_transforms)

    #trainer.fit(detector,
    #    train_dataloader=train_dl,
    #    val_dataloaders=val_dl)

    trainer.test(detector, val_dl)