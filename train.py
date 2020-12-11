from detector import LightFaceDetector
from datasets import get_dataset, get_available_datasets
from utils.utils import seed_everything, get_best_checkpoint_path
from metrics import get_metric

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
    ap.add_argument('--weight-decay', '-wd', type=float, default=1e-5)

    ap.add_argument('--target-size', '-t', type=int, default=640)
    ap.add_argument('--device','-d',type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu','cuda'])

    ap.add_argument('--train-ds', '-tds',type=str,
        default="widerface", choices=get_available_datasets())

    ap.add_argument('--val-ds', '-vds', type=str,
        default="widerface-easy", choices=get_available_datasets())

    ap.add_argument('--checkpoint-path', '-ckpt', type=str, default="./checkpoints/", help='checkpoint dir path')
    ap.add_argument('--resume', action='store_true')
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
        num_workers=num_workers, collate_fn=collate_fn)

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
        FaceDiscarder(min_face_scale=2),
        LFFDRandomSample(
            [
                (10,15),(15,20),(20,40),(40,70),
                (70,110),(110,250),(250,400),(400,560)
            ], target_size=(args.target_size,args.target_size)),
        FaceDiscarder(min_face_scale=8),
        RandomHorizontalFlip(p=0.5),
        Normalize(mean=127.5, std=127.5),
        ToTensor()
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_path,
        verbose=True,
        filename='lffd-widerface-{epoch:02d}-{val_loss:.3f}-{val_ap:.2f}',
        monitor='val_loss',
        save_top_k=3,
        mode='min'
    )

    hyp = {
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay
    }

    if args.resume:
        print(f"resuming from best checkpoint, using: {args.checkpoint_path}")
    else:
        print("traning from scratch")

    metrics = {
        'val_ap': get_metric("widerface_ap")
    }

    detector = LightFaceDetector.build("lffd", metrics=metrics, hyp=hyp, debug=args.debug)
    ckpt_path = None
    if args.resume:
        best_ap_score,ckpt_path = get_best_checkpoint_path(
            args.checkpoint_path, by='val_ap', mode='max')
        print(f"resuming the training, best ap score: {best_ap_score}")

    trainer = pl.Trainer(
        gpus=1 if args.device=='cuda' else 0,
        accumulate_grad_batches=args.accumulation,
        resume_from_checkpoint=ckpt_path,
        checkpoint_callback=checkpoint_callback,
        max_epochs=args.epochs)

    train_dl = generate_dl(args.train_ds, "train",
        args.batch_size, transforms=train_transforms)

    val_dl = generate_dl(args.val_ds, "val",
        args.batch_size, transforms=val_transforms)

    trainer.fit(detector,
        train_dataloader=train_dl,
        val_dataloaders=val_dl)