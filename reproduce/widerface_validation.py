import argparse
import torch
import pytorch_lightning as pl

import fastface as ff

# TODO improve logging

def run_validation(ckpt_path: str, batch_size: int = 1, num_workers: int = 0):

    # build pl.LightningModule with random weights
    model = ff.FaceDetector.load_from_checkpoint(ckpt_path)

    # add average precision pl.metrics.Metric to the model
    model.add_metric("widerface_ap",
        ff.benchmark.WiderFaceAP()
    )

    # build torch.utils.data.DataLoader for validation
    dls = list()
    dls.append(
        ff.dataset.WiderFaceDataset(
            phase="val",
            transforms=model.arch.transforms,
        ).get_dataloader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    )

    # define pl.Trainer
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        precision=32,
        logger=False,
    )

    # start validating
    trainer.validate(model=model, dataloaders=dls)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", "-ckpt", type=str, required=True)
    ap.add_argument("--num-workers", "-nw", type=int, default=0)
    ap.add_argument("--batch-size", "-b", type=int, default=1)

    args = ap.parse_args()

    run_validation(args.checkpoint, batch_size=args.batch_size, num_workers=args.num_workers)
