import os
import argparse
import torch
import pytorch_lightning as pl

import fastface as ff

# TODO improve logging

def run_training(arch: str, config: str, batch_size: int,
    seed: int = 41, ckpt_save_path: str = "checkpoints",
    num_workers: int = 8, root_dir: str = "./"
):
    # set seed
    pl.seed_everything(seed)

    model_save_name = "{}_{}_{}_best".format(arch, config, "widerface")

    # define checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_save_path,
        verbose=True,
        filename=model_save_name,
        monitor="widerface_ap/validation/hard",
        save_top_k=1,
        mode="max" # only pick max of the metric
    )

    # resume with checkpoint if exists
    ckpt_resume_path = os.path.join(ckpt_save_path, model_save_name+".ckpt")
    if not os.path.isfile(ckpt_resume_path):
        ckpt_resume_path = None

    # build pl.LightningModule with random weights
    model = ff.FaceDetector.build(arch, config)

    # check if model support landmark regresssion
    is_keypoints_enabled = getattr(model.arch.config, "num_landmarks", 0) > 0

    # add average precision pl.metrics.Metric to the model
    model.add_metric("widerface_ap",
        ff.benchmark.WiderFaceAP()
    )

    accumulate_grad_batches = max(1, model.arch.config.batch_size // batch_size)

    # build torch.utils.data.DataLoader for training
    train_dl = ff.dataset.WiderFaceDataset(
        phase="train",
        transforms=model.arch.train_transforms,
        drop_keys= [] if is_keypoints_enabled else ["keypoints", "keypoint_ids"],
    ).get_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dls = list()

    val_dls.append(
        # build torch.utils.data.DataLoader for validation
        ff.dataset.WiderFaceDataset(
            phase="val",
            transforms=model.arch.transforms,
        ).get_dataloader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    )

    # define pl.Trainer
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        accumulate_grad_batches=accumulate_grad_batches, # backward every n batches
        callbacks=[checkpoint_callback],
        gpus=1 if torch.cuda.is_available() else 0,
        precision=32,
        resume_from_checkpoint=ckpt_resume_path,
        max_epochs=model.arch.config.max_epochs,
        check_val_every_n_epoch=1, # run validation every n epochs
    )

    # start training
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dls)

    # export model
    # TODO

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", "-a", type=str, choices=ff.list_archs(), required=True)
    ap.add_argument("--config", "-c", type=str, required=True)
    ap.add_argument("--batch-size", "-b", type=int, required=True)

    args = ap.parse_args()

    run_training(args.arch, args.config, args.batch_size)
