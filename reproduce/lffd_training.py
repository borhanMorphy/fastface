import os

import torch
import pytorch_lightning as pl

import fastface as ff

def run_sanity_check():
    seed = 41
    arch = "lffd"
    config = "original"
    accumulate_grad_batches = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set seed
    pl.seed_everything(seed)

    arch = ff.build_arch(arch, config)
    arch.to(device)

    # build torch.utils.data.DataLoader for training
    train_dl = ff.dataset.WiderFaceDataset(
        phase="train",
        transforms=arch.transforms,
        drop_keys=["keypoints", "keypoint_ids"],
    ).get_dataloader(
        batch_size=int(arch.config.batch_size / accumulate_grad_batches),
        shuffle=False,
        num_workers=8,
    )

    batch, targets = next(iter(train_dl))
    batch = batch.to(device)

    optimizers, _ = arch.configure_optimizers()

    while 1:
        for optimizer in optimizers:
            optimizer.zero_grad()

        logits = arch.forward(
            (batch - 127.5) / 127.5
        )
        target_logits = arch.build_targets(batch, targets)
        loss = arch.compute_loss(logits, target_logits)
        loss["loss"].backward()

        for optimizer in optimizers:
            optimizer.step()

        print(loss)


def run_training():
    seed = 41
    arch = "lffd"
    config = "original"
    accumulate_grad_batches = 8

    model_save_name = "{}_{}_{}_best".format(arch, config, "widerface")
    ckpt_save_path = "./checkpoints"

    # define checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_save_path,
        verbose=True,
        filename=model_save_name,
        monitor="widerface_ap/validation/easy",
        save_top_k=1,
        mode="max" # only pick max of the metric
    )

    # resume with checkpoint if exists
    ckpt_resume_path = os.path.join(ckpt_save_path, model_save_name+".ckpt")
    if not os.path.isfile(ckpt_resume_path):
        ckpt_resume_path = None

    # set seed
    pl.seed_everything(seed)

    # build pl.LightningModule with random weights
    model = ff.FaceDetector.build(arch, config)

    # add average precision pl.metrics.Metric to the model
    model.add_metric("widerface_ap",
        ff.benchmark.WiderFaceAP()
    )

    # build torch.utils.data.DataLoader for training
    train_dl = ff.dataset.FDDBDataset(
        phase="train",
        # TODO WARNING training transform failing model
        transforms=model.arch.transforms,
        drop_keys=["keypoints", "keypoint_ids"],
    ).get_dataloader(
        batch_size=int(model.arch.config.batch_size / accumulate_grad_batches),
        shuffle=True,
        num_workers=8,
    )

    val_dls = list()

    val_dls.append(
        # build torch.utils.data.DataLoader for validation
        ff.dataset.FDDBDataset(
            phase="val",
            transforms=model.arch.transforms,
            drop_keys=["keypoints", "keypoint_ids"],
        ).get_dataloader(
            batch_size=int(model.arch.config.batch_size / accumulate_grad_batches),
            shuffle=False,
            num_workers=8
        )
    )

    # define pl.Trainer
    trainer = pl.Trainer(
        default_root_dir=".",
        accumulate_grad_batches=accumulate_grad_batches, # backward every n batches
        callbacks=[checkpoint_callback],
        gpus=1 if torch.cuda.is_available() else 0,
        precision=32,
        resume_from_checkpoint=ckpt_resume_path,
        max_epochs=model.arch.config.max_epochs,
        check_val_every_n_epoch=2, # run validation every n epochs
        gradient_clip_val=10, # TODO calc value
    )

    # start training
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dls)

if __name__ == '__main__':
    run_training()
    #run_sanity_check()
