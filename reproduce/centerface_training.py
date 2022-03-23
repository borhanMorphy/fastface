import os

import numpy as np
from cv2 import cv2
import torch
import pytorch_lightning as pl

import fastface as ff

def run_sanity_check(arch: str, config: str):
    seed = 41
    accumulate_grad_batches = 4
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

    debug_counter = 0

    while 1:
        for optimizer in optimizers:
            optimizer.zero_grad()

        logits = arch.forward(batch / 255)
        target_logits = arch.build_targets(batch, targets)

        loss = arch.compute_loss(logits, target_logits)
        loss["loss"].backward()

        for optimizer in optimizers:
            optimizer.step()

        print(loss)
        if loss["loss"] > 10:
            continue
        else:
            debug_counter += 1

        if debug_counter % 10 != 0:
            continue

        preds = arch.compute_preds(logits)
        viz_preds = list()
        for pred, target in zip(preds, targets):
            p = pred[pred[..., 4].argsort(descending=True)]
            viz_preds.append(p[:10])

        heatmap = torch.sigmoid(logits[..., 0])
        target_heatmap = target_logits[..., 0]
        for i in range(heatmap.shape[0]):
            h = (heatmap[i].detach() * 255).cpu().numpy().astype(np.uint8).copy()
            t_h = (target_heatmap[i] * 255).cpu().numpy().astype(np.uint8).copy()
            img = batch[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
            p = viz_preds[i]
            for x1, y1, x2, y2, _, *l in p.detach().cpu().long().numpy():
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
                for x1, y1 in np.array(l).reshape(5, 2):
                    img = cv2.circle(img, (x1, y1), 5, (0, 0, 255))
            cv2.imshow("img", img[..., [2, 1, 0]])
            cv2.imshow("logits", h)
            cv2.imshow("target", t_h)
            if cv2.waitKey(0) == 27:
                exit(0)


def run_training(arch: str, config: str):
    seed = 41
    accumulate_grad_batches = 4

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
    train_dl = ff.dataset.WiderFaceDataset(
        phase="train",
        transforms=model.arch.train_transforms,
        drop_keys=["keypoints", "keypoint_ids"],
    ).get_dataloader(
        batch_size=int(model.arch.config.batch_size / accumulate_grad_batches),
        shuffle=True,
        num_workers=8,
    )

    val_dls = list()

    val_dls.append(
        # build torch.utils.data.DataLoader for validation
        ff.dataset.WiderFaceDataset(
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
        check_val_every_n_epoch=1, # run validation every n epochs
        logger=False,
    )

    # start training
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dls)

if __name__ == '__main__':
    arch = "centerface"
    config = "original"

    run_training(arch, config)
