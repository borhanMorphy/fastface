import pytorch_lightning as pl
import mypackage

def main():
    arch = "lffd"
    config = "560_25L_8S"

    hparams = {
        "learning_rate": 1e-1,
        "momentum": .9,
        "weight_decay": 1e-5,
        "milestones": [600000, 1000000, 1200000, 1400000],
        "gamma": 0.1,
        "ratio": 10,

        "det_threshold": 0.11,
        "iou_threshold": .4,
        "keep_n": 10000
    }

    model = mypackage.module.build(arch, config, hparams=hparams, num_classes=1, in_channels=3)

    dm = mypackage.datamodule.WiderFaceDataModule(partitions=['easy'],
        train_kwargs={'batch_size':4, 'num_workers':4},
        val_kwargs={'batch_size':8, 'num_workers':4})

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./checkpoints",
        verbose=True,
        filename=f'{arch}_{config}'+'-widerface-{epoch:02d}-{val_loss:.3f}-{val_ap:.2f}',
        monitor='val_loss',
        save_top_k=3,
        mode='min'
    )

    trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=4,
        resume_from_checkpoint=None,
        checkpoint_callback=checkpoint_callback,
        max_epochs=100,
        check_val_every_n_epoch=2,
        precision=32,
        gradient_clip_val=100)

    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
