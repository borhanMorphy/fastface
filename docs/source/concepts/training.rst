Training
========

.. code-block:: python

    import os
    import fastface as ff
    import pytorch_lightning as pl
    import torch

    # set seed
    pl.seed_everything(41)

    # build training transforms
    train_transforms = ff.transforms.Compose(
        ff.transforms.Interpolate(max_dim=480),
        ff.transforms.Padding(target_size=(480, 480)),
        ff.transforms.RandomHorizontalFlip(p=0.5)
    )

    # build val transforms
    val_transforms = ff.transforms.Compose(
        ff.transforms.Interpolate(max_dim=480),
        ff.transforms.Padding(target_size=(480, 480))
    )

    # build torch.utils.data.DataLoader for training
    train_dl = ff.dataset.FDDBDataset(phase="train", transforms=train_transforms).get_dataloader(
        batch_size=8, shuffle=True, num_workers=8
    )

    # build torch.utils.data.DataLoader for validation
    val_dl = ff.dataset.FDDBDataset(phase="val", transforms=val_transforms).get_dataloader(
        batch_size=8, shuffle=False, num_workers=8
    )

    # define preprocess dict
    preprocess = {
        "mean": 127.5,
        "std": 127.5,
        "normalized_input": False
    }

    # define hyper parameter dict
    hparams = {
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.00001,
        "milestones": [500000, 1000000, 1500000],
        "gamma": 0.1,
        "ratio": 10
    }

    # checkout available architectures to train
    print(ff.list_archs())
    # ["lffd"]
    arch = "lffd"

    # checkout available configs for the architecture
    print(ff.list_arch_configs(arch))
    # ["original", "slim"]
    config = "slim"

    # build pl.LightningModule with random weights
    model = ff.FaceDetector.build(arch, config=config,
        preprocess=preprocess, hparams=hparams)

    # add average precision pl.metrics.Metric to the model
    model.add_metric("average_precision",
        ff.metric.AveragePrecision(iou_threshold=0.5))

    model_save_name = "{}_{}_{}_best".format(arch, config, "fddb")
    ckpt_save_path = "./checkpoints"

    # resume with checkpoint
    ckpt_resume_path = os.path.join(ckpt_save_path, model_save_name+".ckpt")
    if not os.path.isfile(ckpt_resume_path):
        ckpt_resume_path = None

    # define checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_save_path,
        verbose=True,
        filename=model_save_name,
        monitor="average_precision",
        save_top_k=1,
        mode="max" # only pick max of `average_precision`
    )

    # define pl.Trainer
    trainer = pl.Trainer(
        default_root_dir=".",
        accumulate_grad_batches=4, # backward every 4 batches
        callbacks=[checkpoint_callback],
        gpus=1 if torch.cuda.is_available() else 0,
        precision=32,
        resume_from_checkpoint=ckpt_resume_path,
        max_epochs=100,
        check_val_every_n_epoch=2, # run validation every 2 epochs
        gradient_clip_val=10, # clip gradient
    )

    # start training
    trainer.fit(model, train_dataloader=train_dl,
        val_dataloaders=[val_dl])
