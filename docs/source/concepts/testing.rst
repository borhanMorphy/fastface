Testing
=======

.. code-block:: python

    import fastface as ff
    import pytorch_lightning as pl
    import torch

    # checkout available pretrained models
    print(ff.list_pretrained_models())
    # ["lffd_slim", "lffd_original"]

    # build pl.LightningModule using pretrained weights
    model = ff.FaceDetector.from_pretrained("lffd_slim")

    # set model to eval mode
    model.eval()

    # build transforms
    transforms = ff.transforms.Compose(
        ff.transforms.Interpolate(max_dim=480),
        ff.transforms.Padding(target_size=(480, 480))
    )

    # build torch.utils.data.Dataset
    ds = ff.dataset.FDDBDataset(phase="test", transforms=transforms)

    # build torch.utils.data.DataLoader
    dl = ds.get_dataloader(batch_size=1, num_workers=0)

    # add average precision pl.metrics.Metric to the model
    model.add_metric("average_precision",
        ff.metric.AveragePrecision(iou_threshold=0.5))

    # define pl.Trainer for testing
    trainer = pl.Trainer(
        benchmark=True,
        logger=False,
        checkpoint_callback=False,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=32)

    # run test
    trainer.test(model, test_dataloaders=[dl])
    """
    DATALOADER:0 TEST RESULTS
    {'average_precision': 0.9138654470443726}
    """