import fastface as ff
import pytorch_lightning as pl
import torch

# checkout available pretrained models
print(ff.list_pretrained_models())
# ["lffd_slim", "lffd_original"]

# build pl.LightningModule using pretrained weights
model = ff.FaceDetector.from_checkpoint("checkpoints/yolov4_tiny_widerface_best-v0.ckpt")

# set model to eval mode
model.eval()

# build transforms
transforms = ff.transforms.Compose(
    ff.transforms.ConditionalInterpolate(max_size=640),
)

# build torch.utils.data.Dataset
ds = ff.dataset.WiderFaceDataset(phase="test", partitions=["easy"], transforms=transforms)

# build torch.utils.data.DataLoader
dl = ds.get_dataloader(batch_size=1, num_workers=0)

# add average precision pl.metrics.Metric to the model
model.add_metric("AP@0.5",
    ff.metric.AveragePrecision(iou_threshold=0.5))

model.add_metric("AR@0.5",
    ff.metric.AverageRecall(iou_threshold_max=0.5, iou_threshold_min=0.5))

model.add_metric("APsmall@0.5",
    ff.metric.AveragePrecision(iou_threshold=0.5, area="small"))

model.add_metric("APmedium@0.5",
    ff.metric.AveragePrecision(iou_threshold=0.5, area="medium"))

model.add_metric("APlarge@0.5",
    ff.metric.AveragePrecision(iou_threshold=0.5, area="large"))

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
{'AP@0.5': 0.9410496354103088,
 'APlarge@0.5': 0.9857767224311829,
 'APmedium@0.5': 0.8848604559898376,
 'APsmall@0.5': 0.7582277655601501,
 'AR@0.5': 0.9590609073638916}
"""
