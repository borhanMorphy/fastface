import pytorch_lightning as pl
import torch

import fastface as ff

# model device, select gpu if exists
device = "cuda" if torch.cuda.is_available() else "cpu"

# widerface dataset partition
partition = "easy"  # also `medium` or `hard` can be selectable

# select and build pretrained model to test on widerface
# for this tutorial `lffd_original` is selected
# for selectable models checkout ff.list_pretrained_models()
model = ff.FaceDetector.from_pretrained("lffd_original")
# model: pl.LightningModule

# get widerface average precision metric, defined in the competition
metric = ff.metric.WiderFaceAP(iou_threshold=0.5)
# metric: pl.metrics.Metric

# add metric to the model
model.add_metric("widerface_ap", metric)

# get widerface dataset
ds = ff.dataset.WiderFaceDataset(
    phase="test",
    partitions=[partition],
    transforms=ff.transforms.Compose(
        ff.transforms.ConditionalInterpolate(max_size=1500)
    ),
)
# ds: torch.utils.data.Dataset

# get dataloader
dl = ds.get_dataloader(batch_size=1, num_workers=1)
# dl: torch.utils.data.DataLoader

# define trainer
trainer = pl.Trainer(
    logger=False,
    checkpoint_callback=False,
    gpus=1 if device == "cuda" else 0,
    precision=32,
)

# run test
trainer.test(model, test_dataloaders=dl)
