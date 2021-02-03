import pytorch_lightning as pl
import fastface as ff
import torch

# model device, select gpu if exists
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# widerface dataset partition
partition = 'easy' # also `medium` or `hard` can be selectable

# select and build pretrained model to test on widerface
# for this tutorial `original_lffd_560_25L_8S` is selected
# for selectable models checkout ff.list_pretrained_models()
model = ff.module.from_pretrained("original_lffd_560_25L_8S")
# model: pl.LightningModule

# get widerface average precision metric, defined in the competition
metric = ff.metric.WiderFaceAP(iou_threshold=0.5)
# metric: pl.metric.Metric

# add metric to the model
model.add_metric("widerface_ap",metric)

# get widerface data module
dm = ff.datamodule.WiderFaceDataModule(
    partitions=[partition],
    test_kwargs={
        'batch_size':1,
        'num_workers':8
    },
    test_transforms= ff.transform.Compose(
        ff.transform.Normalize(mean=127.5, std=127.5),
        ff.transform.ToTensor()
    )
)
# dm: pl.LightningDataModule

# download data if needed
dm.prepare_data(stage='test')

# setup test dataloader
dm.setup(stage='test')

# define trainer
trainer = pl.Trainer(
    logger=False,
    checkpoint_callback=False,
    gpus=1 if device == 'cuda' else 0,
    precision=32)

# run test
trainer.test(model, datamodule=dm)