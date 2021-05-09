# Widerface Benchmark Tutorial

## Setup
Install latest version of `fastface` with
```
pip install fastface -U
```

## Discovery
`fastface` is packed with varius pretrained models, to see full list run the following
```
python -c "import fastface as ff;print(ff.list_pretrained_models())"
```
Output will be look like
```
['lffd_original', 'lffd_slim']
```

## Start

Lets import required packages
```python
import fastface as ff
import pytorch_lightning as pl
import torch
```

Build pretrained model. For this tutorial `lffd_original` is selected but you can also select another model
```python
model = ff.FaceDetector.from_pretrained("lffd_original")
# model: pl.LightningModule
```

**If you don't have pretrained model weights on your PC, `fastface` will automatically download and put it under `$HOME/.cache/fastface/<package_version>/model/`**


Add widerface average precision(defined in the widerface competition) metric to the model
```python
metric = ff.metric.WiderFaceAP(iou_threshold=0.5)
# metric: pl.metrics.Metric

# add metric to the model
model.add_metric("widerface_ap", metric)
```

Define widerface dataset. For this tutorial `easy` partition is selected but `medium` or `hard` partitions are also available<br>
**`Warning!` Do not use `batch_size` > 1**, because tensors can not be stacked due to different size of images. Also using fixed image size drops metric performance.
```python
ds = ff.dataset.WiderFaceDataset(
    phase="test",
    partitions=["easy"],
    transforms= ff.transforms.Compose(
        ff.transforms.ConditionalInterpolate(max_size=1500),
    )
)
# ds: torch.utils.data.Dataset

# get dataloader
dl = ds.get_dataloader(batch_size=1, num_workers=1)
# dl: torch.utils.data.DataLoader
```

**If you don't have widerface validation dataset on your PC, `fastface` will automatically download and put it under `$HOME/.cache/fastface/<package_version>/data/widerface/`**

Define `pytorch_lightning.Trainer`
```python
trainer = pl.Trainer(
    benchmark=True,
    logger=False,
    checkpoint_callback=False,
    gpus=1 if torch.cuda.is_available() else 0,
    precision=32)
```

Run test
```python
trainer.test(model, test_dataloaders=dl)
```

You should get output like this after test is done

```script
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'widerface_ap': 0.8929094818903156}
--------------------------------------------------------------------------------
```

Checkout [test_widerface.py](./test_widerface.py) script to see full code
