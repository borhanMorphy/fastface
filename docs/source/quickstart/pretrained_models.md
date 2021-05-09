# Pretrained Models
**fastface** offers pretrained models and can be easly accessable **without manually downloading weights**.<br>

## Model Zoo

Name|Architecture|Configuration|Parameters|Model Size|Link
:------:|:------:|:------:|:------:|:------:|:------:
**lffd_original**|lffd|original|2.3M|9mb|[weights](https://drive.google.com/file/d/1qFRuGhzoMWrW9WNlWw9jHXPY51MBssQD/view?usp=sharing)
**lffd_slim**|lffd|slim|1.5M|6mb|[weights](https://drive.google.com/file/d/1UOHllYp5NY4mV7lHmq0c9xsryRIufpAQ/view?usp=sharing)

## Usage
To get any of pretrained models as `pl.LightningModule`
```python
import fastface as ff
model = ff.FaceDetector.from_pretrained("<name>")
```
If you don't have pretrained model weights, **fastface** will automatically download and put it under `$HOME/.cache/fastface/<package_version>/model/`