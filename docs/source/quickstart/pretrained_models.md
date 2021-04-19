# Pretrained Models
**fastface** offers pretrained models and can be easly accessable **without manually downloading weights**.<br>


## Model Zoo

Name|Widerface-Easy Val AP|Widerface-Medium Val AP|Widerface-Hard Val AP|Link
:------:|:------:|:------:|:------:|:------:
**lffd_original**|0.893|0.866|0.756|[weights](https://drive.google.com/file/d/1xizV0s_Ei_BQcUQI_MylqC0K2SszrXP1/view?usp=sharing)
**lffd_slim**|0.854|0.845|0.735|[weights](https://drive.google.com/file/d/1vA5Ywi_bJgEKwpMi9bOUD42Aaz6-fiKN/view?usp=sharing)

## Usage
To get any of pretrained models as `pl.LightningModule`
```python
import fastface as ff
model = ff.FaceDetector.from_pretrained("<name>")
```
If you don't have pretrained model weights, **fastface** will automatically download and put it under `$HOME/.cache/fastface/<package_version>/model/`