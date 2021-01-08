# FastFace
Face detection implementations with [pytorch-lightning](https://www.pytorchlightning.ai/)

## GOAL
Supporting lightweight face detection implementations to train, test and deploy in a scalable and maintainable manner.
<br>

![PyPI](https://img.shields.io/pypi/v/fastface)
![PyPI - License](https://img.shields.io/pypi/l/fastface)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastface)
![PyPI - Downloads](https://img.shields.io/pypi/dm/fastface)

## CONTENTS
- [Recent Updates](#recent-updates)
- [Pretrained Models](#pretrained-models)
- [Usage](#usage)
- [TODO](#todo)
- [Reference](#reference)
- [Citation](#citation)

## Recent Updates
* `2021.01.08` "original_lffd_320_20L_5S" pretrained model added to the registry and can be used via fastface.module.from_pretrained api
* `2021.01.08` "320_20L_5S" configuration added to "lffd" architecture
* `2021.01.03` version 0.0.1 is out. Can be downloadable with `pip install fastface`

## Pretrained Models
Pretrained models can be accessable via `fastface.module.from_pretrained(<name>)`
Name|Architecture|Configuration|Parameters|Model Size|Link
:------:|:------:|:--------:|:--------:|:----------:|:--------:
**original_lffd_560_25L_8S**|**LFFD**|560_25L_8S|2.3M|8.8mb|[gdrive](https://drive.google.com/file/d/1xizV0s_Ei_BQcUQI_MylqC0K2SszrXP1/view?usp=sharing)
**original_lffd_320_20L_5S**|**LFFD**|320_20L_5S|1.5M|5.9mb|[gdrive](https://drive.google.com/file/d/1vA5Ywi_bJgEKwpMi9bOUD42Aaz6-fiKN/view?usp=sharing)

## Usage
### Install
From PyPI
```
pip install fastface -U
```

From source
```
pip install .
```

### Inference
Using package
```python
import fastface as ff
from cv2 import cv2

# load image as BGR
img = cv2.imread("<your_image_file_path>")

# build model with pretrained weights
model = ff.module.from_pretrained("original_lffd_560_25L_8S")
# model: pl.LightningModule

# get model summary
model.summarize()

# build required transforms
transforms = ff.transform.Compose(
    ff.transform.Interpolate(max_dim=640),
    ff.transform.Padding(target_size=(640,640)),
    ff.transform.Normalize(mean=127.5, std=127.5),
    ff.transform.ToTensor()
)

# enable tracking to perform postprocess after inference 
transforms.enable_tracking()
# reset queue
transforms.flush()

# set model to eval mode
model.eval()
# freeze model in order to disable gradient tracking
model.freeze()
# [optional] move model to gpu
model.to("cuda")

# apply transforms
batch = transforms(img)

# model inference
preds = model.predict(batch, det_threshold=.8, iou_threshold=.4)

# postprocess to adjust predictions
preds = [transforms.adjust(pred.cpu().numpy()) for pred in preds]

# preds: [np.ndarray(N,5), ...] as x1,y1,x2,y2,score

```

Using [demo.py](/demo.py) script
```
python demo.py --model original_lffd_560_25L_8S --device cuda --input <your_image_file_path>
```
sample output;
![alt text](resources/friends.jpg)

### Evaluation
Evaluation on Widerface Validation Set
Architecture|Configuration|Easy Set|Medium Set|Hard Set
------|--------|--------|----------|--------
**LFFD(paper)**|560_25L_8S     |0.910     |0.881       |0.780
**LFFD**|560_25L_8S     |0.893     |0.866       |0.756
**LFFD**|320_20L_5S     |0.854     |0.845       |0.735

To get these results, run the following scripts
```
# for easy set
python test_widerface.py --model <pretrained_model> --device cuda --partition easy

# for medium set
python test_widerface.py --model <pretrained_model> --device cuda --partition medium

# for hard set
python test_widerface.py --model <pretrained_model> --device cuda --partition hard
```

### Training


## TODO
### Feature
- [x] add lffd `320_20L_5S` configuration to the arch
- [x] add lffd `320_20L_5S` pytorch model to the registry using original repository
- [x] add lffd `320_20L_5S` training configuration to config_zoo
- [ ] add `FDDB` dataset
- [ ] add `FDDB` datamodule

### Training
- [ ] add FDDB dataset support
- [ ] provide lffd model weights that trained from scratch

### Inference

### Depyloment
- [ ] add bentoml support and guideline
- [ ] add ONNX support and onnx.js guideline
- [ ] add mobile demo guideline
- [ ] add torchscript support and C++ guideline

## Reference
- [LFFD Paper](https://arxiv.org/pdf/1904.10633.pdf)
- [Official LFFD Implementation](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)

## Citation
```bibtex
@inproceedings{LFFD,
    title={LFFD: A Light and Fast Face Detector for Edge Devices},
    author={He, Yonghao and Xu, Dezhong and Wu, Lifang and Jian, Meng and Xiang, Shiming and Pan, Chunhong},
    booktitle={arXiv:1904.10633},
    year={2019}
}
```
