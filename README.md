# FastFace
Lightweight face detection framework using [pytorch-lightning](https://www.pytorchlightning.ai/)

## Goal
Supporting lightweight face detection implementations to train, test and deploy in a scalable and maintainable manner.
<br>

![PyPI](https://img.shields.io/pypi/v/fastface)
![PyPI - License](https://img.shields.io/pypi/l/fastface)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastface)
[![Downloads](https://pepy.tech/badge/fastface)](https://pepy.tech/project/fastface)

## Contents
- [Recent Updates](#recent-updates)
- [Installation](#installation)
- [Architectures](#architectures)
- [Pretrained Models](#pretrained-models)
- [Demo](#demo)
- [Tutorials](#tutorials)
- [References](#references)
- [Citations](#citations)

## Recent Updates
* `2021.02.03` added [widerface benchmark tutorial](./tutorials/widerface_benchmark/README.md) to replicate results.
* `2021.02.01` updated `module.predict` api to perform transform operation under the hood for simplicity.
* `2021.01.08` "original_lffd_320_20L_5S" pretrained model added to the registry and can be used via fastface.module.from_pretrained api
* `2021.01.08` "320_20L_5S" configuration added to "lffd" architecture
* `2021.01.03` version 0.0.1 is out. Can be downloadable with `pip install fastface`

## Installation
From PyPI
```
pip install fastface -U
```

From source
```
git clone https://github.com/borhanMorphy/light-face-detection.git
cd light-face-detection
pip install .
```

## Architectures
Available architectures
Architecture|Configuration|Parameters|Model Size|
:------:|:------:|:--------:|:--------:
**lffd**|560_25L_8S|2.3M|8.8mb
**lffd**|320_20L_5S|1.5M|5.9mb

## Pretrained Models
Pretrained models can be accessable via `fastface.module.from_pretrained(<name>)`
Name|Widerface-Easy Val AP|Widerface-Medium Val AP|Widerface-Hard Val AP|Link
:------:|:------:|:------:|:------:|:------:
**original_lffd_560_25L_8S**|0.893|0.866|0.756|[weights](https://drive.google.com/file/d/1xizV0s_Ei_BQcUQI_MylqC0K2SszrXP1/view?usp=sharing)
**original_lffd_320_20L_5S**|0.854|0.845|0.735|[weights](https://drive.google.com/file/d/1vA5Ywi_bJgEKwpMi9bOUD42Aaz6-fiKN/view?usp=sharing)

## Demo
Using package
```python
import fastface as ff
import imageio

# load image as RGB
img = imageio.imread("<your_image_file_path>")[:,:,:3]

# build model with pretrained weights
model = ff.module.from_pretrained("original_lffd_560_25L_8S")
# model: pl.LightningModule

# get model summary
model.summarize()

# set model to eval mode
model.eval()

# [optional] move model to gpu
model.to("cuda")

# model inference
preds = model.predict(img, det_threshold=.8, iou_threshold=.4)
# preds: [
#   {
#       'box': [x1,y1,x2,y2],
#       'score':<float>
#   },
#   ...
# ]

```

Using [demo.py](/demo.py) script
```
python demo.py --model original_lffd_560_25L_8S --device cuda --input <your_image_file_path>
```
sample output;
![alt text](resources/friends.jpg)

## Tutorials
* [Widerface Benchmark](./tutorials/widerface_benchmark/README.md)

## References
- [LFFD Paper](https://arxiv.org/pdf/1904.10633.pdf)
- [Official LFFD Implementation](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)

## Citations
```bibtex
@inproceedings{LFFD,
    title={LFFD: A Light and Fast Face Detector for Edge Devices},
    author={He, Yonghao and Xu, Dezhong and Wu, Lifang and Jian, Meng and Xiang, Shiming and Pan, Chunhong},
    booktitle={arXiv:1904.10633},
    year={2019}
}
```
