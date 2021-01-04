# FastFace
Face detection implementations with [pytorch-lightning](https://www.pytorchlightning.ai/)

## GOAL
Supporting lightweight face detection implementations to train, test and deploy in a scalable and maintainable manner.

## CONTENTS
- [Recent Update](#recent-update)
- [Usage](#usage)
- [TODO](#todo)
- [Reference](#reference)
- [Citation](#citation)

## Recent Update
* `2021.01.02` added unittests for apis under the `tests/` directory
* `2021.01.02` online hard negative mining is added for lffd training
* `2021.01.02` caching is supported and by default it will use `~/.cache/fastface`
* `2021.01.02` with fastface.adapters , models and datasets can be downloadable via gdrive or requests
* `2021.01.02` now this repository can be usable as package
* `2020.12.15` [evaluation scripts and results](#evaluation) are added
* `2020.12.13` added lffd 560_25L_8scales official weights that converted from mxnet to pytorch
* `2020.12.11` tested training script and after 50 epochs, achived 75 ap score on widerface-easy validation set using lffd (560_25L_8scales) with random weight initialization(defined in the paper)
* `2020.12.11` added widerface evaluation metric under the `metrics/widerface_ap.py` as `pytorch_lightning.metrics.Metric`
* `2020.12.06` added lffd weight conversion script under the `tools/lffd_mx2torch.py` to convert official mxnet model weights to pytorch weights

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
import fastface
from fastface.transform import (
    Compose,
    Interpolate,
    Padding,
    Normalize,
    ToTensor
)
from cv2 import cv2

# load image
img = cv2.imread("<your_image_file_path>")

# build model with pretrained weights
model = fastface.module.from_pretrained("original_lffd_560_25L_8S")
# model: pl.LightningModule

# build required transforms
transforms = Compose(
    Interpolate(max_dim=640),
    Padding(target_size=(640,640)),
    Normalize(mean=127.5, std=127.5),
    ToTensor()
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

print(preds)
"""
[
    np.array(N,5), # as x1,y1,x2,y2,score
    ...
]
"""
```

Using [demo.py](/demo.py) script
```
python demo.py --model original_lffd_560_25L_8S --device cuda --input <your_image_file_path>
```
sample output;
![alt text](resources/friends.jpg)

### Evaluation
Evaluation on Widerface Validation Set Using `LFFD 560_25L_8S` original weights
Method|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
**LFFD(paper)**|0.910     |0.881       |0.780
**LFFD(this repository)**|0.893     |0.866       |0.756

To get these results, run the following scripts
```
# for easy set
python test_widerface.py --model original_lffd_560_25L_8S --device cuda --partition easy

# for medium set
python test_widerface.py --model original_lffd_560_25L_8S --device cuda --partition medium

# for hard set
python test_widerface.py --model original_lffd_560_25L_8S --device cuda --partition hard
```

### Training
Train `LFFD 560_25L_8S` on widerface dataset
```
python train_widerface.py --yaml config_zoo/lffd.original.yaml
```

## TODO
### Feature
- [ ] add lffd `320_20L_5S` configuration to the arch
- [ ] add lffd `320_20L_5S` pytorch model to the registry using original repository
- [ ] add lffd `320_20L_5S` training configuration to config_zoo
- [ ] add `FDDB` dataset
- [ ] add `FDDB` datamodule

### Training
- [x] add resume training
- [x] add widerface dataset support
- [x] add widerface dataset download adapter
- [ ] add FDDB dataset support
- [x] add LR Scheduler
- [x] add detector train loop
- [x] add detector val loop
- [x] add detector test loop
- [x] support AP metric
- [x] convert AP metric to `pytorch_lightning.metrics.Metric`
- [x] implement OHNM instead of random sampling
- [ ] provide lffd model weights that training from scratch

### Inference
- [x] add demo.py
- [x] export APIs for package usage
- [x] add setup.py
- [x] support model download via io utility

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
