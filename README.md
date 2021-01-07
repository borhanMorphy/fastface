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
- [Usage](#usage)
- [TODO](#todo)
- [Reference](#reference)
- [Citation](#citation)

## Recent Updates
* `2021.01.03` version 0.0.1 is out. Can be downloadable with `pip install fastface`

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
