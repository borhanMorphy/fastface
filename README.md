# Light Face Detector
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
* `2020.12.15` [evaluation scripts and results](#evaluation) are added
* `2020.12.13` added lffd 560_25L_8scales official weights that converted from mxnet to pytorch
* `2020.12.11` tested training script and achived 75 ap score on `widerface easy` validation set using lffd (560_25L_8scales) with random weight initialization(defined in the paper)
* `2020.12.11` added widerface evaluation metric under the `metrics/widerface_ap.py` as `pytorch_lightning.metrics.Metric`
* `2020.12.06` added lffd weight conversion script under the `tools/lffd_mx2torch.py` to convert official mxnet model weights to pytorch weights

## Usage
### Setup
install the dependencies
```
pip install -r requirements.txt
```

### Demo
run demo with pretrained model
```
python demo.py -i <your_image_file_path> --arch lffd --config 560_25L_8S --weights ./models/original_lffd_560_25L_8S.pt
```
sample output;
![alt text](resources/friends.jpg)

### Inference

### Evaluation
Evaluation on Widerface Validation Set Using `LFFD 560_25L_8S` original weights
Method|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
**LFFD(paper)**|0.910     |0.881       |0.780
**LFFD(this repository)**|0.893     |0.866       |0.756

To get these results run the following scripts
```
python test.py --arch lffd --config 560_25L_8S -mp models/original_lffd_560_25L_8S.pt -ds widerface-easy

python test.py --arch lffd --config 560_25L_8S -mp models/original_lffd_560_25L_8S.pt -ds widerface-medium

python test.py --arch lffd --config 560_25L_8S -mp models/original_lffd_560_25L_8S.pt -ds widerface-hard
```

### Training

## TODO
### Training
- [x] add resume training
- [x] add widerface dataset support
- [ ] add widerface dataset download adapter
- [ ] add FDDB dataset support
- [x] add LR Scheduler
- [x] add detector train loop
- [x] add detector val loop
- [x] add detector test loop
- [x] support AP metric
- [x] convert AP metric to `pytorch_lightning.metrics.Metric`
- [ ] implement OHNM instead of random sampling
- [ ] support lffd `320_20L_5scales` configuration
- [ ] provide lffd model weights that training from scratch

### Inference
- [ ] add demo.py
- [ ] export APIs for package usage
- [ ] add setup.py
- [ ] support model download via io utility

### Depyloment
- [ ] add bentoml support and guideline
- [ ] add ONNX support and onnx.js guideline
- [ ] add mobile demo guideline
- [ ] add torchscript support and C++ guideline

## Reference
- [LFFD Paper](https://arxiv.org/pdf/1904.10633.pdf)
- [Official LFFD Implementation](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)

## Citation
```
@inproceedings{LFFD,
title={LFFD: A Light and Fast Face Detector for Edge Devices},
author={He, Yonghao and Xu, Dezhong and Wu, Lifang and Jian, Meng and Xiang, Shiming and Pan, Chunhong},
booktitle={arXiv:1904.10633},
year={2019}
}
```