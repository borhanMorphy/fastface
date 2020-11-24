# Light Face Detector

Face detection implementation using [pytorch-lightning](https://www.pytorchlightning.ai/)

## TODOs
### Training
- [x] add checkpoint support
- [x] add widerface dataset support
- [ ] add FDDB dataset support
- [x] add detector train loop
- [x] add detector val loop
- [ ] add detector test loop
- [x] support AP metric
- [ ] convert AP metric to `pytorch_lightning.metrics.Metric`
- [ ] implement OHNM instead of random sampling

### Inference
- [ ] add demo.py
- [ ] export APIs for package usage
- [ ] create setup.py

### Depyloment
- [ ] add bentoml support and guideline
- [ ] add ONNX support and onnx.js guideline
- [ ] add mobile demo guideline
- [ ] add torchscript support and C++ guideline

### Style
- [ ] Refactor README.md file