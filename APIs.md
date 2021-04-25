# API REFERENCES

## Base APIs
- fastface.list_pretrained_models
- fastface.download_pretrained_model
- fastface.list_archs
- fastface.list_arch_configs
- fastface.get_arch_config

## Model APIs
- fastface.FaceDetector.build
- fastface.FaceDetector.from_pretrained
- fastface.FaceDetector.from_checkpoint

## Metric APIs
- fastface.metric.get_metric_by_name
- fastface.metric.list_metrics
- fastface.metric.WiderFaceAP

## Loss APIs
- fastface.loss.get_loss_by_name
- fastface.loss.list_losses
- fastface.loss.BinaryCrossEntropyLoss
- fastface.loss.L2Loss

## Transform APIs
- fastface.transforms.Interpolate
- fastface.transforms.Padding
- fastface.transforms.FaceDiscarder
- fastface.transforms.ToTensor
- fastface.transforms.Normalize
- fastface.transforms.LFFDRandomSample
- fastface.transforms.RandomHorizontalFlip
- fastface.transforms.Compose

## Dataset APIs
- fastface.dataset.WiderFaceDataset

## Datamodule APIs
- fastface.datamodule.WiderFaceDataModule