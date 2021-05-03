# API REFERENCES

## [x] Base APIs (tests/test_base_apis.py)
- [x] fastface.list_pretrained_models
- [x] fastface.download_pretrained_model
- [x] fastface.list_archs
- [x] fastface.list_arch_configs
- [x] fastface.get_arch_config

## [x] Module APIs (tests/test_module_apis.py)
- [x] fastface.FaceDetector.build
- [x] fastface.FaceDetector.build_from_yaml
- [x] fastface.FaceDetector.from_pretrained
- [x] fastface.FaceDetector.from_checkpoint
- [x] fastface.FaceDetector.predict
- [x] fastface.FaceDetector.forward

## [x] Metric APIs (tests/test_metric_apis.py)
- [x] fastface.metric.AveragePrecision
- [x] fastface.metric.AverageRecall
- [x] fastface.metric.WiderFaceAP

## [x] Loss APIs (tests/test_loss_apis.py)
- [x] fastface.loss.BinaryFocalLoss
- [x] fastface.loss.DIoULoss

## [ ] Transforms APIs (tests/test_transforms_apis.py)
- [x] fastface.transforms.Interpolate
- [x] fastface.transforms.Padding
- [ ] fastface.transforms.FaceDiscarder
- [x] fastface.transforms.Normalize
- [ ] fastface.transforms.LFFDRandomSample
- [ ] fastface.transforms.RandomHorizontalFlip
- [ ] fastface.transforms.Compose

## [ ] Preprocess APIs (tests/test_preprocess_apis.py)
- [x] fastface.preprocess.Interpolate
- [x] fastface.preprocess.DummyInterpolate
- [x] fastface.preprocess.Pad
- [x] fastface.preprocess.DummyPad
- [x] fastface.preprocess.Normalize
- [ ] fastface.preprocess.Preprocess

## [x] Dataset APIs (tests/test_dataset_apis.py)
- [x] fastface.dataset.FDDBDataset
- [x] fastface.dataset.UFDDDataset
- [x] fastface.dataset.WiderFaceDataset

## [x] Datamodule APIs (tests/test_datamodule_apis.py)
- [x] fastface.datamodule.FDDBDataModule
- [x] fastface.datamodule.UFDDDataModule
- [x] fastface.datamodule.WiderFaceDataModule

## [ ] Utility APIs (tests/test_utility_apis.py)
- [ ] fastface.utils.box.generate_grids
- [ ] fastface.utils.box.jaccard_vectorized
- [ ] fastface.utils.box.intersect
- [ ] fastface.utils.box.cxcywh2xyxy
- [ ] fastface.utils.box.xyxy2cxcywh
- [ ] fastface.utils.box.batched_nms

- [x] fastface.utils.cache.get_cache_dir
- [x] fastface.utils.cache.get_model_cache_dir
- [x] fastface.utils.cache.get_data_cache_dir
- [x] fastface.utils.cache.get_checkpoint_cache_dir

- [x] fastface.utils.config.get_pkg_root_path
- [x] fastface.utils.config.get_pkg_arch_path
- [x] fastface.utils.config.get_registry_path
- [x] fastface.utils.config.get_registry
- [x] fastface.utils.config.discover_archs
- [x] fastface.utils.config.get_arch_pkg
- [x] fastface.utils.config.get_arch_cls

- [ ] fastface.utils.preprocess.prepare_batch
- [ ] fastface.utils.preprocess.adjust_results

- [ ] fastface.utils.visualize.prettify_detections
- [ ] fastface.utils.visualize.render_targets
- [ ] fastface.utils.visualize.draw_rects

TODO add random