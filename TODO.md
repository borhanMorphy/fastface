# mypackage APIs

## Unittest
### Discovery APIs (mypackage)
- [x] unittest for list_archs
- [x] unittest for list_arch_configs
- [x] unittest for get_arch_config

### Module APIs (mypackage.module)
- [x] unittest for build
- [ ] unittest for from_checkpoint
- [ ] unittest for from_pretrained

### Dataset APIs (mypackage.dataset)
- [ ] unittest for WiderFaceDataset

### Datamodule APIs (mypackage.datamodule)
- [ ] unittest for WiderFaceDataModule

### Metric APIs (mypackage.metric)
- [x] unittest for get_metric (has unittest)
- [x] unittest for get_available_metrics (has unittest)

### Transform APIs (mypackage.transform)
- [ ] unittest for Compose
- [ ] unittest for Interpolate
- [ ] unittest for Padding
- [ ] unittest for FaceDiscarder
- [ ] unittest for ToTensor
- [ ] unittest for Normalize
- [ ] unittest for LFFDRandomSample
- [ ] unittest for RandomHorizontalFlip

### Utility Functions (mypackage.utils)
- [ ] unittest for cache.get_cache_path
- [ ] unittest for cache.get_model_cache_path
- [ ] unittest for cache.get_data_cache_path
- [ ] unittest for config.get_pkg_root_path
- [ ] unittest for config.get_pkg_arch_path
- [ ] unittest for config.discover_archs
- [ ] unittest for config.get_arch_pkg
- [ ] unittest for config.get_arch_cls
- [ ] unittest for utils.seed_everything
- [ ] unittest for utils.random_sample_selection
- [ ] unittest for utils.get_best_checkpoint_path
- [ ] unittest for visualize.prettify_detections

## Documentation
- [ ] update README.md

### Discovery APIs (mypackage)
- [ ] Api Reference Documentation
- [ ] Usage Examples

### Module APIs (mypackage.module)
- [ ] Api Reference Documentation
- [ ] Usage Examples

### Dataset APIs (mypackage.dataset)
- [ ] Api Reference Documentation
- [ ] Usage Examples

### Datamodule APIs (mypackage.datamodule)
- [ ] Api Reference Documentation
- [ ] Usage Examples

### Metric APIs (mypackage.metric)
- [ ] Api Reference Documentation
- [ ] Usage Examples

### Transform APIs (mypackage.transform)
- [ ] Api Reference Documentation
- [ ] Usage Examples

### Utility Functions (mypackage.utils)
- [ ] Api Reference Documentation
- [ ] Usage Examples

## Features
- [ ] add this repository to pypi as a package
- [ ] add lffd `320_20L_5S` configuration to the arch
- [ ] add lffd `320_20L_5S` pytorch model to the registry using original repository
- [ ] add lffd `320_20L_5S` training configuration to config_zoo
- [ ] add `FDDB` dataset
- [ ] add `FDDB` datamodule