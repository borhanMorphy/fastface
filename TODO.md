## Bugs
- [x] fix performance drop for tutorials/widerface_benchmark
- [ ] average recall seems to be bugged, check it out

## Hot
- [x] fill `Inference` section under the `CORE CONCEPTS`
- [x] fill `Testing` section under the `CORE CONCEPTS`
- [x] fill `Training` section under the `CORE CONCEPTS`
- [x] fill `Export` section under the `CORE CONCEPTS`
- [x] make sure tutorials works
- [ ] add ci/cd for unittest, linting test and doc test
- [ ] add coverage badge
- [ ] add ci/cd pipeline badge
- [ ] add doctest badge
- [ ] test on windows 10
- [x] add FDDB dataset
- [ ] add UFDD dataset

## Near Future
- [ ] add `ADVENCED GUIDE` to the docs
- [ ] increase coverage of pydoc and doc-test for apis
- [ ] add ONNX & torchscipt usecase deployment notebooks

## Future
- [ ] extend Architecture docs
- [ ] extend pytest.ini configuration
- [ ] support CoreML
- [ ] add CoreML deployment tutorial

## Maybe
- [ ] support TFLite


## Now
- [x] ! decide build targets !

- [x] define new ArcConfig type
    * [x] LFFDConfig
    * [x] CenterFaceConfig

- [x] define new ArchInterface type
- [x] define _Factory methods
    * [x] register
    * [x] get_arch_names
    * [x] get_arch_config_names
    * [x] get_arch_config
    * [x] build_arch

- [ ] document fastface types
    * [ ] ArchConfig
    * [ ] ArchInterface
    * [ ] BaseDataset
    * [ ] FaceDetector

- [x] apply ArchInterface
    * [x] LFFD
    * [x] CenterFace

- [x] adjust Arch to ArchConfig
    * [x] LFFD
    * [x] CenterFace

- [x] integrate albumentations
    * [x] FDDB
    * [x] WiderFace

- [x] update centerface
    * [x] without landmark
    * [x] add landmark support

- [x] train centerface
    * [x] without landmark
    * [x] add landmark support

- [ ] add benchmark ops (Widerface, FDDB)
    * [x] Widerface benchmark
    * [ ] FDDB benchmark

- [ ] add guide for manual download of datasets

- [ ] move model download from drive to torchub or s3

- [ ] update test cases
    * [x] test_archs.py
    * [x] test_base_apis.py
    * [x] test_dataset_apis.py
    * [ ] test_module_apis.py
    * [ ] test_utility_apis.py

- [x] update dependecies
    * [x] kornia
    * [x] albumentations
    * [x] torchmetrics

- [ ] expand Makefile

- [ ] update FaceDetection API
    * [x] handle dynamic
    * [ ] add landmark to FaceDetection API

- [ ] train on Linode
