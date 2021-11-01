from typing import List
import pytest

import fastface as ff

def get_dataset_names() -> List[str]:
    dataset_names = [
        dataset_name
        for dataset_name in dir(ff.dataset)
        if dataset_name.endswith("Dataset")
    ]

    dataset_names.pop(dataset_names.index("BaseDataset"))

    return dataset_names


@pytest.mark.parametrize("dataset_name", get_dataset_names())
def test_api_exists(dataset_name: str):
    assert hasattr(
        ff.dataset, dataset_name
    ), "{} not found in the fastface.dataset".format(dataset_name)


@pytest.mark.parametrize("dataset_name", get_dataset_names())
def test_dataset_name_exists(dataset_name: str):
    key_variable = "__DATASET_NAME__"
    assert hasattr(
        getattr(ff.dataset, dataset_name), key_variable
    ), "fastface.dataset.{} not contains `{}` ".format(dataset_name, key_variable)


@pytest.mark.parametrize("dataset_name", get_dataset_names())
def test_dataset_urls_exist(dataset_name: str):
    key_variable = "__URLS__"
    assert hasattr(
        getattr(ff.dataset, dataset_name), key_variable
    ), "fastface.dataset.{} not contains `{}` ".format(dataset_name, key_variable)
