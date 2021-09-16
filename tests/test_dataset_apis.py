import pytest

import fastface as ff

# TODO expand this


@pytest.mark.parametrize("dataset_name", ["FDDBDataset", "WiderFaceDataset"])
def test_api_exists(dataset_name: str):
    assert hasattr(
        ff.dataset, dataset_name
    ), "{} not found in the fastface.dataset".format(dataset_name)
