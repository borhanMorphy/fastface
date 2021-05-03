import pytest
import fastface as ff

# TODO expand this

@pytest.mark.parametrize("datamodule_name",
    [
        "FDDBDataModule",
        "WiderFaceDataModule"
    ]
)
def test_api_exists(datamodule_name: str):
    assert hasattr(ff.datamodule, datamodule_name), "{} not found in the fastface.dataset".format(datamodule_name)
