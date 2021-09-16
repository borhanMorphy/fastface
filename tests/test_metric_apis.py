import pytest
from pytorch_lightning.metrics import Metric

import fastface as ff

# TODO expand this


@pytest.mark.parametrize(
    "metric_name", ["AveragePrecision", "AverageRecall", "WiderFaceAP"]
)
def test_api_exists(metric_name: str):
    assert hasattr(
        ff.metric, metric_name
    ), "{} not found in the fastface.metric".format(metric_name)


@pytest.mark.parametrize(
    "metric_name", ["AveragePrecision", "AverageRecall", "WiderFaceAP"]
)
def test_get_available_metrics(metric_name: str):
    metric_cls = getattr(ff.metric, metric_name)
    metric = metric_cls()
    assert isinstance(
        metric, Metric
    ), "returned value must be `pytorch_lightning.metrics.Metric` but found:{}".format(
        type(metric)
    )
