import pytest
from pytorch_lightning.metrics import Metric
import fastface as ff

@pytest.mark.parametrize("api",
    [
        "list_metrics",
        "get_metric_by_name"
    ]
)
def test_api_exists(api):
    assert api in dir(ff.metric), f"{api} not found in the fastface.metric"

def test_get_available_metrics():
    metrics = ff.metric.list_metrics()
    assert isinstance(metrics, list), f"returned value must be list but found:{type(metrics)}"
    for metric in metrics:
        assert isinstance(metric,str), f"metric must contain name as string but found:{type(metric)}"

@pytest.mark.parametrize("metric_name", ff.metric.list_metrics())
def test_list_arch_configs(metric_name: str):
    metric = ff.metric.get_metric_by_name(metric_name)
    assert isinstance(metric, Metric), f"returned value must be Metric but found:{type(metric)}"
