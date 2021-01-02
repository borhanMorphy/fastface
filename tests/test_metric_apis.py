import pytest
import mypackage
from typing import List,Dict
from pytorch_lightning.metrics import Metric

@pytest.mark.parametrize("api",
    [
        "get_available_metrics","get_metric"
    ]
)
def test_api_exists(api):
    assert api in dir(mypackage.metric),f"{api} not found in the mypackage.metric"

def test_get_available_metrics():
    metrics = mypackage.metric.get_available_metrics()
    assert isinstance(metrics,List),f"returned value must be list but found:{type(metrics)}"
    for metric in metrics:
        assert isinstance(metric,str),f"metric must contain name as string but found:{type(metric)}"

@pytest.mark.parametrize("metric_name", mypackage.metric.get_available_metrics())
def test_list_arch_configs(metric_name:str):
    metric = mypackage.metric.get_metric(metric_name)
    assert isinstance(metric, Metric),f"returned value must be Metric but found:{type(metric)}"