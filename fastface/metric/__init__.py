__all__ = [
    "get_metric_by_name", "list_metrics",
    "WiderFaceAP",
    "AveragePrecision",
    "AverageRecall"
]

from pytorch_lightning.metrics import Metric

from .widerface_ap import WiderFaceAP
from .ap import AveragePrecision
from .ar import AverageRecall

from typing import List

__metric_mapper__ = {
    'widerface_ap': {
        'cls': WiderFaceAP,
        'args': (),
        'kwargs': {
            'iou_threshold':0.5
        }
    },

    'ap': {
        'cls': AveragePrecision,
        'args': (),
        'kwargs': {
            'iou_threshold':0.5
        }
    },

    'ar': {
        'cls': AverageRecall,
        'args': (),
        'kwargs': {
            'iou_threshold_min':0.5,
            'iou_threshold_max':1,
        }
    }
}

def get_metric_by_name(metric_name:str, *args, **kwargs) -> Metric:
    """Returns metric that matches with the `metric_name`

    Args:
        metric_name (str): metric name

    Returns:
        Metric: pytorch_lightning.metrics.Metric instance
    """
    assert metric_name in __metric_mapper__.keys(),f"given metric name must be one of the {list(__metric_mapper__.keys())}"
    metric = __metric_mapper__[metric_name]
    args += metric['args']
    kwargs.update(metric['kwargs'])
    return metric['cls'](*args,**kwargs)

def list_metrics() -> List:
    """Returns list of available metric names

    Returns:
        List: available metric names as list of strings

    >>> import fastface as ff
    >>> ff.metric.list_metrics()
    ['ap', 'ar', 'widerface_ap']

    """
    return list(sorted(__metric_mapper__.keys()))