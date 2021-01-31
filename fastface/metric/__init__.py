__all__ = [
    "get_metric_by_name", "list_metrics",
    "WiderFaceAP"
]

from pytorch_lightning.metrics import Metric
from .widerface_ap import WiderFaceAP

from typing import List

__metric_mapper__ = {
    'widerface_ap': {
        'cls': WiderFaceAP,
        'args': (),
        'kwargs': {
            'iou_threshold':0.5
        }
    }
}

def get_metric_by_name(metric_name:str, *args, **kwargs) -> Metric:
    assert metric_name in __metric_mapper__.keys(),f"given metric name must be one of the {list(__metric_mapper__.keys())}"
    metric = __metric_mapper__[metric_name]
    args += metric['args']
    kwargs.update(metric['kwargs'])
    return metric['cls'](*args,**kwargs)

def list_metrics() -> List:
    return list(__metric_mapper__.keys())