from .average_precision import AveragePrecision

from typing import List

__metric_mapper__ = {
    'ap': {
        'cls': AveragePrecision,
        'args': (),
        'kwargs': {
            'iou_threshold':0.5
        }
    }
}

def get_metric(metric_name:str, *args, **kwargs):
    assert metric_name in __metric_mapper__.keys(),f"given metric name must be one of the {list(__metric_mapper__.keys())}"
    metric = __metric_mapper__[metric_name]
    args += metric['args']
    kwargs.update(metric['kwargs'])
    return metric['cls'](*args,**kwargs)

def get_available_metrics() -> List:
    return list(__metric_mapper__.keys())