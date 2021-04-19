import fastface as ff
import torch
import time
import numpy as np
from tqdm import tqdm

dataset = "widerface"
arch = "yolov4"
config = "tiny"
img_size = 608

transforms = ff.transform.Compose(
    ff.transform.Interpolate(max_dim=img_size),
    ff.transform.Padding(target_size=(img_size, img_size), pad_value=0),
)

def target_transform(targets: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(targets)

source_dir = ff.utils.cache.get_data_cache_path(dataset)

# TODO make it parametric
ds = ff.dataset.WiderFaceDataset(source_dir, transforms=transforms,
    target_transform=target_transform, phase='train')

average_recall = ff.metric.get_metric_by_name("ar")

arck_cls = ff.utils.config.get_arch_cls(arch)

anchors = arck_cls.get_anchor_generators(config)

counter = 100
for img, targets in tqdm(ds):
    counter -= 1
    h,w = img.shape[:2]
    preds = torch.cat([anchor.estimated_forward(h,w).reshape(-1,4) for anchor in anchors], dim=0)
    average_recall.update(preds, targets)
    if counter == 0:
        break

print("average recall: ", average_recall.compute())