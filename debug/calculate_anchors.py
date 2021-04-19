import fastface as ff
import torch
from debug.kmeans import KMeans
import time
import numpy as np
from tqdm import tqdm

img_size = 608

transforms = ff.transform.Compose(
    ff.transform.Interpolate(max_dim=img_size),
    ff.transform.Padding(target_size=(img_size, img_size), pad_value=0),
)

source_dir = ff.utils.cache.get_data_cache_path("widerface")

ds = ff.dataset.WiderFaceDataset(source_dir, transforms=transforms, phase='train')

all_boxes = []

for img,targets in tqdm(ds):
    all_boxes.append(torch.from_numpy(targets).float())

all_boxes = torch.cat(all_boxes, dim=0)

km = KMeans(6, nstart=5)
st = time.time()

all_boxes = (all_boxes[:, [2, 3]] - all_boxes[:, [0, 1]]) / img_size

clusters, prior_boxes = km.estimate_anchors(all_boxes)
print(clusters)
print(prior_boxes)
print(f"took: {time.time()-st}")
