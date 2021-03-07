import fastface as ff
import torch
from debug.kmeans import KMeans
import time
import numpy as np

img_size = 416

transforms = ff.transform.Compose(
    ff.transform.Interpolate(max_dim=img_size),
    ff.transform.Padding(target_size=(img_size,img_size), pad_value=0),
    ff.transform.Normalize(mean=0, std=255),
    ff.transform.ToTensor()
)

source_dir = ff.utils.cache.get_data_cache_path("widerface")
ff.utils.random.seed_everything(42)

ds = ff.dataset.WiderFaceDataset(source_dir, phase='train')

boxes = torch.from_numpy(np.concatenate(ds.targets, axis=0))
min_d = (boxes[:, [2,3]] - boxes[:, [0,1]]).min(dim=-1)[0]

boxes = (boxes[:, [2,3]] - boxes[:, [0,1]]) / img_size

km = KMeans(8, nstart=10)
st = time.time()

clusters,prior_boxes = km.estimate_anchors(boxes)
print(clusters)
print(prior_boxes)
print(f"took: {time.time()-st}")