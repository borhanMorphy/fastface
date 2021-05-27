from typing import Union
import random
import torch
import math

class KMeans():
    def __init__(self, k: int, max_iter: int = 5,
            distance_fn=None, dim_size: int = 2):
        assert distance_fn is not None, "please provide a distance function"

        self._params = torch.empty(k, dim_size, dtype=torch.float32)
        self._max_iter = max_iter
        self._distance_fn = distance_fn
        self._best_distance_score = math.inf

    def fit(self, points: torch.Tensor):
        assert len(points.shape) == 2, "shape length of the points\
             must be 2 but found {}".format(len(points.shape))

        sample_size = points.size(0)

        self._params = points[random.sample(range(sample_size), k=self._params.size(0)), :]
        latest_params = torch.zeros_like(self._params)

        while 1:
            # points: torch.Tensor(sample_size, dim_size)
            # self._params: torch.Tensor(k, dim_size)
            dists = self._distance_fn(points, self._params)
            # dists: torch.Tensor(sample_size, k)


            assigned_clusters = torch.argmin(dists, axis=1)

            if (latest_params == assigned_clusters).all():
                break

            for i in range(self.k):
                self._params[i] = boxes[assigned_clusters == i].median(dim=0)[0]

            last_clusters = assigned_clusters

            if ()

