import math
import random

import torch


class KMeans:
    """Test"""

    def __init__(self, k: int, distance_fn=None, dim_size: int = 2, nstart: int = 2):
        # TODO use nstart
        assert distance_fn is not None, "please provide a distance function"

        self._params = torch.empty(k, dim_size, dtype=torch.float32)
        self._distance_fn = distance_fn
        self._best_distance_score = math.inf

    def fit(self, points: torch.Tensor):
        assert (
            len(points.shape) == 2
        ), "shape length of the points \
            must be 2 but found {}".format(
            len(points.shape)
        )
        assert isinstance(
            points, torch.Tensor
        ), "points must be torch.tensor but found {}".format(type(points))
        sample_size = points.size(0)
        k = self._params.size(0)

        self._params = points[random.sample(range(sample_size), k=k), :]
        # self._params: torch.Tensor(k, dim_size)

        latest_cluster = torch.zeros(sample_size, dtype=torch.long)
        # latest_cluster: torch.Tensor(sample_size)

        while 1:
            # points: torch.Tensor(sample_size, dim_size)
            # self._params: torch.Tensor(k, dim_size)
            dists = self._distance_fn(points, self._params)
            # dists: torch.Tensor(sample_size, k)

            assigned_clusters = torch.argmin(dists, dim=1)
            # assigned_clusters: torch.Tensor(sample_size)

            if (latest_cluster == assigned_clusters).all():
                # break if converged
                break

            for i in range(k):
                self._params[i] = points[assigned_clusters == i, :].median(dim=0)[0]

            latest_cluster = assigned_clusters
