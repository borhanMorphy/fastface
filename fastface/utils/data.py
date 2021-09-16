import numpy as np
import torch


def default_collate_fn(batch):
    batch, targets = zip(*batch)
    batch = np.stack(batch, axis=0).astype(np.float32)
    batch = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
    for i, target in enumerate(targets):
        for k, v in target.items():
            if isinstance(v, np.ndarray):
                targets[i][k] = torch.from_numpy(v)

    return batch, targets
