"""Small training utilities: seeding, meters, EMA."""
import copy
import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val, k=1):
        self.sum += float(val) * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


class ModelEMA:
    """Exponential moving average of weights - steadier val QWK."""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)
