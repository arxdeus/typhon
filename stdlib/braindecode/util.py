# Ported from braindecode 2.0.0 (BSD-3-Clause)

import numpy as np
import random
import torch
from ._random_state import seed as _legacy_seed


def set_random_seeds(seed: int, cuda: bool = False, cudnn_benchmark = None):
    """Seed python random, torch and the legacy numpy stream (cuda ignored)."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    _legacy_seed(seed)


def np_to_th(X, requires_grad: bool = False, dtype = None, pin_memory: bool = False, **kwargs):
    t = torch.from_numpy(np.ascontiguousarray(np.asarray(X)))
    if requires_grad:
        t.requires_grad_(True)
    return t


def th_to_np(var, dtype: type = np.float32, ndim: Literal[int] = 2):
    return var.detach().numpy(dtype, ndim)
