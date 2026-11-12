# Ported from torch 2.15.0 (BSD-3-Clause)

from .optimizer import Optimizer, ParamGroup
from .sgd import SGD
from .adam import Adam, AdamW
import torch.optim.lr_scheduler as lr_scheduler
