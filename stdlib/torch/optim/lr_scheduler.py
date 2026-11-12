# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Chainable (recursive) learning-rate schedules, matching torch's float
# arithmetic step by step.

import math
from .optimizer import Optimizer

class LRScheduler:
    optimizer: Optimizer
    base_lrs: List[float]
    last_epoch: int
    _step_count: int
    _is_initial: bool
    _last_lr: List[float]

    def _setup(self, optimizer: Optimizer, last_epoch: int):
        self.optimizer = optimizer
        if last_epoch == -1:
            for g in optimizer.param_groups:
                g.initial_lr = g.lr
        self.base_lrs = [g.initial_lr for g in optimizer.param_groups]
        self.last_epoch = last_epoch
        self._step_count = 0
        self._is_initial = True
        self.step()
        self._is_initial = False

    def get_lr(self) -> List[float]:
        raise NotImplementedError("get_lr")

    def get_last_lr(self) -> List[float]:
        return self._last_lr

    def step(self):
        self._step_count += 1
        self.last_epoch += 1
        lrs = self.get_lr()
        for i in range(len(lrs)):
            self.optimizer.param_groups[i].lr = lrs[i]
        self._last_lr = [g.lr for g in self.optimizer.param_groups]

    def _cur(self) -> List[float]:
        return [g.lr for g in self.optimizer.param_groups]

class StepLR(LRScheduler):
    step_size: int
    gamma: float

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        self._setup(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self._cur()
        return [lr * self.gamma for lr in self._cur()]

class MultiStepLR(LRScheduler):
    milestones: List[int]
    gamma: float

    def __init__(self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self._setup(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        c = self.milestones.count(self.last_epoch)
        if c == 0:
            return self._cur()
        return [lr * self.gamma ** c for lr in self._cur()]

class ExponentialLR(LRScheduler):
    gamma: float

    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1):
        self.gamma = gamma
        self._setup(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self._is_initial:
            return self._cur()
        return [lr * self.gamma for lr in self._cur()]

class CosineAnnealingLR(LRScheduler):
    T_max: int
    eta_min: float

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        self._setup(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self._is_initial:
            return self._cur()
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min + (b - self.eta_min) * (1 + math.cos(self.last_epoch * math.pi / self.T_max)) / 2 for b in self.base_lrs]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            cur = self._cur()
            return [cur[i] + (self.base_lrs[i] - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2 for i in range(len(cur))]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
                * (lr - self.eta_min) + self.eta_min for lr in self._cur()]

class LambdaLR(LRScheduler):
    lr_lambda: Function[[int], float]

    def __init__(self, optimizer: Optimizer, lr_lambda: Function[[int], float], last_epoch: int = -1):
        self.lr_lambda = lr_lambda
        self._setup(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [b * self.lr_lambda(self.last_epoch) for b in self.base_lrs]
