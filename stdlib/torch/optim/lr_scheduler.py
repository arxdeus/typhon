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

class ReduceLROnPlateau:
    """Reduce the learning rate when a metric has stopped improving."""
    optimizer: Optimizer
    factor: float
    min_lrs: List[float]
    default_min_lr: float
    patience: int
    cooldown: int
    cooldown_counter: int
    eps: float
    last_epoch: int
    best: float
    num_bad_epochs: int
    mode: str
    threshold: float
    threshold_mode: str
    mode_worse: float
    _last_lr: List[float]

    def __init__(self, optimizer: Optimizer, mode: str = "min", factor: float = 0.1,
                 patience: int = 10, threshold: float = 1e-4, threshold_mode: str = "rel",
                 cooldown: int = 0, min_lr = 0, eps: float = 1e-8):
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor
        self.optimizer = optimizer
        ng = len(optimizer.param_groups)
        if isinstance(min_lr, List) or isinstance(min_lr, Tuple):
            if len(min_lr) != ng:
                raise ValueError(f"expected {ng} min_lrs, got {len(min_lr)}")
            self.default_min_lr = -1.0
            self.min_lrs = [float(v) for v in min_lr]
        else:
            self.default_min_lr = float(min_lr)
            self.min_lrs = [float(min_lr)] * ng
        self.patience = patience
        self.cooldown = cooldown
        self.eps = eps
        self.last_epoch = 0
        self._last_lr = [g.lr for g in optimizer.param_groups]
        if mode != "min" and mode != "max":
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode != "rel" and threshold_mode != "abs":
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")
        self.mode_worse = math.inf if mode == "min" else -math.inf
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self._reset()

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    @property
    def in_cooldown(self) -> bool:
        return self.cooldown_counter > 0

    def _is_better(self, a: float, best: float) -> bool:
        if self.mode == "min" and self.threshold_mode == "rel":
            return a < best * (1.0 - self.threshold)
        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            return a > best * (self.threshold + 1.0)
        else:
            return a > best + self.threshold

    def step(self, metrics, epoch: Optional[int] = None):
        current = float(metrics)
        self.last_epoch = epoch if epoch is not None else self.last_epoch + 1
        if self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        self._last_lr = [g.lr for g in self.optimizer.param_groups]

    def _reduce_lr(self):
        ng = len(self.optimizer.param_groups)
        if ng != len(self.min_lrs):
            if self.default_min_lr < 0.0:
                raise RuntimeError("The number of param groups in the `optimizer` differs from when `ReduceLROnPlateau` was initialized")
            self.min_lrs = [self.default_min_lr] * ng
        for i in range(ng):
            g = self.optimizer.param_groups[i]
            old_lr = g.lr
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                g.lr = new_lr

    def get_last_lr(self) -> List[float]:
        return self._last_lr
