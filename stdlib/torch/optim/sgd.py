# Ported from torch 2.15.0 (BSD-3-Clause)

from torch._tensor import Tensor, no_grad
from .optimizer import Optimizer, ParamGroup

class SGD(Optimizer):
    momentum_buffer: Dict[int, Tensor]

    def __init__(self, params, lr: float = 0.001, momentum: float = 0.0, dampening: float = 0.0, weight_decay: float = 0.0,
                 nesterov: bool = False, maximize: bool = False, foreach = None, differentiable: bool = False, fused = None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self._init_groups(params, lr)
        for g in self.param_groups:
            g.momentum = momentum
            g.dampening = dampening
            g.weight_decay = weight_decay
            g.nesterov = nesterov
            g.maximize = maximize
        self.momentum_buffer = {}

    def step(self):
        with no_grad():
            for g in self.param_groups:
                for p in g.params:
                    if p.grad is None:
                        continue
                    grad = p.grad.__val__()
                    if g.maximize:
                        grad = -grad
                    if g.weight_decay != 0:
                        grad = grad.add(p, alpha=g.weight_decay)
                    if g.momentum != 0:
                        k = id(p)
                        if k not in self.momentum_buffer:
                            buf = grad.detach().clone()
                            self.momentum_buffer[k] = buf
                        else:
                            buf = self.momentum_buffer[k]
                            buf.mul_(g.momentum).add_(grad, alpha=1 - g.dampening)
                        if g.nesterov:
                            grad = grad.add(self.momentum_buffer[k], alpha=g.momentum)
                        else:
                            grad = self.momentum_buffer[k]
                    p.add_(grad, alpha=-g.lr)
        self._step_count += 1
