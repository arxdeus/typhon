# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Optimizer base. Param groups are typed (ParamGroup) instead of dicts;
# `optimizer.param_groups[i].lr` replaces `group["lr"]` (both spellings are
# supported: ParamGroup implements __getitem__/__setitem__ for float keys).
# Per-parameter state lives in parallel lists indexed like the parameters.

from torch._tensor import Tensor, no_grad

class ParamGroup:
    params: List[Tensor]
    lr: float
    initial_lr: float
    momentum: float
    dampening: float
    weight_decay: float
    nesterov: bool
    betas: Tuple[float, float]
    eps: float
    amsgrad: bool
    maximize: bool

    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr
        self.initial_lr = lr
        self.momentum = 0.0
        self.dampening = 0.0
        self.weight_decay = 0.0
        self.nesterov = False
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.amsgrad = False
        self.maximize = False

    def __getitem__(self, key: str) -> float:
        if key == "lr": return self.lr
        if key == "initial_lr": return self.initial_lr
        if key == "momentum": return self.momentum
        if key == "dampening": return self.dampening
        if key == "weight_decay": return self.weight_decay
        if key == "eps": return self.eps
        raise KeyError(key)

    def __setitem__(self, key: str, v: float):
        if key == "lr": self.lr = v
        elif key == "initial_lr": self.initial_lr = v
        elif key == "momentum": self.momentum = v
        elif key == "dampening": self.dampening = v
        elif key == "weight_decay": self.weight_decay = v
        elif key == "eps": self.eps = v
        else: raise KeyError(key)

def _param_list(params) -> List[Tensor]:
    out = List[Tensor]()
    if isinstance(params, Tensor):
        out.append(params)
    else:
        for p in params:
            out.append(p)
    return out

class Optimizer:
    param_groups: List[ParamGroup]
    _step_count: int

    def _init_groups(self, params, lr: float):
        self.param_groups = []
        self._step_count = 0
        if isinstance(params, List[ParamGroup]):
            for g in params:
                self.param_groups.append(g)
        else:
            pl = _param_list(params)
            if len(pl) == 0:
                raise ValueError("optimizer got an empty parameter list")
            self.param_groups.append(ParamGroup(pl, lr))

    def zero_grad(self, set_to_none: bool = True):
        for g in self.param_groups:
            for p in g.params:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        gr = p.grad
                        gr.detach_()
                        gr.zero_()

    def step(self):
        raise NotImplementedError("step")

    def _all_params(self) -> List[Tensor]:
        out = List[Tensor]()
        for g in self.param_groups:
            out.extend(g.params)
        return out
