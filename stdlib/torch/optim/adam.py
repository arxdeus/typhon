# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Adam / AdamW, single-tensor (CPU default) update path of torch.

import math
from torch._tensor import Tensor, no_grad, _F32
from .optimizer import Optimizer, ParamGroup

def _adam_f32_step(param: Tensor, grad: Tensor, st, lr: float, beta1: float, beta2: float, eps: float, wd: float):
    # torch 2.15 CPU default = _multi_tensor_adam (foreach) with ATen elementwise kernels:
    #   grad = grad + wd * param            (add alpha: fma)
    #   exp_avg.lerp_(grad, 1 - beta1)       (|w| < .5: fma(w, end - self, self))
    #   exp_avg_sq = exp_avg_sq * beta2 + ((1 - beta2) * grad) * grad   (addcmul, separate roundings)
    #   denom = sqrt(exp_avg_sq) / bc2_sqrt + eps
    #   param = param + (step_size * exp_avg) / denom                   (addcdiv)
    p = Ptr[float32](param._st.ptr) + param._off
    gp = Ptr[float32](grad._st.ptr) + grad._off
    ea = Ptr[float32](st.exp_avg._st.ptr) + st.exp_avg._off
    es = Ptr[float32](st.exp_avg_sq._st.ptr) + st.exp_avg_sq._off
    step = st.step
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    ss = float32((lr / bc1) * -1.0)
    bc2s = float32(bc2 ** 0.5)
    w = float32(1.0 - beta1)
    small = abs(1.0 - beta1) < 0.5
    one_m_w = float32(1.0) - w
    b2 = float32(beta2)
    vb2 = float32(1.0 - beta2)
    fwd = float32(wd)
    fe = float32(eps)
    from C import sqrtf(float32) -> float32
    for i in range(param.numel()):
        g = gp[i]
        if wd != 0.0:
            g = float32._fma(fwd, p[i], g)
        a = ea[i]
        if small:
            a = float32._fma(w, g - a, a)
        else:
            a = g - (g - a) * one_m_w
        ea[i] = a
        s = es[i] * b2
        s = s + (vb2 * g) * g
        es[i] = s
        den = sqrtf(s) / bc2s + fe
        p[i] = p[i] + (ss * a) / den

class _AdamState:
    step: float
    exp_avg: Tensor
    exp_avg_sq: Tensor
    max_exp_avg_sq: Optional[Tensor]

    def __init__(self, p: Tensor, amsgrad: bool):
        from torch import zeros_like
        self.step = 0.0
        self.exp_avg = zeros_like(p)
        self.exp_avg_sq = zeros_like(p)
        self.max_exp_avg_sq = zeros_like(p) if amsgrad else None

class _AdamBase(Optimizer):
    state: Dict[int, _AdamState]
    decoupled: bool

    def _setup(self, params, lr: float, betas: Tuple[float, float], eps: float, weight_decay: float, amsgrad: bool, maximize: bool, decoupled: bool):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self._init_groups(params, lr)
        for g in self.param_groups:
            g.betas = betas
            g.eps = eps
            g.weight_decay = weight_decay
            g.amsgrad = amsgrad
            g.maximize = maximize
        self.state = {}
        self.decoupled = decoupled

    def step(self):
        with no_grad():
            for g in self.param_groups:
                beta1, beta2 = g.betas
                lr = g.lr
                for param in g.params:
                    if param.grad is None:
                        continue
                    grad = param.grad.__val__()
                    if g.maximize:
                        grad = -grad
                    k = id(param)
                    if k not in self.state:
                        self.state[k] = _AdamState(param, g.amsgrad)
                    st = self.state[k]
                    st.step += 1.0
                    if (not g.amsgrad and not self.decoupled and param._dt == _F32 and grad._dt == _F32
                            and param.is_contiguous() and grad.is_contiguous()):
                        _adam_f32_step(param, grad, st, lr, beta1, beta2, g.eps, g.weight_decay)
                        continue
                    if g.weight_decay != 0:
                        if self.decoupled:
                            param.mul_(1 - lr * g.weight_decay)
                        else:
                            grad = grad.add(param, alpha=g.weight_decay)
                    st.exp_avg.lerp_(grad, 1 - beta1)
                    st.exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    step = st.step
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    step_size = lr / bias_correction1
                    bias_correction2_sqrt = bias_correction2 ** 0.5
                    denom: Tensor
                    if g.amsgrad:
                        from torch import maximum
                        m = st.max_exp_avg_sq.__val__()
                        m.copy_(maximum(m, st.exp_avg_sq))
                        denom = (m.sqrt() / bias_correction2_sqrt).add_(g.eps)
                    else:
                        denom = (st.exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(g.eps)
                    param.addcdiv_(st.exp_avg, denom, value=-step_size)
        self._step_count += 1

class Adam(_AdamBase):
    _f_Adam: int
    def __init__(self, params, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08,
                 weight_decay: float = 0.0, amsgrad: bool = False, foreach = None, maximize: bool = False,
                 capturable: bool = False, differentiable: bool = False, fused = None, decoupled_weight_decay: bool = False):
        self._f_Adam = 0
        self._setup(params, lr, betas, eps, weight_decay, amsgrad, maximize, decoupled_weight_decay)

class AdamW(_AdamBase):
    _f_AdamW: int
    def __init__(self, params, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08,
                 weight_decay: float = 0.01, amsgrad: bool = False, maximize: bool = False, foreach = None,
                 capturable: bool = False, differentiable: bool = False, fused = None):
        self._f_AdamW = 0
        self._setup(params, lr, betas, eps, weight_decay, amsgrad, maximize, True)
