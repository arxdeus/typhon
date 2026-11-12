# Ported from torch 2.15.0 (BSD-3-Clause)

from torch._tensor import Tensor, no_grad

def _as_list(parameters) -> List[Tensor]:
    if isinstance(parameters, Tensor):
        return [parameters]
    else:
        out = List[Tensor]()
        for p in parameters:
            out.append(p)
        return out

def _grads(params: List[Tensor]) -> List[Tensor]:
    out = List[Tensor]()
    for p in params:
        if p.grad is not None:
            out.append(p.grad)
    return out

def get_total_norm(tensors, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach = None) -> Tensor:
    from torch import stack, tensor, linalg
    ts = _as_list(tensors)
    if len(ts) == 0:
        return tensor(0.0)
    with no_grad():
        norms = [linalg.vector_norm(g, norm_type) for g in ts]
        total = linalg.vector_norm(stack(norms), norm_type)
    if error_if_nonfinite and (bool(total.isnan()) or bool(total.isinf())):
        raise RuntimeError(f"The total norm of order {norm_type} for gradients from `parameters` is non-finite, so it cannot be clipped.")
    return total

def clip_grads_with_norm_(parameters, max_norm: float, total_norm: Tensor, foreach = None):
    grads = _grads(_as_list(parameters))
    if len(grads) == 0:
        return
    with no_grad():
        clip_coef = (total_norm + 1e-6).reciprocal() * max_norm
        clip_coef_clamped = clip_coef.clamp(None, 1.0)
        for g in grads:
            g.mul_(clip_coef_clamped.to(g.dtype))

def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach = None) -> Tensor:
    params = _as_list(parameters)
    grads = _grads(params)
    total = get_total_norm(grads, norm_type, error_if_nonfinite)
    clip_grads_with_norm_(params, max_norm, total)
    return total

def clip_grad_value_(parameters, clip_value: float, foreach = None):
    with no_grad():
        for g in _grads(_as_list(parameters)):
            g.clamp_(-clip_value, clip_value)
