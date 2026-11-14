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

def _norm2_f32(p: Ptr[float32], n: int) -> float32:
    # ATen norm_kernel (p=2, last-dim reduce, aarch64 kAccVecs=2): 8 float lanes of
    # acc += x*x (separate rounding), lanes summed in order, then the n % 8 tail; clang
    # vectorizes the first 4 tail elements when >= 4 remain (separate mul/add) and
    # contracts the rest to fma. CPU _foreach_norm gives identical results.
    from C import sqrtf(float32) -> float32
    acc = Ptr[float32](8)
    for l in range(8):
        acc[l] = float32(0.0)
    d = 0
    lim = n - (n % 8)
    while d < lim:
        for l in range(8):
            v = p[d + l]
            acc[l] = acc[l] + v * v
        d += 8
    s = acc[0]
    for l in range(1, 8):
        s = s + acc[l]
    r = n - d
    for i in range(r):
        v = p[d + i]
        if r >= 4 and i < 4:
            s = s + v * v
        else:
            s = float32._fma(v, v, s)
    return sqrtf(s)

def _all_f32_contig(ts: List[Tensor]) -> bool:
    from torch._tensor import _F32
    for g in ts:
        if g._dt != _F32 or not g.is_contiguous():
            return False
    return True

def get_total_norm(tensors, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach = None) -> Tensor:
    from torch import stack, tensor, linalg
    from torch._tensor import _new, _F32
    ts = _as_list(tensors)
    if len(ts) == 0:
        return tensor(0.0)
    total: Tensor
    if float(norm_type) == 2.0 and _all_f32_contig(ts):
        nb = Ptr[float32](len(ts))
        for i in range(len(ts)):
            g = ts[i]
            nb[i] = _norm2_f32(Ptr[float32](g._st.ptr) + g._off, g.numel())
        total = _new(List[int](), _F32)
        Ptr[float32](total._st.ptr)[0] = _norm2_f32(nb, len(ts))
    else:
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
    from torch._tensor import _F32
    if total_norm._dt == _F32 and total_norm.numel() == 1 and _all_f32_contig(grads):
        # torch: clip_coef = max_norm / (total + 1e-6); clamp(max=1); _foreach_mul_
        tv = (Ptr[float32](total_norm._st.ptr) + total_norm._off)[0]
        cc = float32(max_norm) / (tv + float32(1e-6))
        if cc > float32(1.0):
            cc = float32(1.0)
        for g in grads:
            pg = Ptr[float32](g._st.ptr) + g._off
            for i in range(g.numel()):
                pg[i] = pg[i] * cc
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
