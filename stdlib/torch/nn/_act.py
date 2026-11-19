# Copyright (C) 2026 the Typhon authors
# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Bit-exact float32 ELU forward/backward as ATen's CPU kernels on aarch64
# (native/cpu/Elu.h get_vectorized_elu_elementwise_func, Activation.cpp
# elu_backward_kernel with is_result=False): 4-lane Vectorized<float> with the
# SLEEF expm1f/expf u10 kernels; scalar tail elements use std::expm1 (libm).

from torch._tensor import Tensor, _NodeBase, _new, _F32
from torch._sleef import expf_block as _expf_block, expm1f_v
from simd import Vec

@pure
@llvm
def _vsel_ge0(v: Vec[float32, 4], neg: Vec[float32, 4]) -> Vec[float32, 4]:
    # v >= 0 ? v : neg   (Elu.h blendv on cmp_ge)
    %c = fcmp oge <4 x float> %v, zeroinitializer
    %r = select <4 x i1> %c, <4 x float> %v, <4 x float> %neg
    ret <4 x float> %r

_GRAIN = 32768   # at::internal::GRAIN_SIZE

def _pieces(n: int) -> List[Tuple[int, int]]:
    # ATen cpu_kernel_vec runs TensorIterator::for_each -> at::parallel_for(0, n, GRAIN_SIZE):
    # with T = torch.get_num_threads(), nt = min(T, ceil(n / grain)) chunks of ceil(n / nt);
    # every chunk has its own vector part (8-element steps) and scalar tail, so results depend on T.
    # Each chunk's vector part is further split for our workers at 8-element offsets from
    # the chunk start, so the worker count never changes results.
    from torch._misc import get_num_threads, _num_workers
    T = get_num_threads()
    nt = 1
    if n >= _GRAIN and T > 1:
        nt = (n + _GRAIN - 1) // _GRAIN
        if nt > T:
            nt = T
    cs = (n + nt - 1) // nt
    nw = _num_workers()
    sub = n // (_GRAIN * nt)   # worker pieces per chunk (>= 32768 elements each)
    if sub > nw:
        sub = nw
    if sub < 1:
        sub = 1
    r = List[Tuple[int, int]]()
    s = 0
    while s < n:
        e = min(s + cs, n)
        full = (e - s) - (e - s) % 8
        per = ((full // 8 + sub - 1) // sub) * 8
        q = 0
        while q < full:
            m = min(per, full - q)
            r.append((s + q, m))
            q += m
        if full < e - s:
            r.append((s + full, e - s - full))
        s = e
    return r

def _elu_fwd(x: Ptr[float32], y: Ptr[float32], n: int, alpha: float32):
    ps = _pieces(n)
    if len(ps) == 1:
        _elu_fwd1(x, y, n, alpha)
        return
    from torch._misc import _num_workers
    @par(num_threads=_num_workers(), schedule='dynamic', chunk_size=1)
    for k in range(len(ps)):
        s, m = ps[k]
        _elu_fwd1(x + s, y + s, m, alpha)

def _elu_bwd(g: Ptr[float32], x: Ptr[float32], gx: Ptr[float32], n: int, alpha: float32):
    ps = _pieces(n)
    if len(ps) == 1:
        _elu_bwd1(g, x, gx, n, alpha)
        return
    from torch._misc import _num_workers
    @par(num_threads=_num_workers(), schedule='dynamic', chunk_size=1)
    for k in range(len(ps)):
        s, m = ps[k]
        _elu_bwd1(g + s, x + s, gx + s, m, alpha)

def _elu_fwd1(x: Ptr[float32], y: Ptr[float32], n: int, alpha: float32):
    # cpu_kernel_vec: vectorized_loop steps 2 x 4 lanes, so the scalar tail is n % 8;
    # the vector op computes expm1 only if any lane is negative (same values per lane).
    from C import expm1f(float32) -> float32
    full = n - (n % 8)
    if __aarch64__:
        # native SLEEF expm1f (libtyphonrt), lane values identical to the Vec port
        # SLEEF only on runs of 4-lane blocks with a lane not >= 0 (lanes are
        # independent, so skipping all-positive blocks keeps every bit).
        from torch._sleef import expm1f_block
        av = Vec[float32, 4](alpha)
        zero = Vec[float32, 4](float32(0.0))
        i = 0
        while i < full:
            if (Vec[float32, 4].load(x + i) >= zero).all():
                Vec[float32, 4].load(x + i).store(y + i)
                i += 4
                continue
            j = i + 4
            while j < full and not (Vec[float32, 4].load(x + j) >= zero).all():
                j += 4
            expm1f_block(x + i, y + i, j - i)
            for k in range(i, j, 4):
                _vsel_ge0(Vec[float32, 4].load(x + k), Vec[float32, 4].load(y + k) * av).store(y + k)
            i = j
        for i in range(full, n):
            v = x[i]
            y[i] = expm1f(v) * alpha if v < float32(0.0) else v
        return
    av = Vec[float32, 4](alpha)
    zero = Vec[float32, 4](float32(0.0))
    for i in range(0, full, 4):
        v = Vec[float32, 4].load(x + i)
        if (v < zero).any():
            _vsel_ge0(v, expm1f_v(v) * av).store(y + i)
        else:
            v.store(y + i)
    for i in range(full, n):
        v = x[i]
        y[i] = expm1f(v) * alpha if v < float32(0.0) else v

def _elu_bwd1(g: Ptr[float32], x: Ptr[float32], gx: Ptr[float32], n: int, alpha: float32):
    # vector: blendv(a * negiptcoef * negcoef * exp(b * negiptcoef), a * poscoef, b > 0)
    # scalar: b <= 0 ? a * negiptcoef * negcoef * exp(b * negiptcoef) : a * poscoef
    from C import expf(float32) -> float32
    # vectorized_loop steps 2 x 4 lanes: scalar tail is n % 8
    full = _expf_block(x, gx, n - (n % 8))    # gx[i] = sleef expf(x[i] * 1) on full blocks
    one = float32(1.0)
    for i in range(full):
        b = x[i]
        a = g[i]
        gx[i] = a * one if b > float32(0.0) else ((a * one) * alpha) * gx[i]
    for i in range(full, n):
        b = x[i]
        a = g[i]
        gx[i] = ((a * one) * alpha) * expf(b * one) if b <= float32(0.0) else a * one

class _EluNode(_NodeBase[Tensor]):
    alpha: float

    def __init__(self, x: Tensor, alpha: float):
        super().__init__("EluBackward0", [x])
        self.alpha = alpha

    def apply(self, grad: Tensor) -> List[Optional[Tensor]]:
        x = self.inputs[0].detach().contiguous()
        gc = grad.detach().contiguous()
        out = _new(x._shape.copy(), _F32)
        _elu_bwd(Ptr[float32](gc._st.ptr) + gc._off, Ptr[float32](x._st.ptr) + x._off,
                 Ptr[float32](out._st.ptr), x.numel(), float32(self.alpha))
        r = List[Optional[Tensor]]()
        r.append(out)
        return r

def elu_f32(input: Tensor, alpha: float) -> Tensor:
    from torch.nn._ops import _attach, _grad_on
    x = input.detach().contiguous()
    out = _new(x._shape.copy(), _F32)
    _elu_fwd(Ptr[float32](x._st.ptr) + x._off, Ptr[float32](out._st.ptr), x.numel(), float32(alpha))
    if not _grad_on(input):
        return out
    return _attach(out, _EluNode(input, alpha))
