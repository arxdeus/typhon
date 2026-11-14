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

def _elu_fwd(x: Ptr[float32], y: Ptr[float32], n: int, alpha: float32):
    # cpu_kernel_vec: vectorized loop over full 4-lane blocks, scalar op on the tail;
    # the vector op computes expm1 only if any lane is negative (same values per lane).
    from C import expm1f(float32) -> float32
    full = n - (n % 4)
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

def _elu_bwd(g: Ptr[float32], x: Ptr[float32], gx: Ptr[float32], n: int, alpha: float32):
    # vector: blendv(a * negiptcoef * negcoef * exp(b * negiptcoef), a * poscoef, b > 0)
    # scalar: b <= 0 ? a * negiptcoef * negcoef * exp(b * negiptcoef) : a * poscoef
    from C import expf(float32) -> float32
    full = _expf_block(x, gx, n)    # gx[i] = sleef expf(x[i] * 1) on full blocks
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
    from torch.nn._ops import _attach
    x = input.detach().contiguous()
    out = _new(x._shape.copy(), _F32)
    _elu_fwd(Ptr[float32](x._st.ptr) + x._off, Ptr[float32](out._st.ptr), x.numel(), float32(alpha))
    return _attach(out, _EluNode(input, alpha))
