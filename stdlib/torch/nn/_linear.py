# Copyright (C) 2026 the Typhon authors
# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Bit-exact float32 F.linear for 2-D inputs as ATen on CPU/BLAS:
#   forward : addmm(bias, x, W^T) -> out filled with bias, then
#             gemm(T, N, O, N, K, W, K, x, K, beta=1, out, O)          (column-major)
#   backward: grad_x = mm(g, W)   -> gemm(N, N, K, N, O, W, K, g, O, 0, gx, K)
#             grad_W = mm(g^T, x) -> gemm(N, T, K, O, N, x, K, g, O, 0, gw, K)
#             grad_b = g.sum(0)   (SumKernel vectorized_outer_sum)

from torch._tensor import Tensor, _NodeBase, _new, _F32, _outer_sum_f32
from numpy.linalg.blas import cblas_sgemm, fint

def _sg(ta: bool, tb: bool, M: int, N: int, K: int, A: Ptr[float32], lda: int, B: Ptr[float32], ldb: int,
        beta: float32, C: Ptr[float32], ldc: int):
    if M == 0 or N == 0:
        return
    if K == 0:
        if beta == float32(0.0):
            for j in range(N):
                for i in range(M):
                    C[j * ldc + i] = float32(0.0)
        return
    cblas_sgemm(fint(102), fint(112 if ta else 111), fint(112 if tb else 111), fint(M), fint(N), fint(K),
                float32(1.0), A, fint(lda), B, fint(ldb), beta, C, fint(ldc))

def _p(t: Tensor) -> Ptr[float32]:
    return Ptr[float32](t._st.ptr) + t._off

class _LinearNode(_NodeBase[Tensor]):
    has_b: bool

    def __init__(self, x: Tensor, w: Tensor, b: Optional[Tensor]):
        ins = [x, w]
        if b is not None:
            ins.append(b)
        super().__init__("AddmmBackward0", ins)
        self.has_b = b is not None

    def apply(self, grad: Tensor) -> List[Optional[Tensor]]:
        x = self.inputs[0].detach().contiguous()
        w = self.inputs[1].detach().contiguous()
        g = grad.detach().contiguous()
        N = x._shape[0]
        K = x._shape[1]
        O = w._shape[0]
        res = List[Optional[Tensor]]()
        if self.inputs[0].requires_grad:
            gx = _new([N, K], _F32)
            _sg(False, False, K, N, O, _p(w), K, _p(g), O, float32(0.0), _p(gx), K)
            res.append(gx)
        else:
            res.append(None)
        if self.inputs[1].requires_grad:
            gw = _new([O, K], _F32)
            _sg(False, True, K, O, N, _p(x), K, _p(g), O, float32(0.0), _p(gw), K)
            res.append(gw)
        else:
            res.append(None)
        if self.has_b:
            if self.inputs[2].requires_grad:
                gb = _new([O], _F32)
                _outer_sum_f32(_p(g), N, O, _p(gb))
                res.append(gb)
            else:
                res.append(None)
        return res

def linear_f32(input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
    from torch.nn._ops import _attach, _grad_on
    x = input.detach().contiguous()
    w = weight.detach().contiguous()
    N = x._shape[0]
    K = x._shape[1]
    O = w._shape[0]
    out = _new([N, O], _F32)
    po = _p(out)
    beta = float32(0.0)
    if bias is not None:
        b = bias.detach().contiguous()
        pb = _p(b)
        for n in range(N):
            for o in range(O):
                po[n * O + o] = pb[o]
        beta = float32(1.0)
    _sg(True, False, O, N, K, _p(w), K, _p(x), K, beta, po, O)
    if not (_grad_on(input) or _grad_on(weight) or (bias is not None and _grad_on(bias.__val__()))):
        return out
    return _attach(out, _LinearNode(input, weight, bias))
