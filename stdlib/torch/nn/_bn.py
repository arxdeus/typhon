# Copyright (C) 2026 the Typhon authors
# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Bit-exact float32 batch_norm in training mode, as ATen's CPU kernels on aarch64
# (aten/src/ATen/native/Normalization.cpp, native/cpu/batch_norm_kernel.cpp):
#   forward : collect_stats (contiguous: per-channel double sums; image_size == 1:
#             channels-last per-thread buffers), save_invstd = 1/sqrt(var_sum/n + eps)
#             in double, running stats updated with momentum, output = x*alpha + beta
#             with alpha = invstd*w and beta = fma(-mean, alpha, b) (clang contraction).
#   backward: native_batch_norm_backward (contiguous: per-sample 4-lane vec::reduce_all
#             sums accumulated in double; image_size == 1: per-thread buffers).
# The channels-last paths depend on at::get_num_threads() (chunked parallel_for);
# torch.get_num_threads() mirrors torch's default (performance cores) so that
# results equal the reference torch process.

import math as _math
from torch._tensor import Tensor, _NodeBase, _new, _F32
import torch._tensor as _T

def _threads() -> int:
    from torch._misc import get_num_threads
    return get_num_threads()

def _chunk(N: int, T: int) -> int:
    # at::parallel_for(0, N, grain 1) with OpenMP: ceil(N / min(T, N)) per thread
    t = T if T < N else N
    return (N + t - 1) // t

def _vred(a: Ptr[float32]) -> float32:
    # VecReduceAllSIMD<float> on aarch64: v + vext(v, 2), then + vrev64
    return (a[0] + a[2]) + (a[1] + a[3])

def _red4(p: Ptr[float32], n: int, acc: Ptr[float32]) -> float32:
    # vec::reduce_all<float>(plus) with Vectorized<float>::size() == 4
    if n < 4:
        for l in range(4):
            acc[l] = p[l] if l < n else float32(0.0)
        return _vred(acc)
    for l in range(4):
        acc[l] = p[l]
    d = 4
    lim = n - (n % 4)
    while d < lim:
        for l in range(4):
            acc[l] = acc[l] + p[d + l]
        d += 4
    for l in range(n - d):
        acc[l] = acc[l] + p[d + l]
    return _vred(acc)

def _red4_dot(x: Ptr[float32], dy: Ptr[float32], mean: float32, n: int, acc: Ptr[float32]) -> float32:
    # vec::map2_reduce_all: map (x - mean) * dy, reduce plus
    if n < 4:
        for l in range(4):
            acc[l] = (x[l] - mean) * dy[l] if l < n else float32(0.0)
        return _vred(acc)
    for l in range(4):
        acc[l] = (x[l] - mean) * dy[l]
    d = 4
    lim = n - (n % 4)
    while d < lim:
        for l in range(4):
            acc[l] = acc[l] + (x[d + l] - mean) * dy[d + l]
        d += 4
    for l in range(n - d):
        acc[l] = acc[l] + (x[d + l] - mean) * dy[d + l]
    return _vred(acc)

def bn_train_forward(x: Tensor, w: Optional[Tensor], b: Optional[Tensor], rm: Optional[Tensor], rv: Optional[Tensor],
                     momentum: float, eps: float, out: Tensor, save_mean: Tensor, save_invstd: Tensor):
    # x contiguous float32 (N, C, *); out/save_* preallocated float32
    N = x._shape[0]
    C = x._shape[1]
    S = x.numel() // (N * C) if N * C > 0 else 0
    n = N * S
    px = Ptr[float32](x._st.ptr) + x._off
    mean = Ptr[float32](save_mean._st.ptr)
    vsum = Ptr[float32](C)
    if S != 1:
        for c in range(C):
            s = 0.0
            for i in range(N):
                q = px + (i * C + c) * S
                for j in range(S):
                    s += float(q[j])
            m = float32(s / float(n))
            mean[c] = m
            v = 0.0
            for i in range(N):
                q = px + (i * C + c) * S
                for j in range(S):
                    d = q[j] - m
                    v += float(d * d)
            vsum[c] = float32(v)
    else:
        T = _threads()
        if n > T:
            cs = _chunk(n, T)
            buf = Ptr[float32](T * C)
            for k in range(T * C):
                buf[k] = float32(0.0)
            for i in range(n):
                bp = buf + (i // cs) * C
                q = px + i * C
                for c in range(C):
                    bp[c] = q[c] + bp[c]
            for c in range(C):
                s = 0.0
                for t in range(T):
                    s += float(buf[t * C + c])
                mean[c] = float32(s / float(n))
            for k in range(T * C):
                buf[k] = float32(0.0)
            for i in range(n):
                bp = buf + (i // cs) * C
                q = px + i * C
                for c in range(C):
                    d = q[c] - mean[c]
                    bp[c] = bp[c] + d * d
            for c in range(C):
                s = 0.0
                for t in range(T):
                    s += float(buf[t * C + c])
                vsum[c] = float32(s)
        else:
            # TILE_SIZE path (num_threads == 1 or C <= 16 or C > 2048) or per-channel double sums
            if T == 1 or C <= 16 or C > 2048:
                for c in range(C):
                    mean[c] = float32(0.0)
                for i in range(n):
                    for c in range(C):
                        mean[c] = px[i * C + c] + mean[c]
                for c in range(C):
                    mean[c] = mean[c] / float32(n)
                for c in range(C):
                    vsum[c] = float32(0.0)
                for i in range(n):
                    for c in range(C):
                        d = px[i * C + c] - mean[c]
                        vsum[c] = vsum[c] + d * d
            else:
                for c in range(C):
                    s = 0.0
                    for i in range(n):
                        s += float(px[i * C + c])
                    mean[c] = float32(s / float(n))
                for c in range(C):
                    s = 0.0
                    for i in range(n):
                        d = px[i * C + c] - mean[c]
                        s += float(d * d)
                    vsum[c] = float32(s)
    inv = Ptr[float32](save_invstd._st.ptr)
    mom = float32(momentum)
    for c in range(C):
        # VarTransform<accscalar_t>{}(_var_sum_a[f] / n, eps): _var_sum_a is float (opmath),
        # so var = float32(var_sum) / n is a float32 division; the sqrt happens in double.
        var = float(vsum[c] / float32(n))
        inv[c] = float32(0.0) if (var == 0.0 and eps == 0.0) else float32(1.0 / _math.sqrt(var + eps))
        if rm is not None:
            r = Ptr[float32](rm._st.ptr) + rm._off
            r[c] = float32._fma(mom, mean[c], (float32(1.0) - mom) * r[c])
        if rv is not None:
            r = Ptr[float32](rv._st.ptr) + rv._off
            unb = vsum[c] / float32(n - 1)
            r[c] = float32._fma(mom, unb, (float32(1.0) - mom) * r[c])
    po = Ptr[float32](out._st.ptr)
    for c in range(C):
        wv = float32(1.0) if w is None else (Ptr[float32](w._st.ptr) + w._off)[c]
        bv = float32(0.0) if b is None else (Ptr[float32](b._st.ptr) + b._off)[c]
        alpha = inv[c] * wv
        beta = float32._fma(-mean[c], alpha, bv)
        for i in range(N):
            q = px + (i * C + c) * S
            o = po + (i * C + c) * S
            for j in range(S):
                o[j] = q[j] * alpha + beta

def bn_train_backward(x: Tensor, dy: Tensor, w: Optional[Tensor], save_mean: Tensor, save_invstd: Tensor,
                      gx: Tensor, gw: Tensor, gb: Tensor, need_x: bool, need_w: bool, need_b: bool):
    N = x._shape[0]
    C = x._shape[1]
    S = x.numel() // (N * C) if N * C > 0 else 0
    n = N * S
    px = Ptr[float32](x._st.ptr) + x._off
    pd = Ptr[float32](dy._st.ptr) + dy._off
    mean = Ptr[float32](save_mean._st.ptr)
    inv = Ptr[float32](save_invstd._st.ptr)
    pgx = Ptr[float32](gx._st.ptr)
    pgw = Ptr[float32](gw._st.ptr)
    pgb = Ptr[float32](gb._st.ptr)
    if S != 1:
        acc = Ptr[float32](4)
        for c in range(C):
            wv = float32(1.0) if w is None else (Ptr[float32](w._st.ptr) + w._off)[c]
            m = mean[c]
            iv = inv[c]
            s = 0.0
            dp = 0.0
            for i in range(N):
                xq = px + (i * C + c) * S
                dq = pd + (i * C + c) * S
                s += float(_red4(dq, S, acc))
                dp += float(_red4_dot(xq, dq, m, S, acc))
            if need_x:
                k = float32(dp) * iv * iv / float32(n)
                gm = float32(s / float(n))
                for i in range(N):
                    xq = px + (i * C + c) * S
                    dq = pd + (i * C + c) * S
                    oq = pgx + (i * C + c) * S
                    for j in range(S):
                        dx = (xq[j] - m) * k
                        oq[j] = (dq[j] - gm - dx) * iv * wv
            if need_w:
                pgw[c] = float32(dp * float(iv))
            if need_b:
                pgb[c] = float32(s)
    else:
        T = _threads()
        cs = _chunk(n, T)
        sb = Ptr[float32](T * C)
        db = Ptr[float32](T * C)
        for k in range(T * C):
            sb[k] = float32(0.0)
            db[k] = float32(0.0)
        for i in range(n):
            t = i // cs
            sp = sb + t * C
            dpp = db + t * C
            xq = px + i * C
            dq = pd + i * C
            for c in range(C):
                sp[c] = sp[c] + dq[c]
                dpp[c] = dpp[c] + (xq[c] - mean[c]) * dq[c]
        ssum = Ptr[float32](C)
        dsum = Ptr[float32](C)
        for c in range(C):
            s = 0.0
            d = 0.0
            for t in range(T):
                s += float(sb[t * C + c])
                d += float(db[t * C + c])
            ssum[c] = float32(s)
            dsum[c] = float32(d)
        if need_x:
            for i in range(n):
                xq = px + i * C
                dq = pd + i * C
                oq = pgx + i * C
                for c in range(C):
                    wv = float32(1.0) if w is None else (Ptr[float32](w._st.ptr) + w._off)[c]
                    iv = inv[c]
                    k = dsum[c] * iv * iv / float32(n)
                    dx = (xq[c] - mean[c]) * k
                    gm = ssum[c] / float32(n)
                    oq[c] = (dq[c] - gm - dx) * iv * wv
        for c in range(C):
            if need_w:
                pgw[c] = dsum[c] * inv[c]
            if need_b:
                pgb[c] = ssum[c]

class _BNTrainNode(_NodeBase[Tensor]):
    has_w: bool
    has_b: bool

    def __init__(self, x: Tensor, w: Optional[Tensor], b: Optional[Tensor], save_mean: Tensor, save_invstd: Tensor):
        ins = [x]
        if w is not None:
            ins.append(w)
        if b is not None:
            ins.append(b)
        super().__init__("NativeBatchNormBackward0", ins)
        self.has_w = w is not None
        self.has_b = b is not None
        self.saved = [save_mean, save_invstd]

    def apply(self, grad: Tensor) -> List[Optional[Tensor]]:
        x = self.inputs[0].detach().contiguous()
        wt: Optional[Tensor] = None
        if self.has_w:
            wt = self.inputs[1].detach().contiguous()
        dy = grad.detach().contiguous()
        C = x._shape[1]
        need_x = self.inputs[0].requires_grad
        need_w = self.has_w and self.inputs[1].requires_grad
        need_b = self.has_b and self.inputs[len(self.inputs) - 1].requires_grad
        gx = _new(x._shape.copy() if need_x else [0], _F32)
        gw = _new([C], _F32)
        gb = _new([C], _F32)
        bn_train_backward(x, dy, wt, self.saved[0], self.saved[1], gx, gw, gb, need_x, need_w, need_b)
        res = List[Optional[Tensor]]()
        res.append(gx if need_x else None)
        if self.has_w:
            res.append(gw if need_w else None)
        if self.has_b:
            res.append(gb if need_b else None)
        return res

def batch_norm_train_f32(input: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor],
                         weight: Optional[Tensor], bias: Optional[Tensor], momentum: float, eps: float) -> Tensor:
    from torch.nn._ops import _attach
    x = input.detach().contiguous()
    C = x._shape[1]
    w: Optional[Tensor] = None
    b: Optional[Tensor] = None
    if weight is not None:
        w = weight.detach().contiguous()
    if bias is not None:
        b = bias.detach().contiguous()
    out = _new(x._shape.copy(), _F32)
    sm = _new([C], _F32)
    si = _new([C], _F32)
    bn_train_forward(x, w, b, running_mean, running_var, momentum, eps, out, sm, si)
    return _attach(out, _BNTrainNode(input, weight, bias, sm, si))
