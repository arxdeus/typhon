# Copyright (C) 2026 the Typhon authors
# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Bit-exact float32 cross_entropy (2-D logits, class-index targets, no label smoothing)
# as ATen's CPU kernels on aarch64:
#   log_softmax  : serial_vec_log_softmax_lastdim_range (max via reduce_all, sum of
#                  Sleef exp(x - max) via map_reduce_all, Sleef log, x - max - logsum)
#   nll_loss     : nll_loss_out_frame (8-level cascade partial sums, weighted mean)
#   backward     : nll_loss_backward_out_frame (-w[t] * grad/total_weight at target),
#                  then _vec_host_softmax_backward_lastdim<log_softmax>
#                  (gx = g - exp(out) * sum(g); Sleef exp, reduce_all sum)

from torch._tensor import Tensor, _NodeBase, _new, _F32, _I64
from torch._sleef import expf as _sexp, logf as _slog

@inline
def _vred(a: Ptr[float32]) -> float32:
    return (a[0] + a[2]) + (a[1] + a[3])

def _vred_max(a: Ptr[float32]) -> float32:
    m01 = a[0] if a[0] > a[2] else a[2]
    m13 = a[1] if a[1] > a[3] else a[3]
    if a[0] != a[0] or a[2] != a[2]:
        m01 = a[0] + a[2]
    if a[1] != a[1] or a[3] != a[3]:
        m13 = a[1] + a[3]
    r = m01 if m01 > m13 else m13
    if m01 != m01 or m13 != m13:
        r = m01 + m13
    return r

def _max_all(p: Ptr[float32], n: int, acc: Ptr[float32]) -> float32:
    # vec::reduce_all(maximum): lanes beyond n are 0 for n < 4 (loadu pads with 0)
    if n < 4:
        for l in range(4):
            acc[l] = p[l] if l < n else float32(0.0)
        # vec_reduce_all(vec_fun, v, size) for size < 4: sequential over first `size` lanes
        r = acc[0]
        for l in range(1, n):
            r = r if r > acc[l] else acc[l]
        return r
    for l in range(4):
        acc[l] = p[l]
    d = 4
    lim = n - (n % 4)
    while d < lim:
        for l in range(4):
            v = p[d + l]
            acc[l] = acc[l] if acc[l] > v else v
        d += 4
    for l in range(n - d):
        v = p[d + l]
        acc[l] = acc[l] if acc[l] > v else v
    return _vred_max(acc)

def _sumexp(p: Ptr[float32], n: int, mx: float32, acc: Ptr[float32]) -> float32:
    # vec::map_reduce_all((x - max).exp(), +)
    if n < 4:
        for l in range(4):
            acc[l] = _sexp((p[l] if l < n else float32(0.0)) - mx)
        r = acc[0]
        for l in range(1, n):
            r = r + acc[l]
        return r
    for l in range(4):
        acc[l] = _sexp(p[l] - mx)
    d = 4
    lim = n - (n % 4)
    while d < lim:
        for l in range(4):
            acc[l] = acc[l] + _sexp(p[d + l] - mx)
        d += 4
    for l in range(n - d):
        acc[l] = acc[l] + _sexp(p[d + l] - mx)
    return _vred(acc)

def _sum_all(p: Ptr[float32], n: int, acc: Ptr[float32]) -> float32:
    if n < 4:
        r = p[0]
        for l in range(1, n):
            r = r + p[l]
        return r
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

def log_softmax_lastdim_f32(x: Ptr[float32], out: Ptr[float32], N: int, C: int):
    acc = Ptr[float32](4)
    for i in range(N):
        row = x + i * C
        mx = _max_all(row, C, acc)
        ls = _slog(_sumexp(row, C, mx, acc))
        o = out + i * C
        for j in range(C):
            o[j] = (row[j] - mx) - ls

class _CENode(_NodeBase[Tensor]):
    reduction: int      # 0 none, 1 mean, 2 sum
    has_w: bool
    total_w: float32

    def __init__(self, x: Tensor, logp: Tensor, target: Tensor, weight: Optional[Tensor], reduction: int, total_w: float32):
        super().__init__("LogSoftmaxBackward0", [x])
        self.saved = [logp, target]
        if weight is not None:
            self.saved.append(weight)
        self.has_w = weight is not None
        self.reduction = reduction
        self.total_w = total_w

    def apply(self, grad: Tensor) -> List[Optional[Tensor]]:
        logp = self.saved[0]
        tg = self.saved[1]
        N = logp._shape[0]
        C = logp._shape[1]
        g = grad.detach().contiguous()
        pg = Ptr[float32](g._st.ptr) + g._off
        pt = Ptr[int](tg._st.ptr) + tg._off
        pw = Ptr[float32]()
        if self.has_w:
            pw = Ptr[float32](self.saved[2]._st.ptr) + self.saved[2]._off
        # nll backward -> d logp
        dl = _new([N, C], _F32)
        pd = Ptr[float32](dl._st.ptr)
        for k in range(N * C):
            pd[k] = float32(0.0)
        if self.reduction == 0:
            for i in range(N):
                t = pt[i]
                if t == -100:
                    continue
                w = pw[t] if self.has_w else float32(1.0)
                pd[i * C + t] = -w * pg[i]
        else:
            go = pg[0]
            gr = -(go / self.total_w if self.reduction == 1 else go)
            for i in range(N):
                t = pt[i]
                if t == -100:
                    continue
                pd[i * C + t] = pw[t] * gr if self.has_w else gr
        # log_softmax backward
        gx = _new([N, C], _F32)
        px = Ptr[float32](gx._st.ptr)
        pl = Ptr[float32](logp._st.ptr) + logp._off
        acc = Ptr[float32](4)
        for i in range(N):
            s = _sum_all(pd + i * C, C, acc)
            for j in range(C):
                px[i * C + j] = pd[i * C + j] - _sexp(pl[i * C + j]) * s
        r = List[Optional[Tensor]]()
        r.append(gx)
        return r

def cross_entropy_f32(input: Tensor, target: Tensor, weight: Optional[Tensor], ignore_index: int, reduction: str) -> Tensor:
    from torch.nn._ops import _attach
    x = input.detach().contiguous()
    N = x._shape[0]
    C = x._shape[1]
    logp = _new([N, C], _F32)
    log_softmax_lastdim_f32(Ptr[float32](x._st.ptr) + x._off, Ptr[float32](logp._st.ptr), N, C)
    tg = target.detach().contiguous()
    pt = Ptr[int](tg._st.ptr) + tg._off
    pl = Ptr[float32](logp._st.ptr)
    w: Optional[Tensor] = None
    pw = Ptr[float32]()
    if weight is not None:
        wv = weight.detach().contiguous()
        w = wv
        pw = Ptr[float32](wv._st.ptr) + wv._off
    red = 0 if reduction == "none" else (1 if reduction == "mean" else 2)
    total_w = float32(0.0)
    out: Tensor
    if red == 0:
        out = _new([N], _F32)
        po = Ptr[float32](out._st.ptr)
        for i in range(N):
            t = pt[i]
            if t == ignore_index:
                po[i] = float32(0.0)
                continue
            if t < 0 or t >= C:
                raise IndexError(f"Target {t} is out of bounds.")
            cw = pw[t] if weight is not None else float32(1.0)
            po[i] = -pl[i * C + t] * cw
    else:
        L = 8
        lp = 4
        if N > 1:
            cl = 0
            while (1 << cl) < N:
                cl += 1
            if cl // L > lp:
                lp = cl // L
        mask = (1 << lp) - 1
        wps = Ptr[float32](L)
        lps = Ptr[float32](L)
        for k in range(L):
            wps[k] = float32(0.0)
            lps[k] = float32(0.0)
        ignored = 0
        for b in range(N):
            t = pt[b]
            if t == ignore_index:
                ignored += 1
                continue
            if t < 0 or t >= C:
                raise IndexError(f"Target {t} is out of bounds.")
            dv = pl[b * C + t]
            if weight is not None:
                wv2 = pw[t]
                # loss_partial_sums[0] -= cur_weight * data: clang contracts to fma
                lps[0] = float32._fma(-dv, wv2, lps[0])
                wps[0] = wps[0] + wv2
            else:
                lps[0] = lps[0] - dv
            for j in range(L - 1):
                if (b & (mask << (j * lp))) != 0:
                    break
                wps[j + 1] = wps[j + 1] + wps[j]
                lps[j + 1] = lps[j + 1] + lps[j]
                wps[j] = float32(0.0)
                lps[j] = float32(0.0)
        if weight is None:
            total_w = float32(N - ignored)
        else:
            total_w = float32(0.0)
            for k in range(L):
                total_w = total_w + wps[k]
        ov = float32(0.0)
        for k in range(L):
            ov = ov + lps[k]
        if red == 1:
            ov = ov / total_w
        out = _new(List[int](), _F32)
        Ptr[float32](out._st.ptr)[0] = ov
    return _attach(out, _CENode(input, logp, tg, w, red, total_w))
