# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Native kernels used by torch.nn (convolution, pooling, padding) with
# hand-written backward passes, attached to core autograd as custom nodes.
# All kernels operate on contiguous tensors of dtype float32/float64 and
# accumulate in float64.

from torch._tensor import Tensor, _NodeBase, _new, _F32, _F64, _dispatch_float, is_grad_enabled

def _attach(out: Tensor, node: _NodeBase[Tensor]) -> Tensor:
    req = False
    for t in node.inputs:
        if t.requires_grad:
            req = True
    if req and is_grad_enabled():
        out.requires_grad = True
        out.grad_fn = node
    return out

def _grad_on(x: Tensor) -> bool:
    # no_grad / frozen inputs: skip building a backward node (and its saved data)
    return x.requires_grad and is_grad_enabled()

def _zeros(shape: List[int], dt: int) -> Tensor:
    t = _new(shape, dt)
    n = t.numel()
    if dt == _F64:
        p = t._ptr(float)
        for i in range(n):
            p[i] = 0.0
    else:
        p = t._ptr(float32)
        for i in range(n):
            p[i] = float32(0.0)
    return t

def _c(x: Tensor) -> Tensor:
    return x.contiguous()

# ======================================================================
# Convolution (2-D core; 1-D is 2-D with H == 1). Zero padding only:
# other padding modes are applied beforehand via pad().
# ======================================================================

@tuple
class _ConvGeo:
    N: int
    C: int
    H: int
    W: int
    O: int
    KH: int
    KW: int
    OH: int
    OW: int
    sh: int
    sw: int
    ph: int
    pw: int
    dh: int
    dw: int
    groups: int

def _conv_fwd_k(z: T, x: Tensor, w: Tensor, b: Optional[Tensor], g: _ConvGeo, out: Tensor, T: type):
    xp = x._ptr(T)
    wp = w._ptr(T)
    op = out._ptr(T)
    cin_g = g.C // g.groups
    cout_g = g.O // g.groups
    bp = Ptr[T]()
    hasb = b is not None
    if hasb:
        bp = b.__val__()._ptr(T)
    acc = [0.0] * g.OW
    for n in range(g.N):
        for o in range(g.O):
            gi = o // cout_g
            for oy in range(g.OH):
                for ox in range(g.OW):
                    acc[ox] = float(bp[o]) if hasb else 0.0
                for c in range(cin_g):
                    ci = gi * cin_g + c
                    xbase = (n * g.C + ci) * g.H * g.W
                    wbase = (o * cin_g + c) * g.KH * g.KW
                    for ky in range(g.KH):
                        iy = oy * g.sh - g.ph + ky * g.dh
                        if iy < 0 or iy >= g.H:
                            continue
                        xrow = xbase + iy * g.W
                        for kx in range(g.KW):
                            wv = float(wp[wbase + ky * g.KW + kx])
                            off = kx * g.dw - g.pw
                            for ox in range(g.OW):
                                ix = ox * g.sw + off
                                if ix >= 0 and ix < g.W:
                                    acc[ox] += wv * float(xp[xrow + ix])
                obase = ((n * g.O + o) * g.OH + oy) * g.OW
                for ox in range(g.OW):
                    op[obase + ox] = T(acc[ox])

def _conv_bwd_k(z: T, x: Tensor, w: Tensor, gy: Tensor, g: _ConvGeo, gx: Tensor, gw: Tensor, need_x: bool, need_w: bool, T: type):
    xp = x._ptr(T)
    wp = w._ptr(T)
    gp = gy._ptr(T)
    cin_g = g.C // g.groups
    cout_g = g.O // g.groups
    gxa = List[float](0)
    gwa = List[float](0)
    if need_x:
        gxa = [0.0] * (g.N * g.C * g.H * g.W)
    if need_w:
        gwa = [0.0] * (g.O * cin_g * g.KH * g.KW)
    for n in range(g.N):
        for o in range(g.O):
            gi = o // cout_g
            for oy in range(g.OH):
                gbase = ((n * g.O + o) * g.OH + oy) * g.OW
                for c in range(cin_g):
                    ci = gi * cin_g + c
                    xbase = (n * g.C + ci) * g.H * g.W
                    wbase = (o * cin_g + c) * g.KH * g.KW
                    for ky in range(g.KH):
                        iy = oy * g.sh - g.ph + ky * g.dh
                        if iy < 0 or iy >= g.H:
                            continue
                        xrow = xbase + iy * g.W
                        for kx in range(g.KW):
                            wv = float(wp[wbase + ky * g.KW + kx])
                            off = kx * g.dw - g.pw
                            sw = 0.0
                            for ox in range(g.OW):
                                ix = ox * g.sw + off
                                if ix >= 0 and ix < g.W:
                                    gv = float(gp[gbase + ox])
                                    if need_x:
                                        gxa[xrow + ix] += wv * gv
                                    if need_w:
                                        sw += gv * float(xp[xrow + ix])
                            if need_w:
                                gwa[wbase + ky * g.KW + kx] += sw
    if need_x:
        q = gx._ptr(T)
        for i in range(len(gxa)):
            q[i] = T(gxa[i])
    if need_w:
        q = gw._ptr(T)
        for i in range(len(gwa)):
            q[i] = T(gwa[i])

def _bias_grad_k(z: T, gy: Tensor, N: int, O: int, S: int, gb: Tensor, T: type):
    gp = gy._ptr(T)
    q = gb._ptr(T)
    for o in range(O):
        s = 0.0
        for n in range(N):
            base = (n * O + o) * S
            for i in range(S):
                s += float(gp[base + i])
        q[o] = T(s)

class _ConvNode(_NodeBase[Tensor]):
    geo: _ConvGeo
    has_bias: bool
    xshape: List[int]
    wshape: List[int]

    def __init__(self, x: Tensor, w: Tensor, b: Optional[Tensor], geo: _ConvGeo, xshape: List[int], wshape: List[int]):
        ins = [x, w]
        if b is not None:
            ins.append(b.__val__())
        super().__init__("ConvolutionBackward0", ins)
        self.geo = geo
        self.has_bias = b is not None
        self.xshape = xshape
        self.wshape = wshape

    def apply(self, grad: Tensor) -> List[Optional[Tensor]]:
        from torch._conv import conv2d_backward
        g = self.geo
        x = _c(self.inputs[0].detach())
        w = _c(self.inputs[1].detach())
        gy = _c(grad.detach())
        need_x = self.inputs[0].requires_grad
        need_w = self.inputs[1].requires_grad
        gx, gw, gb = conv2d_backward(x, w, gy, (g.sh, g.sw), (g.ph, g.pw), (g.dh, g.dw), g.groups,
                                     need_x, need_w, self.has_bias)
        res = List[Optional[Tensor]]()
        res.append(gx.view(self.xshape) if need_x else None)
        res.append(gw.view(self.wshape) if need_w else None)
        if self.has_bias:
            res.append(gb)
        return res

def conv2d_raw(x: Tensor, w: Tensor, b: Optional[Tensor], stride: Tuple[int, int], padding: Tuple[int, int],
               dilation: Tuple[int, int], groups: int, xshape: List[int], wshape: List[int]) -> Tensor:
    # x: [N, C, H, W]; w: [O, C/groups, KH, KW]
    xs = x._shape
    ws = w._shape
    N, C, H, W = xs[0], xs[1], xs[2], xs[3]
    O, KH, KW = ws[0], ws[2], ws[3]
    if C != ws[1] * groups:
        raise RuntimeError(f"Given groups={groups}, weight of size {Tuple(wshape)}, expected input{Tuple(xshape)} to have {ws[1] * groups} channels, but got {C} channels instead")
    if x._dt != w._dt:
        raise RuntimeError(f"Input type ({x.dtype}) and weight type ({w.dtype}) should be the same")
    OH = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    OW = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1
    if OH <= 0 or OW <= 0:
        raise RuntimeError("Calculated output size is too small")
    geo = _ConvGeo(N, C, H, W, O, KH, KW, OH, OW, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], groups)
    xc = _c(x.detach())
    wc = _c(w.detach())
    bc: Optional[Tensor] = None
    if b is not None:
        bc = _c(b.__val__().detach())
    from torch._conv import conv2d_forward
    out = conv2d_forward(xc, wc, bc, stride, padding, dilation, groups)
    if not (_grad_on(x) or _grad_on(w) or (b is not None and _grad_on(b.__val__()))):
        return out
    return _attach(out, _ConvNode(x, w, b, geo, xshape, wshape))

# ======================================================================
# Pooling (2-D core over the last two dims of a [B, H, W] view)
# ======================================================================

def pool_out_size(L: int, k: int, s: int, p: int, d: int, ceil_mode: bool) -> int:
    num = L + 2 * p - d * (k - 1) - 1
    if ceil_mode:
        o = (num + s - 1) // s + 1
        if (o - 1) * s >= L + p:
            o -= 1
    else:
        o = num // s + 1
    return o

@tuple
class _PoolGeo:
    B: int
    H: int
    W: int
    OH: int
    OW: int
    kh: int
    kw: int
    sh: int
    sw: int
    ph: int
    pw: int
    dh: int
    dw: int

def _maxpool_k(z: T, x: Tensor, g: _PoolGeo, out: Tensor, idx: List[int], T: type):
    xp = x._ptr(T)
    op = out._ptr(T)
    for b in range(g.B):
        base = b * g.H * g.W
        for oy in range(g.OH):
            for ox in range(g.OW):
                best = T(0)
                bi = -1
                for ky in range(g.kh):
                    iy = oy * g.sh - g.ph + ky * g.dh
                    if iy < 0 or iy >= g.H:
                        continue
                    for kx in range(g.kw):
                        ix = ox * g.sw - g.pw + kx * g.dw
                        if ix < 0 or ix >= g.W:
                            continue
                        v = xp[base + iy * g.W + ix]
                        if bi < 0 or v > best or v != v:
                            if not (bi >= 0 and best != best):
                                best = v
                                bi = iy * g.W + ix
                o = (b * g.OH + oy) * g.OW + ox
                op[o] = best
                idx[o] = bi

class _MaxPoolNode(_NodeBase[Tensor]):
    idx: List[int]
    geo: _PoolGeo
    xshape: List[int]

    def __init__(self, x: Tensor, idx: List[int], geo: _PoolGeo, xshape: List[int]):
        super().__init__("MaxPool2DWithIndicesBackward0", [x])
        self.idx = idx
        self.geo = geo
        self.xshape = xshape

    def apply(self, grad: Tensor) -> List[Optional[Tensor]]:
        g = self.geo
        gy = _c(grad.detach())
        gx = _zeros([g.B, g.H, g.W], gy._dt)
        n_out = g.OH * g.OW
        if gy._dt == _F64:
            gp = gy._ptr(float)
            q = gx._ptr(float)
            for b in range(g.B):
                for j in range(n_out):
                    i = self.idx[b * n_out + j]
                    if i >= 0:
                        q[b * g.H * g.W + i] += gp[b * n_out + j]
        else:
            gp = gy._ptr(float32)
            q = gx._ptr(float32)
            for b in range(g.B):
                for j in range(n_out):
                    i = self.idx[b * n_out + j]
                    if i >= 0:
                        q[b * g.H * g.W + i] += gp[b * n_out + j]
        return [Optional(gx.view(self.xshape))]

def max_pool2d_raw(x: Tensor, k: Tuple[int, int], s: Tuple[int, int], p: Tuple[int, int], d: Tuple[int, int],
                   ceil_mode: bool, out_lead: List[int]) -> Tuple[Tensor, List[int]]:
    xs = x._shape
    nd = len(xs)
    H, W = xs[nd - 2], xs[nd - 1]
    B = 1
    for i in range(nd - 2):
        B *= xs[i]
    if p[0] * 2 > k[0] or p[1] * 2 > k[1]:
        raise RuntimeError("pad should be at most half of effective kernel size")
    OH = pool_out_size(H, k[0], s[0], p[0], d[0], ceil_mode)
    OW = pool_out_size(W, k[1], s[1], p[1], d[1], ceil_mode)
    if OH <= 0 or OW <= 0:
        raise RuntimeError("Output size is too small")
    g = _PoolGeo(B, H, W, OH, OW, k[0], k[1], s[0], s[1], p[0], p[1], d[0], d[1])
    xc = _c(x.detach())
    out = _new([B, OH, OW], x._dt)
    if (x._dt == _F32 and not (is_grad_enabled() and x.requires_grad) and H == 1 and k[0] == 1
            and p[1] == 0 and d[1] == 1 and k[1] <= W):
        # inference fast path (max_pool1d): no index bookkeeping. Same
        # selection rule as _maxpool_k (first max, NaN propagates).
        xp = xc._ptr(float32)
        op = out._ptr(float32)
        for b in range(B):
            row = xp + b * W
            for ox in range(OW):
                st = ox * s[1]
                best = row[st]
                if best == best:
                    for kx in range(1, k[1]):
                        if st + kx >= W:
                            break
                        v = row[st + kx]
                        if v > best or v != v:
                            best = v
                            if v != v:
                                break
                op[b * OW + ox] = best
        return (out.view(out_lead + [OH, OW]), List[int]())
    idx = [0] * (B * OH * OW)
    _dispatch_float(x._dt, _maxpool_k, xc, g, out, idx)
    out = out.view(out_lead + [OH, OW])
    if not _grad_on(x):
        return (out, idx)
    return (_attach(out, _MaxPoolNode(x, idx, g, xs.copy())), idx)

def _avgpool_div(g: _PoolGeo, oy: int, ox: int, count_include_pad: bool, divisor_override: int) -> Tuple[int, int, int, int, int]:
    hstart = oy * g.sh - g.ph
    wstart = ox * g.sw - g.pw
    hend = min(hstart + g.kh, g.H + g.ph)
    wend = min(wstart + g.kw, g.W + g.pw)
    pool_size = (hend - hstart) * (wend - wstart)
    hstart = max(hstart, 0)
    wstart = max(wstart, 0)
    hend = min(hend, g.H)
    wend = min(wend, g.W)
    div = 0
    if divisor_override > 0:
        div = divisor_override
    elif count_include_pad:
        div = pool_size
    else:
        div = (hend - hstart) * (wend - wstart)
    return (hstart, hend, wstart, wend, div)

def _avgpool_k(z: T, x: Tensor, g: _PoolGeo, cip: bool, dov: int, out: Tensor, T: type):
    xp = x._ptr(T)
    op = out._ptr(T)
    for b in range(g.B):
        base = b * g.H * g.W
        for oy in range(g.OH):
            for ox in range(g.OW):
                hs, he, ws, we, div = _avgpool_div(g, oy, ox, cip, dov)
                if T is float32:
                    # ATen cpu_avg_pool2d: acc_t = opmath (float) sum, float division
                    sf = float32(0.0)
                    for iy in range(hs, he):
                        for ix in range(ws, we):
                            sf += xp[base + iy * g.W + ix]
                    op[(b * g.OH + oy) * g.OW + ox] = sf / float32(div) if (he > hs and we > ws) else T(0)
                    continue
                s = 0.0
                for iy in range(hs, he):
                    for ix in range(ws, we):
                        s += float(xp[base + iy * g.W + ix])
                op[(b * g.OH + oy) * g.OW + ox] = T(s / float(div)) if (he > hs and we > ws) else T(0)

def _avgpool_bwd_k(z: T, gy: Tensor, g: _PoolGeo, cip: bool, dov: int, gx: Tensor, T: type):
    gp = gy._ptr(T)
    q = gx._ptr(T)
    if T is float32:
        # ATen cpu_avg_pool2d_backward: grad_delta = gy / divide_factor (float), += into gx
        for i in range(g.B * g.H * g.W):
            q[i] = float32(0.0)
        for b in range(g.B):
            base = b * g.H * g.W
            for oy in range(g.OH):
                for ox in range(g.OW):
                    hs, he, ws, we, div = _avgpool_div(g, oy, ox, cip, dov)
                    v = gp[(b * g.OH + oy) * g.OW + ox] / float32(div)
                    for iy in range(hs, he):
                        for ix in range(ws, we):
                            q[base + iy * g.W + ix] += v
        return
    acc = [0.0] * (g.B * g.H * g.W)
    for b in range(g.B):
        base = b * g.H * g.W
        for oy in range(g.OH):
            for ox in range(g.OW):
                hs, he, ws, we, div = _avgpool_div(g, oy, ox, cip, dov)
                v = float(gp[(b * g.OH + oy) * g.OW + ox]) / float(div)
                for iy in range(hs, he):
                    for ix in range(ws, we):
                        acc[base + iy * g.W + ix] += v
    for i in range(len(acc)):
        q[i] = T(acc[i])

class _AvgPoolNode(_NodeBase[Tensor]):
    geo: _PoolGeo
    cip: bool
    dov: int
    xshape: List[int]

    def __init__(self, x: Tensor, geo: _PoolGeo, cip: bool, dov: int, xshape: List[int]):
        super().__init__("AvgPool2DBackward0", [x])
        self.geo = geo
        self.cip = cip
        self.dov = dov
        self.xshape = xshape

    def apply(self, grad: Tensor) -> List[Optional[Tensor]]:
        g = self.geo
        gy = _c(grad.detach())
        gx = _new([g.B, g.H, g.W], gy._dt)
        _dispatch_float(gy._dt, _avgpool_bwd_k, gy, g, self.cip, self.dov, gx)
        return [Optional(gx.view(self.xshape))]

def avg_pool2d_raw(x: Tensor, k: Tuple[int, int], s: Tuple[int, int], p: Tuple[int, int], ceil_mode: bool,
                   count_include_pad: bool, divisor_override: int, out_lead: List[int]) -> Tensor:
    xs = x._shape
    nd = len(xs)
    H, W = xs[nd - 2], xs[nd - 1]
    B = 1
    for i in range(nd - 2):
        B *= xs[i]
    if p[0] * 2 > k[0] or p[1] * 2 > k[1]:
        raise RuntimeError("pad should be at most half of effective kernel size")
    OH = pool_out_size(H, k[0], s[0], p[0], 1, ceil_mode)
    OW = pool_out_size(W, k[1], s[1], p[1], 1, ceil_mode)
    if OH <= 0 or OW <= 0:
        raise RuntimeError("Output size is too small")
    g = _PoolGeo(B, H, W, OH, OW, k[0], k[1], s[0], s[1], p[0], p[1], 1, 1)
    xc = _c(x.detach())
    out = _new([B, OH, OW], x._dt)
    _dispatch_float(x._dt, _avgpool_k, xc, g, count_include_pad, divisor_override, out)
    out = out.view(out_lead + [OH, OW])
    if not _grad_on(x):
        return out
    return _attach(out, _AvgPoolNode(x, g, count_include_pad, divisor_override, xs.copy()))

# ======================================================================
# Padding: general N-d over trailing dims. pads in F.pad order
# (last dim first: [l_last, r_last, l_prev, r_prev, ...]).
# mode: 0 constant, 1 reflect, 2 replicate, 3 circular
# ======================================================================

def _src_index(i: int, L: int, l: int, mode: int) -> int:
    j = i - l
    if 0 <= j < L:
        return j
    if mode == 0:
        return -1
    elif mode == 1:
        if j < 0:
            return -j
        return 2 * (L - 1) - j
    elif mode == 2:
        return 0 if j < 0 else L - 1
    else:
        return j % L

def _pad_maps(shape: List[int], pads: List[int], mode: int) -> Tuple[List[int], List[List[int]]]:
    nd = len(shape)
    npad = len(pads) // 2
    out_shape = shape.copy()
    maps = List[List[int]]()
    for d in range(nd):
        k = nd - 1 - d
        L = shape[d]
        if k < npad:
            l = pads[2 * k]
            r = pads[2 * k + 1]
            if mode == 1 and (l >= L or r >= L):
                raise RuntimeError(f"Argument #4: Padding size should be less than the corresponding input dimension, but got: padding ({l}, {r}) at dimension {d} of input {shape}")
            if mode == 3 and (l > L or r > L):
                raise RuntimeError("Padding value causes wrapping around more than once.")
            O = L + l + r
            if O < 0:
                raise RuntimeError("negative output size from padding")
            out_shape[d] = O
            m = List[int](O)
            for i in range(O):
                m.append(_src_index(i, L, l, mode))
            maps.append(m)
        else:
            maps.append(list(range(L)))
    return (out_shape, maps)

def _pad_k(z: T, x: Tensor, out: Tensor, maps: List[List[int]], value: float, T: type):
    xp = x._ptr(T)
    op = out._ptr(T)
    oshape = out._shape
    nd = len(oshape)
    n = out.numel()
    xstr = x._strides
    idx = [0] * nd
    for o in range(n):
        src = 0
        ok = True
        for d in range(nd):
            s = maps[d][idx[d]]
            if s < 0:
                ok = False
                break
            src += s * xstr[d]
        op[o] = xp[src] if ok else T(value)
        d = nd - 1
        while d >= 0:
            idx[d] += 1
            if idx[d] < oshape[d]:
                break
            idx[d] = 0
            d -= 1

def _pad_bwd_k(z: T, gy: Tensor, gx: Tensor, maps: List[List[int]], T: type):
    gp = gy._ptr(T)
    oshape = gy._shape
    nd = len(oshape)
    n = gy.numel()
    xstr = gx._strides
    acc = [0.0] * gx.numel()
    idx = [0] * nd
    for o in range(n):
        src = 0
        ok = True
        for d in range(nd):
            s = maps[d][idx[d]]
            if s < 0:
                ok = False
                break
            src += s * xstr[d]
        if ok:
            acc[src] += float(gp[o])
        d = nd - 1
        while d >= 0:
            idx[d] += 1
            if idx[d] < oshape[d]:
                break
            idx[d] = 0
            d -= 1
    q = gx._ptr(T)
    for i in range(len(acc)):
        q[i] = T(acc[i])

class _PadNode(_NodeBase[Tensor]):
    maps: List[List[int]]
    xshape: List[int]

    def __init__(self, x: Tensor, maps: List[List[int]], name: str):
        super().__init__(name, [x])
        self.maps = maps
        self.xshape = x._shape.copy()

    def apply(self, grad: Tensor) -> List[Optional[Tensor]]:
        gy = _c(grad.detach())
        gx = _new(self.xshape, gy._dt)
        _dispatch_float(gy._dt, _pad_bwd_k, gy, gx, self.maps)
        return [Optional(gx)]

def pad_raw(x: Tensor, pads: List[int], mode: int, value: float) -> Tensor:
    if len(pads) % 2 != 0:
        raise RuntimeError("Padding length must be divisible by 2")
    if len(pads) // 2 > len(x._shape):
        raise RuntimeError("Padding length should be less than or equal to two times the input dimension")
    out_shape, maps = _pad_maps(x._shape, pads, mode)
    xc = _c(x.detach())
    out = _new(out_shape, x._dt)
    if x._dt == _F64 or x._dt == _F32:
        _dispatch_float(x._dt, _pad_k, xc, out, maps, value)
    else:
        raise NotImplementedError("pad: only floating tensors are supported")
    if not _grad_on(x):
        return out
    names = ["ConstantPadNdBackward0", "ReflectionPadBackward0", "ReplicationPadBackward0", "CircularPadBackward0"]
    return _attach(out, _PadNode(x, maps, names[mode]))

# ======================================================================
# Transposed convolution via the conv adjoint:
#   conv_transpose(x, w) == d conv(z, w) / d z applied to gy=x,
#   where z has the transposed output shape. Backward:
#   gx = conv(gy_out, w), gw = d conv(gy_out, w)/dw with gy=x.
# ======================================================================

class _ConvTNode(_NodeBase[Tensor]):
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int
    has_bias: bool
    xshape: List[int]
    wshape: List[int]

    def __init__(self, x: Tensor, w: Tensor, b: Optional[Tensor], stride: Tuple[int, int], padding: Tuple[int, int],
                 dilation: Tuple[int, int], groups: int, xshape: List[int], wshape: List[int]):
        ins = [x, w]
        if b is not None:
            ins.append(b.__val__())
        super().__init__("ConvolutionBackward0", ins)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = b is not None
        self.xshape = xshape
        self.wshape = wshape

    def apply(self, grad: Tensor) -> List[Optional[Tensor]]:
        from torch._conv import conv2d_forward, conv2d_backward
        x = _c(self.inputs[0].detach())
        w = _c(self.inputs[1].detach())
        gy = _c(grad.detach())
        need_x = self.inputs[0].requires_grad
        need_w = self.inputs[1].requires_grad
        res = List[Optional[Tensor]]()
        if need_x:
            gx = conv2d_forward(gy, w, None, self.stride, self.padding, self.dilation, self.groups)
            res.append(gx.view(self.xshape))
        else:
            res.append(None)
        if need_w:
            # conv(z=gy, w) has output x-shape; its weight grad with upstream x
            _, gw, _ = conv2d_backward(gy, w, x, self.stride, self.padding, self.dilation, self.groups, False, True, False)
            res.append(gw.view(self.wshape))
        else:
            res.append(None)
        if self.has_bias:
            dims = [0, 2, 3]
            res.append(gy.sum(dims))
        return res

def conv_transpose2d_raw(x: Tensor, w: Tensor, b: Optional[Tensor], stride: Tuple[int, int], padding: Tuple[int, int],
                         output_padding: Tuple[int, int], dilation: Tuple[int, int], groups: int,
                         xshape: List[int], wshape: List[int]) -> Tensor:
    from torch._conv import conv2d_backward
    N, Cin, H, W = x._shape[0], x._shape[1], x._shape[2], x._shape[3]
    if w._shape[0] != Cin:
        raise RuntimeError(f"Given transposed=1, weight of size {Tuple(wshape)}, expected input{Tuple(xshape)} to have {w._shape[0]} channels, but got {Cin} channels instead")
    Cout = w._shape[1] * groups
    KH, KW = w._shape[2], w._shape[3]
    OH = (H - 1) * stride[0] - 2 * padding[0] + dilation[0] * (KH - 1) + output_padding[0] + 1
    OW = (W - 1) * stride[1] - 2 * padding[1] + dilation[1] * (KW - 1) + output_padding[1] + 1
    if OH <= 0 or OW <= 0:
        raise RuntimeError("Calculated output size is too small")
    z = _zeros([N, Cout, OH, OW], x._dt)
    xc = _c(x.detach())
    wc = _c(w.detach())
    out, _, _ = conv2d_backward(z, wc, xc, stride, padding, dilation, groups, True, False, False)
    if b is not None:
        out = out + b.__val__().detach().reshape([1, Cout, 1, 1])
    out = out.detach()
    return _attach(out, _ConvTNode(x, w, b, stride, padding, dilation, groups, xshape, wshape))
