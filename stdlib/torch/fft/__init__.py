# Ported from torch 2.15.0 (BSD-3-Clause)
#
# torch.fft on top of numpy's native pocketfft port. Complex results are
# complex64/complex128 tensors (see torch.view_as_real). rfft/irfft/fft/ifft
# are differentiable.

from numpy.fft.pocketfft import fft as _raw_fft, ifft as _raw_ifft, rfft as _raw_rfft, irfft as _raw_irfft
from torch._tensor import Tensor, Node, arange, div, cat, mul, _DType, _F32, _F64, _C64, _C128, _dt_or, _is_complex, _is_float, _new, _needs, _attach, _wrap, movedim, _complex_code, _real_code, _to_cplx, narrow, zeros, view_as_real, view_as_complex, _O, conj, _pad, real, add, _contig_as, select, clone

def _norm_fwd(norm: Optional[str]) -> Optional[str]:
    return norm

def _move_last(x: Tensor, d: int) -> Tensor:
    return movedim(x, d, len(x._shape) - 1).contiguous()

def _k_cfft(z, x: Tensor, out: Tensor, N: int, inv: bool, norm: Optional[str]):
    # x complex contiguous with last dim N (already sized)
    T = type(z)
    rows = x.numel() // N if N > 0 else 0
    px = Ptr[T](x._st.ptr) + x._off
    po = Ptr[T](out._st.ptr)
    for r in range(rows):
        for i in range(N):
            po[r * N + i] = px[r * N + i]
        if inv:
            _raw_ifft(po + r * N, N, norm)
        else:
            _raw_fft(po + r * N, N, norm)

def _resize_last(x: Tensor, n: int) -> Tensor:
    m = x._shape[-1]
    if n == m:
        return x
    if n < m:
        return narrow(x, -1, 0, n)
    return _pad(x, [0, n - m], "constant", 0.0)

def _c2c(input: Tensor, n: Optional[int], dim: int, norm: Optional[str], inv: bool) -> Tensor:
    d = _wrap(dim, len(input._shape))
    x = input if _is_complex(input._dt) else _to_cplx(input, _complex_code(input._dt if _is_float(input._dt) else _F32))
    xm = movedim(x.detach(), d, len(x._shape) - 1)
    N = xm._shape[-1]
    if n is not None:
        N = n
    xm = _resize_last_c(xm, N).contiguous()
    out = _new(xm._shape.copy(), xm._dt)
    if xm._dt == _C128:
        _k_cfft(complex(), xm, out, N, inv, norm)
    else:
        _k_cfft(complex64(), xm, out, N, inv, norm)
    r = movedim(out, len(out._shape) - 1, d)
    if _needs([x]):
        nd = _FFTBackward(x, d, N, x._shape[d], inv, norm, False)
        _attach(r, nd)
    return r

def _resize_last_c(x: Tensor, n: int) -> Tensor:
    m = x._shape[-1]
    if n == m:
        return x
    if n < m:
        return narrow(x, -1, 0, n)
    r = view_as_real(x).detach()
    rr = _pad(movedim(r, len(r._shape) - 1, 0), [0, n - m], "constant", 0.0)
    return view_as_complex(movedim(rr, 0, len(rr._shape) - 1).contiguous())

def _inv_scale(norm: Optional[str], N: int) -> float:
    # scale of the adjoint relative to the opposite-direction transform
    return 1.0

class _FFTBackward(Node):
    d: int
    N: int
    m: int
    inv: bool
    nrm: Optional[str]
    real_in: bool

    def __init__(self, x: Tensor, d: int, N: int, m: int, inv: bool, norm: Optional[str], real_in: bool):
        super().__init__("FftC2CBackward0" if not real_in else "FftR2CBackward0", [x])
        self.d = d
        self.N = N
        self.m = m
        self.inv = inv
        self.nrm = norm
        self.real_in = real_in

    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        # adjoint of unnormalized DFT F is conj(F) = N * IDFT; adjoint of
        # c2c(fwd, norm) is c2c(inv, swapped norm).
        sw = _swap(self.nrm)
        gi = _c2c(g, None, self.d, sw, not self.inv)
        if self.N != self.m:
            gm = movedim(gi, self.d, len(gi._shape) - 1)
            if self.N > self.m:
                gm = narrow(gm, -1, 0, self.m)
            else:
                gm = _resize_last_c(gm.contiguous(), self.m)
            gi = movedim(gm, len(gm._shape) - 1, self.d)
        if self.real_in:
            return [_O(real(gi))]
        return [_O(gi)]

def _swap(norm: Optional[str]) -> Optional[str]:
    if norm is None or norm == "backward":
        return "forward"
    if norm == "forward":
        return "backward"
    return norm

def fft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    if not _is_complex(input._dt):
        x = _to_cplx(input, _complex_code(input._dt if _is_float(input._dt) else _F32))
        r = _c2c(x, n, dim, norm, False)
        return r
    return _c2c(input, n, dim, norm, False)

def ifft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    return _c2c(input, n, dim, norm, True)

def _k_rfft(z, x: Tensor, out: Tensor, N: int, norm: Optional[str]):
    T0 = type(z)
    M = N // 2 + 1
    rows = x.numel() // N if N > 0 else 0
    px = Ptr[T0](x._st.ptr) + x._off
    if T0 is float32:
        po = Ptr[complex64](out._st.ptr)
        for r in range(rows):
            _raw_rfft(px + r * N, N, True, norm, po + r * M)
    else:
        po2 = Ptr[complex](out._st.ptr)
        for r in range(rows):
            _raw_rfft(px + r * N, N, True, norm, po2 + r * M)

def rfft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    if _is_complex(input._dt):
        raise RuntimeError("rfft expects a real input tensor, but got " + str(input.dtype))
    x = input if _is_float(input._dt) else input.float()
    d = _wrap(dim, len(x._shape))
    xm = movedim(x.detach(), d, len(x._shape) - 1)
    N = xm._shape[-1]
    if n is not None:
        N = n
    xm = clone(_resize_last(xm, N))  # pocketfft may scribble on its input
    M = N // 2 + 1
    oshape = xm._shape[:-1] + [M]
    out = _new(oshape, _complex_code(x._dt))
    if x._dt == _F64:
        _k_rfft(0.0, xm, out, N, norm)
    else:
        _k_rfft(float32(0.0), xm, out, N, norm)
    r = movedim(out, len(out._shape) - 1, d)
    if _needs([x]):
        _attach(r, _RFFTBackward(x, d, N, x._shape[d], norm))
    return r

class _RFFTBackward(Node):
    d: int
    N: int
    m: int
    nrm: Optional[str]

    def __init__(self, x: Tensor, d: int, N: int, m: int, norm: Optional[str]):
        super().__init__("FftR2CBackward0", [x])
        self.d = d
        self.N = N
        self.m = m
        self.nrm = norm

    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        # rfft = first M bins of the full c2c fft of the real input. Pad the
        # gradient with zeros to N bins and apply the c2c adjoint, then take
        # the real part.
        M = self.N // 2 + 1
        gm = movedim(g, self.d, len(g._shape) - 1)
        gm = _resize_last_c(gm.contiguous(), self.N)
        gi = _c2c(gm, None, -1, _swap(self.nrm), True)
        gr = real(gi)
        if self.N != self.m:
            if self.N > self.m:
                gr = narrow(gr, -1, 0, self.m)
            else:
                gr = _pad(gr, [0, self.m - self.N], "constant", 0.0)
        return [_O(movedim(gr, len(gr._shape) - 1, self.d))]

def _k_irfft(z, x: Tensor, out: Tensor, Nin: int, N: int, norm: Optional[str]):
    T0 = type(z)
    rows = out.numel() // N if N > 0 else 0
    po = Ptr[T0](out._st.ptr)
    if T0 is float32:
        px = Ptr[complex64](x._st.ptr) + x._off
        for r in range(rows):
            _raw_irfft(px + r * Nin, N, norm, po + r * N)
    else:
        px2 = Ptr[complex](x._st.ptr) + x._off
        for r in range(rows):
            _raw_irfft(px2 + r * Nin, N, norm, po + r * N)

def irfft(input: Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Tensor:
    x = input if _is_complex(input._dt) else _to_cplx(input, _complex_code(input._dt if _is_float(input._dt) else _F32))
    d = _wrap(dim, len(x._shape))
    xm = movedim(x.detach(), d, len(x._shape) - 1)
    m = xm._shape[-1]
    N = 2 * (m - 1)
    if n is not None:
        N = n
    Nin = N // 2 + 1
    xm = clone(_resize_last_c(xm, Nin))
    out = _new(xm._shape[:-1] + [N], _real_code(x._dt))
    if x._dt == _C128:
        _k_irfft(0.0, xm, out, Nin, N, norm)
    else:
        _k_irfft(float32(0.0), xm, out, Nin, N, norm)
    r = movedim(out, len(out._shape) - 1, d)
    if _needs([x]):
        _attach(r, _IRFFTBackward(x, d, N, m, norm))
    return r

class _IRFFTBackward(Node):
    d: int
    N: int
    m: int
    nrm: Optional[str]

    def __init__(self, x: Tensor, d: int, N: int, m: int, norm: Optional[str]):
        super().__init__("FftC2RBackward0", [x])
        self.d = d
        self.N = N
        self.m = m
        self.nrm = norm

    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        # torch: grad = rfft(g, swapped norm), doubled for non-edge bins.
        Nin = self.N // 2 + 1
        gm = movedim(g, self.d, len(g._shape) - 1).contiguous()
        gr = rfft(gm, None, -1, _swap(self.nrm))
        # double bins 1..(last-non-nyquist)
        last = Nin - 1 if self.N % 2 == 0 else Nin
        w = _new([Nin], _real_code(gr._dt))
        for i in range(Nin):
            v = 2.0 if (i >= 1 and i < last) else 1.0
            if w._dt == _F64:
                Ptr[float](w._st.ptr)[i] = v
            else:
                Ptr[float32](w._st.ptr)[i] = float32(v)
        gr = mul(gr, w)
        if Nin != self.m:
            gr = _resize_last_c(gr, self.m)
        return [_O(movedim(gr, len(gr._shape) - 1, self.d))]

def rfftfreq(n: int, d: float = 1.0, dtype: Optional[_DType] = None) -> Tensor:
    return div(arange(0, n // 2 + 1, dtype=_DType(_dt_or(dtype, _F32))), n * d)

def fftfreq(n: int, d: float = 1.0, dtype: Optional[_DType] = None) -> Tensor:
    m = (n - 1) // 2 + 1
    a = arange(0, m, dtype=_DType(_dt_or(dtype, _F32)))
    b = arange(-(n // 2), 0, dtype=_DType(_dt_or(dtype, _F32)))
    return div(cat([a, b]), n * d)

def fftshift(input: Tensor, dim = None) -> Tensor:
    from torch._tensor import roll, _to_list
    if dim is None:
        ds = list(range(len(input._shape)))
    else:
        ds = _to_list(dim)
    return roll(input, [input._shape[d] // 2 for d in ds], ds)

def ifftshift(input: Tensor, dim = None) -> Tensor:
    from torch._tensor import roll, _to_list
    if dim is None:
        ds = list(range(len(input._shape)))
    else:
        ds = _to_list(dim)
    return roll(input, [-(input._shape[d] // 2) for d in ds], ds)

# ---------------------------------------------------------------- stft / istft

def _stft(input: Tensor, n_fft: int, hop_length: Optional[int] = None, win_length: Optional[int] = None,
          window: Optional[Tensor] = None, center: bool = True, pad_mode: str = "reflect", normalized: bool = False,
          onesided: Optional[bool] = None, return_complex: Optional[bool] = None) -> Tensor:
    from torch._tensor import unsqueeze, squeeze, ones, transpose, _unfold, zeros as _zeros, cat as _cat
    hop = n_fft // 4
    if hop_length is not None:
        hop = hop_length
    wl = n_fft
    if win_length is not None:
        wl = win_length
    if return_complex is None or not return_complex:
        raise RuntimeError("stft requires the return_complex parameter be given for real inputs, and will further require that return_complex=True in a future PyTorch release.")
    batched = len(input._shape) == 2
    x = input if batched else unsqueeze(input, 0)
    if center:
        p = n_fft // 2
        x3 = unsqueeze(x, 1)
        x = squeeze(_pad(x3, [p, p], pad_mode, 0.0), 1)
    w: Tensor
    if window is not None:
        w = window.__val__()
    else:
        w = ones([wl], dtype=_DType(x._dt))
    if wl < n_fft:
        left = (n_fft - wl) // 2
        w = _pad(w, [left, n_fft - wl - left], "constant", 0.0)
    frames = _unfold(x, -1, n_fft, hop)  # (B, n_frames, n_fft)
    fr = mul(frames, w)
    one = True
    if onesided is not None:
        one = onesided
    spec = rfft(fr, None, -1, "ortho" if normalized else None) if one else fft(fr, None, -1, "ortho" if normalized else None)
    out = transpose(spec, -1, -2)  # (B, freq, n_frames)
    return out if batched else squeeze(out, 0)

def _istft(input: Tensor, n_fft: int, hop_length: Optional[int] = None, win_length: Optional[int] = None,
           window: Optional[Tensor] = None, center: bool = True, normalized: bool = False,
           onesided: Optional[bool] = None, length: Optional[int] = None, return_complex: bool = False) -> Tensor:
    from torch._tensor import unsqueeze, squeeze, ones, transpose, zeros as _zeros, add as _add, div as _div, select as _select, mul as _mul, _real_code
    hop = n_fft // 4
    if hop_length is not None:
        hop = hop_length
    wl = n_fft
    if win_length is not None:
        wl = win_length
    batched = len(input._shape) == 3
    X = input if batched else unsqueeze(input, 0)
    rdt = _real_code(X._dt)
    w: Tensor
    if window is not None:
        w = window.__val__()
    else:
        w = ones([wl], dtype=_DType(rdt))
    if wl < n_fft:
        left = (n_fft - wl) // 2
        w = _pad(w, [left, n_fft - wl - left], "constant", 0.0)
    one = True
    if onesided is not None:
        one = onesided
    nrm: Optional[str] = "ortho" if normalized else None
    fr = transpose(X, -1, -2)  # (B, n_frames, freq)
    frames = irfft(fr, n_fft, -1, nrm) if one else real(ifft(fr, n_fft, -1, nrm))
    frames = mul(frames, w)
    B = frames._shape[0]
    nf = frames._shape[1]
    total = n_fft + hop * (nf - 1)
    # overlap-add via index_add (differentiable)
    from torch._tensor import index_add, _new, _I64, reshape
    idx = _new([nf * n_fft], _I64)
    pi = Ptr[int](idx._st.ptr)
    for f in range(nf):
        for k in range(n_fft):
            pi[f * n_fft + k] = f * hop + k
    y = index_add(_zeros([B, total], dtype=_DType(rdt)), 1, idx, reshape(frames, [B, nf * n_fft]))
    wsq = mul(w, w).detach()
    env = index_add(_zeros([total], dtype=_DType(rdt)), 0, idx, reshape(unsqueeze(wsq, 0).expand([nf, n_fft]), [nf * n_fft]))
    start = n_fft // 2 if center else 0
    end = total
    if length is not None:
        end = start + length.__val__()
    elif center:
        end = total - n_fft // 2
    if end > total:
        y = _pad(y, [0, end - total], "constant", 0.0)
        env = _pad(env, [0, end - total], "constant", 1.0)
    y = narrow(y, 1, start, end - start)
    env = narrow(env, 0, start, end - start)
    from torch._tensor import _where3, gt, ones_like
    env_safe = _where3(gt(env, 1e-11), env, ones_like(env))
    y = div(y, env_safe)
    return y if batched else squeeze(y, 0)
