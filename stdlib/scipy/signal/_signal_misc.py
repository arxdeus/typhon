# Ported from scipy 1.18.1 (BSD-3-Clause)

# max_len_seq, firwin_2d, gammatone, find_peaks_cwt, envelope.
# Deviations: gammatone returns float arrays (b, a) for both ftypes; envelope
# supports real 1-D/2-D input (complex input is not supported); find_peaks_cwt
# only accepts the default Ricker wavelet.

import math
import cmath
import numpy as np
from ._fir_filter_design import firwin
from ._signaltools import convolve

_MLS_TAPS = {2: [1], 3: [2], 4: [3], 5: [3], 6: [5], 7: [6], 8: [7, 6, 1],
             9: [5], 10: [7], 11: [9], 12: [11, 10, 4], 13: [12, 11, 8],
             14: [13, 12, 2], 15: [14], 16: [15, 13, 4], 17: [14],
             18: [11], 19: [18, 17, 14], 20: [17], 21: [19], 22: [21],
             23: [18], 24: [23, 22, 17], 25: [22], 26: [25, 24, 20],
             27: [26, 25, 22], 28: [25], 29: [27], 30: [29, 28, 7],
             31: [28], 32: [31, 30, 10]}

def max_len_seq(nbits: int, state = None, length: Optional[int] = None, taps = None):
    tp: List[int] = []
    if taps is None:
        if nbits not in _MLS_TAPS:
            raise ValueError('nbits must be between 2 and 32 if taps is None')
        tp = _MLS_TAPS[nbits]
    else:
        ta = np.asarray(taps).flatten()
        vals: List[int] = []
        for i in range(ta.size):
            vals.append(int(ta[i]))
        vals = sorted(set(vals))
        vals = vals[::-1]
        if len(vals) < 1 or vals[0] > nbits or vals[len(vals) - 1] < 0:
            raise ValueError('taps must be non-empty with values between zero and nbits (inclusive)')
        tp = vals
    n_max = (1 << nbits) - 1
    n = n_max
    if length is not None:
        n = int(length)
        if n < 0:
            raise ValueError('length must be greater than or equal to 0')
    st = np.ones(nbits, dtype=int)
    if state is not None:
        sa = np.asarray(state).flatten()
        if sa.size != nbits:
            raise ValueError('state must be a 1-D array of size nbits')
        for i in range(nbits):
            st[i] = 1 if sa[i] != 0 else 0
    allz = True
    for i in range(nbits):
        if st[i] != 0:
            allz = False
    if allz:
        raise ValueError('state must not be all zeros')
    seq = np.empty(n, dtype=int)
    idx = 0
    for i in range(n):
        fb = st[idx]
        seq[i] = fb
        for t in tp:
            fb ^= st[(t + idx) % nbits]
        st[idx] = fb
        idx = (idx + 1) % nbits
    rolled = np.empty(nbits, dtype=int)
    for i in range(nbits):
        rolled[i] = st[(i + idx) % nbits]
    return seq.astype(np.int8), rolled.astype(np.int8)

def firwin_2d(hsize, window, fc = None, fs: float = 2, circular: bool = False,
              pass_zero = True, scale: bool = True):
    h0 = int(hsize[0])
    h1 = int(hsize[1])
    if circular:
        if fc is None:
            raise ValueError('Cutoff frequency `fc` must be provided when `circular` is True')
        n_r = max(h0, h1) * 8
        win_r = firwin(n_r, fc, None, window, pass_zero, scale, fs)
        f1 = np.linspace(-1.0, 1.0, h0)
        f2 = np.linspace(-1.0, 1.0, h1)
        grid = np.linspace(0.0, 1.0, n_r)
        out = np.empty((h1, h0))
        for i in range(h1):
            for j in range(h0):
                r = math.sqrt(f1[j] * f1[j] + f2[i] * f2[i])
                out[i, j] = np.interp(r, grid, win_r)
        return out
    if static.len(window) != 2:
        compile_error('window must be a 2-element tuple or list')
    row_filter = firwin(h0, fc, None, window[0], pass_zero, scale, fs)
    col_filter = firwin(h1, fc, None, window[1], pass_zero, scale, fs)
    out = np.empty((h0, h1))
    for i in range(h0):
        for j in range(h1):
            out[i, j] = row_filter[i] * col_filter[j]
    return out

def _hz_to_erb(hz: float) -> float:
    return hz / 9.26449 + 24.7

def gammatone(freq: float, ftype: str, order: Optional[int] = None,
              numtaps: Optional[int] = None, fs: Optional[float] = None):
    f = float(freq)
    fsv = 2.0 if fs is None else float(fs)
    ft = ftype.lower()
    if not (0 < f and f < fsv / 2):
        raise ValueError(f'The frequency must be between 0 and {fsv / 2} (Nyquist), but given {f}.')
    if ft != 'fir' and ft != 'iir':
        raise ValueError('ftype must be either fir or iir.')
    if ft == 'fir':
        od = 4 if order is None else order
        nt = max(int(fsv * 0.015), 15) if numtaps is None else numtaps
        if not (0 < od and od <= 24):
            raise ValueError('Invalid order: order must be > 0 and <= 24.')
        bw = 1.019 * _hz_to_erb(f)
        b = np.empty(nt)
        scale = 2 * math.pow(2 * math.pi * bw, float(od))
        scale /= math.gamma(float(od))
        scale /= fsv
        for i in range(nt):
            t = i / fsv
            v = math.pow(t, float(od - 1)) * math.exp(-2 * math.pi * bw * t)
            v = v * math.cos(2 * math.pi * f * t)
            b[i] = v * scale
        return b, np.array([1.0])
    T = 1.0 / fsv
    bw = 2 * math.pi * 1.019 * _hz_to_erb(f)
    fr = 2 * f * math.pi * T
    bwT = bw * T
    g1 = -2 * cmath.exp(2j * fr) * T
    g2 = 2 * cmath.exp(-(bwT) + 1j * fr) * T
    g3 = math.sqrt(3 + math.pow(2.0, 1.5)) * math.sin(fr)
    g4 = math.sqrt(3 - math.pow(2.0, 1.5)) * math.sin(fr)
    g5 = cmath.exp(2j * fr)
    g = g1 + g2 * (math.cos(fr) - g4)
    g *= (g1 + g2 * (math.cos(fr) + g4))
    g *= (g1 + g2 * (math.cos(fr) - g3))
    g *= (g1 + g2 * (math.cos(fr) + g3))
    den = (-2 / math.exp(2 * bwT) - 2 * g5 + 2 * (1 + g5) / math.exp(bwT))
    g /= den ** 4
    gm = math.hypot(g.real, g.imag)
    b = np.empty(5)
    a = np.empty(9)
    b[0] = (T ** 4) / gm
    b[1] = -4 * T ** 4 * math.cos(fr) / math.exp(bw * T) / gm
    b[2] = 6 * T ** 4 * math.cos(2 * fr) / math.exp(2 * bw * T) / gm
    b[3] = -4 * T ** 4 * math.cos(3 * fr) / math.exp(3 * bw * T) / gm
    b[4] = T ** 4 * math.cos(4 * fr) / math.exp(4 * bw * T) / gm
    a[0] = 1
    a[1] = -8 * math.cos(fr) / math.exp(bw * T)
    a[2] = 4 * (4 + 3 * math.cos(2 * fr)) / math.exp(2 * bw * T)
    a[3] = -8 * (6 * math.cos(fr) + math.cos(3 * fr)) / math.exp(3 * bw * T)
    a[4] = 2 * (18 + 16 * math.cos(2 * fr) + math.cos(4 * fr)) / math.exp(4 * bw * T)
    a[5] = -8 * (6 * math.cos(fr) + math.cos(3 * fr)) / math.exp(5 * bw * T)
    a[6] = 4 * (4 + 3 * math.cos(2 * fr)) / math.exp(6 * bw * T)
    a[7] = -8 * math.cos(fr) / math.exp(7 * bw * T)
    a[8] = math.exp(-8 * bw * T)
    return b, a

# ------------------------------------------------------------ find_peaks_cwt

def _ricker(points: int, a: float) -> np.ndarray[float, 1]:
    A = 2 / (math.sqrt(3 * a) * math.pow(math.pi, 0.25))
    wsq = a * a
    out = np.empty(points)
    for i in range(points):
        v = i - (points - 1.0) / 2
        xsq = v * v
        out[i] = A * (1 - xsq / wsq) * math.exp(-xsq / (2 * wsq))
    return out

def _cwt_ricker(data: np.ndarray[float, 1], widths: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
    out = np.empty((widths.size, data.size))
    for ind in range(widths.size):
        w = widths[ind]
        N = int(min(10 * w, float(data.size)))
        wd = _ricker(N, w)[::-1].copy()
        c = convolve(data, wd, 'same')
        for j in range(data.size):
            out[ind, j] = c[j]
    return out

def _relmax_cols(row: np.ndarray[float, 1]) -> List[bool]:
    n = row.size
    out = [False] * n
    for i in range(n):
        l = row[max(i - 1, 0)]
        r = row[min(i + 1, n - 1)]
        out[i] = row[i] > l and row[i] > r
    return out

def _percentile10(window: List[float], per: float) -> float:
    s = sorted(window)
    n = len(s)
    idx = per / 100.0 * (n - 1)
    i = int(idx)
    if float(i) == idx:
        return s[i]
    j = min(i + 1, n - 1)
    f = idx - i
    return s[i] * (1 - f) + s[j] * f

class _Line:
    rows: List[int]
    cols: List[int]
    gap: int

    def __init__(self, r: int, c: int):
        self.rows = [r]
        self.cols = [c]
        self.gap = 0

def find_peaks_cwt(vector, widths, wavelet = None, max_distances = None,
                   gap_thresh = None, min_length = None, min_snr: float = 1,
                   noise_perc: float = 10, window_size = None):
    data = np.asarray(vector, float).flatten()
    w = np.atleast_1d(np.asarray(widths, float)).flatten()
    gt = float(math.ceil(w[0]))
    if gap_thresh is not None:
        gt = float(gap_thresh)
    md = w / 4.0
    if max_distances is not None:
        md = np.asarray(max_distances, float).flatten()
    cwt = _cwt_ricker(data, w)
    nrow = cwt.shape[0]
    ncol = cwt.shape[1]
    if md.size < nrow:
        raise ValueError('Max_distances must have at least as many rows as matr')
    relmax = List[List[bool]]()
    for i in range(nrow):
        relmax.append(_relmax_cols(np.ascontiguousarray(cwt[i])))
    start_row = -1
    for i in range(nrow):
        for v in relmax[i]:
            if v:
                start_row = i
                break
    peaks: List[int] = []
    if start_row < 0:
        return np.array(peaks, dtype=int)
    ridge: List[_Line] = []
    for c in range(ncol):
        if relmax[start_row][c]:
            ridge.append(_Line(start_row, c))
    final: List[_Line] = []
    for row in range(start_row - 1, -1, -1):
        for ln in ridge:
            ln.gap += 1
        prev_cols = [ln.cols[len(ln.cols) - 1] for ln in ridge]
        for c in range(ncol):
            if not relmax[row][c]:
                continue
            found = -1
            if len(prev_cols) > 0:
                best = abs(c - prev_cols[0])
                closest = 0
                for k in range(1, len(prev_cols)):
                    d = abs(c - prev_cols[k])
                    if d < best:
                        best = d
                        closest = k
                if float(best) <= md[row]:
                    found = closest
            if found >= 0:
                ridge[found].cols.append(c)
                ridge[found].rows.append(row)
                ridge[found].gap = 0
            else:
                ridge.append(_Line(row, c))
        for ind in range(len(ridge) - 1, -1, -1):
            if float(ridge[ind].gap) > gt:
                final.append(ridge[ind])
                ridge.pop(ind)
    lines = final + ridge
    ml = float(math.ceil(nrow / 4.0))
    if min_length is not None:
        ml = float(min_length)
    ws = float(math.ceil(ncol / 20.0))
    if window_size is not None:
        ws = float(window_size)
    wsz = int(ws)
    hf = wsz // 2
    odd = wsz % 2
    row_one = np.ascontiguousarray(cwt[0])
    noises = np.empty(ncol)
    for ind in range(ncol):
        a0 = max(ind - hf, 0)
        a1 = min(ind + hf + odd, ncol)
        win: List[float] = []
        for t in range(a0, a1):
            win.append(row_one[t])
        noises[ind] = _percentile10(win, noise_perc)
    for ln in lines:
        if float(len(ln.rows)) < ml:
            continue
        # sort by row, as the ridge line start (smallest row)
        k0 = 0
        for k in range(len(ln.rows)):
            if ln.rows[k] < ln.rows[k0]:
                k0 = k
        snr = abs(cwt[ln.rows[k0], ln.cols[k0]] / noises[ln.cols[k0]])
        if snr < min_snr:
            continue
        peaks.append(ln.cols[k0])
    peaks = sorted(peaks)
    return np.array(peaks, dtype=int)

# ------------------------------------------------------------------ envelope

def _env_row(z: np.ndarray[float, 1], b0: int, b1: int, n_out: int, squared: bool,
             residual: int):
    # residual: 0 none, 1 lowpass, 2 all. returns (env, res)
    n = z.size
    fak = n_out / n
    bs = b0
    be = b1
    if not (-n // 2 <= bs and bs < be and be <= (n + 1) // 2):
        raise ValueError(f'`-n//2 <= bp_in[0] < bp_in[1] <= (n+1)//2` does not hold for n={n} and bp_in={(b0, b1)}!')
    Z = np.zeros(n, dtype=complex)
    rf = np.fft.rfft(z)
    for i in range(n // 2 + 1):
        Z[i] = rf[i]
    if bs > 0:
        for i in range(bs, be):
            Z[i] = Z[i] * 2.0
    elif be > 0:
        for i in range(1, be):
            Z[i] = Z[i] * 2.0
    m = be - bs
    sub = np.zeros(m, dtype=complex)
    for j in range(m):
        sub[j] = Z[(bs + j) % n]
    zb = np.fft.ifft(sub, n_out)
    env = np.empty(n_out)
    for i in range(n_out):
        v = zb[i] * fak
        if squared:
            env[i] = v.real * v.real + v.imag * v.imag
        else:
            env[i] = math.hypot(v.real, v.imag)
    res = np.zeros(n_out)
    if residual == 0:
        return env, res
    if not (bs <= 0 and 0 < be):
        for j in range(m):
            Z[(bs + j) % n] = complex(0.0)
    else:
        for i in range(0, be):
            Z[i] = complex(0.0)
        for i in range(n + bs, n):
            Z[i] = complex(0.0)
    if residual == 1:
        if be > 0:
            for i in range(be, (n + 1) // 2):
                Z[i] = complex(0.0)
        else:
            for i in range(n + bs, n):
                Z[i] = complex(0.0)
            for i in range(0, (n + 1) // 2):
                Z[i] = complex(0.0)
    if n_out != n:
        mm = min(n, n_out)
        if mm % 2 == 0:
            f = 2.0 if n_out < n else 0.5
            Z[mm // 2] = Z[mm // 2] * f
    ir = np.fft.irfft(Z, n_out)
    for i in range(n_out):
        res[i] = fak * ir[i]
    return env, res

def _env_1d(za, b0: int, b1: int, no: int, squared: bool, rk: int) -> np.ndarray[float, 2]:
    env, res = _env_row(np.ascontiguousarray(za), b0, b1, no, squared, rk)
    out = np.empty((2, no))
    for i in range(no):
        out[0, i] = env[i]
        out[1, i] = res[i]
    return out

def _env_2d(za, ax: int, b0: int, b1: int, no: int, squared: bool, rk: int) -> np.ndarray[float, 3]:
    rows = za.shape[0] if ax == 1 else za.shape[1]
    shape0 = rows if ax == 1 else no
    shape1 = no if ax == 1 else rows
    out = np.empty((2, shape0, shape1))
    for r in range(rows):
        row = np.ascontiguousarray(za[r]) if ax == 1 else np.ascontiguousarray(za[:, r])
        e, s = _env_row(row, b0, b1, no, squared, rk)
        for i in range(no):
            if ax == 1:
                out[0, r, i] = e[i]
                out[1, r, i] = s[i]
            else:
                out[0, i, r] = e[i]
                out[1, i, r] = s[i]
    return out

def _res_kind(residual) -> int:
    if isinstance(residual, str):
        if residual == 'lowpass':
            return 1
        elif residual == 'all':
            return 2
        raise ValueError(f"residual={residual} not in ['lowpass', 'all', None]!")
    return 0

def envelope(z, bp_in = (1, None), n_out: Optional[int] = None, squared: bool = False,
             residual = 'lowpass', axis: int = -1):
    za = np.asarray(z, float)
    nd = za.ndim
    ax = axis + nd if axis < 0 else axis
    if not (0 <= ax and ax < nd):
        raise ValueError(f'Invalid parameter axis={axis} for z.shape={za.shape}!')
    n = za.shape[ax]
    if n <= 0:
        raise ValueError(f'z.shape[axis] not > 0 for z.shape={za.shape}, axis={axis}!')
    no = n if n_out is None else n_out
    if no <= 0:
        raise ValueError(f'n_out={n_out} is not a positive integer or None!')
    b0 = -(n // 2)
    b1 = (n + 1) // 2
    if bp_in[0] is not None:
        b0 = bp_in[0]
    if bp_in[1] is not None:
        b1 = bp_in[1]
    rk = _res_kind(residual)
    if static.len(za.shape) == 1:
        o = _env_1d(za, b0, b1, no, squared, rk)
        if isinstance(residual, str):
            return o
        return o[0].copy()
    elif static.len(za.shape) == 2:
        o = _env_2d(za, ax, b0, b1, no, squared, rk)
        if isinstance(residual, str):
            return o
        return o[0].copy()
    else:
        compile_error('envelope: only 1-D and 2-D real input is supported')
