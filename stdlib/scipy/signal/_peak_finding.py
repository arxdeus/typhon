# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# find_peaks returns (peaks, properties) where properties is a
# Dict[str, ndarray[float, 1]]: integer-valued properties (left_bases,
# right_bases, plateau_sizes, left_edges, right_edges) are stored as float
# because a dict needs a single value type (deviation). `width` is not
# supported. Conditions may be a number, a (min, max) tuple with None
# entries, or None.

import numpy as np

def _local_maxima_1d(x):
    peaks: List[int] = []
    left: List[int] = []
    right: List[int] = []
    n = x.size
    i = 1
    i_max = n - 1
    while i < i_max:
        if x[i - 1] < x[i]:
            i_ahead = i + 1
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1
            if x[i_ahead] < x[i]:
                left.append(i)
                right.append(i_ahead - 1)
                peaks.append((i + i_ahead - 1) // 2)
                i = i_ahead
        i += 1
    return peaks, left, right

def _cond(c) -> Tuple[Optional[float], Optional[float]]:
    lo: Optional[float] = None
    hi: Optional[float] = None
    if c is None:
        pass
    elif isinstance(c, Tuple):
        if c[0] is not None:
            lo = float(c[0])
        if c[1] is not None:
            hi = float(c[1])
    else:
        lo = float(c)
    return lo, hi

def _keep(v: float, lo: Optional[float], hi: Optional[float]) -> bool:
    if lo is not None and not (lo <= v):
        return False
    if hi is not None and not (v <= hi):
        return False
    return True

def _prominence(x, peak: int, wlen: int):
    n = x.size
    i_min = 0
    i_max = n - 1
    if wlen >= 2:
        i_min = max(peak - wlen // 2, i_min)
        i_max = min(peak + wlen // 2, i_max)
    i = peak
    lb = peak
    left_min = x[peak]
    while i_min <= i and x[i] <= x[peak]:
        if x[i] < left_min:
            left_min = x[i]
            lb = i
        i -= 1
    i = peak
    rb = peak
    right_min = x[peak]
    while i <= i_max and x[i] <= x[peak]:
        if x[i] < right_min:
            right_min = x[i]
            rb = i
        i += 1
    return float(x[peak] - max(left_min, right_min)), lb, rb

def peak_prominences(x, peaks, wlen: Optional[int] = None):
    xa = np.asarray(x).astype(float)
    pk = np.asarray(peaks)
    w = -1 if wlen is None else wlen
    n = pk.size
    prom = np.empty(n, dtype=float)
    lb = np.empty(n, dtype=int)
    rb = np.empty(n, dtype=int)
    for i in range(n):
        p, l, r = _prominence(xa, int(pk[i]), w)
        prom[i] = p
        lb[i] = l
        rb[i] = r
    return prom, lb, rb

def _width1(x, peak: int, rel_height: float, prom: float, i_min: int, i_max: int):
    if rel_height < 0:
        raise ValueError('`rel_height` must be greater or equal to 0.0')
    height = x[peak] - prom * rel_height
    i = peak
    while i_min < i and height < x[i]:
        i -= 1
    left_ip = float(i)
    if x[i] < height:
        left_ip += (height - x[i]) / (x[i + 1] - x[i])
    i = peak
    while i < i_max and height < x[i]:
        i += 1
    right_ip = float(i)
    if x[i] < height:
        right_ip -= (height - x[i]) / (x[i - 1] - x[i])
    return right_ip - left_ip, height, left_ip, right_ip

def peak_widths(x, peaks, rel_height: float = 0.5, prominence_data = None, wlen: Optional[int] = None):
    xa = np.asarray(x).astype(float)
    pk = np.asarray(peaks)
    n = pk.size
    if prominence_data is None:
        pr, lb, rb = peak_prominences(xa, pk, wlen)
        return _widths_impl(xa, pk, rel_height, pr, lb, rb)
    else:
        return _widths_impl(xa, pk, rel_height, prominence_data[0], prominence_data[1], prominence_data[2])

def _widths_impl(xa, pk, rel_height: float, pr, lb, rb):
    n = pk.size
    w = np.empty(n, dtype=float)
    h = np.empty(n, dtype=float)
    l = np.empty(n, dtype=float)
    r = np.empty(n, dtype=float)
    for i in range(n):
        a1, a2, a3, a4 = _width1(xa, int(pk[i]), rel_height, float(pr[i]), int(lb[i]), int(rb[i]))
        w[i] = a1; h[i] = a2; l[i] = a3; r[i] = a4
    return w, h, l, r

def _farr(v: List[float]):
    a = np.empty(len(v), dtype=float)
    for i in range(len(v)):
        a[i] = v[i]
    return a

def find_peaks(x, height = None, threshold = None, distance: Optional[float] = None,
               prominence = None, width = None, wlen: Optional[int] = None,
               rel_height: float = 0.5, plateau_size = None):
    xa = np.asarray(x).astype(float)
    if xa.ndim != 1:
        compile_error('`x` must be a 1-D array')
    if distance is not None and distance < 1:
        raise ValueError('`distance` must be greater or equal to 1')
    peaks, le, re = _local_maxima_1d(xa)
    props: Dict[str, List[float]] = {}

    def filt(keep: List[bool]):
        nonlocal peaks
        peaks = [peaks[i] for i in range(len(peaks)) if keep[i]]
        for k in props.keys():
            v = props[k]
            props[k] = [v[i] for i in range(len(v)) if keep[i]]

    if plateau_size is not None:
        lo, hi = _cond(plateau_size)
        props['plateau_sizes'] = [float(re[i] - le[i] + 1) for i in range(len(peaks))]
        props['left_edges'] = [float(v) for v in le]
        props['right_edges'] = [float(v) for v in re]
        filt([_keep(v, lo, hi) for v in props['plateau_sizes']])
    if height is not None:
        lo, hi = _cond(height)
        props['peak_heights'] = [xa[p] for p in peaks]
        filt([_keep(v, lo, hi) for v in props['peak_heights']])
    if threshold is not None:
        lo, hi = _cond(threshold)
        lt = [xa[p] - xa[p - 1] for p in peaks]
        rt = [xa[p] - xa[p + 1] for p in peaks]
        props['left_thresholds'] = lt
        props['right_thresholds'] = rt
        keep = [True] * len(peaks)
        for i in range(len(peaks)):
            if lo is not None and not (lo <= min(lt[i], rt[i])):
                keep[i] = False
            if hi is not None and not (max(lt[i], rt[i]) <= hi):
                keep[i] = False
        filt(keep)
    if distance is not None:
        n = len(peaks)
        keep = [True] * n
        d = int(-(-distance // 1))  # ceil
        order = list(range(n))
        order.sort(key=lambda i: xa[peaks[i]])  # stable argsort ascending
        for idx in range(n - 1, -1, -1):
            j = order[idx]
            if not keep[j]:
                continue
            k = j - 1
            while 0 <= k and peaks[j] - peaks[k] < d:
                keep[k] = False
                k -= 1
            k = j + 1
            while k < n and peaks[k] - peaks[j] < d:
                keep[k] = False
                k += 1
        filt(keep)
    if prominence is not None or width is not None:
        w = -1 if wlen is None else wlen
        pr: List[float] = []
        lbs: List[float] = []
        rbs: List[float] = []
        for p in peaks:
            a, l, r = _prominence(xa, p, w)
            pr.append(a)
            lbs.append(float(l))
            rbs.append(float(r))
        props['prominences'] = pr
        props['left_bases'] = lbs
        props['right_bases'] = rbs
    if prominence is not None:
        lo, hi = _cond(prominence)
        filt([_keep(v, lo, hi) for v in props['prominences']])
    if width is not None:
        wd: List[float] = []
        wh: List[float] = []
        li: List[float] = []
        ri: List[float] = []
        for i in range(len(peaks)):
            a1, a2, a3, a4 = _width1(xa, peaks[i], rel_height, props['prominences'][i],
                                     int(props['left_bases'][i]), int(props['right_bases'][i]))
            wd.append(a1); wh.append(a2); li.append(a3); ri.append(a4)
        props['widths'] = wd
        props['width_heights'] = wh
        props['left_ips'] = li
        props['right_ips'] = ri
        lo, hi = _cond(width)
        filt([_keep(v, lo, hi) for v in wd])
    out = np.empty(len(peaks), dtype=int)
    for i in range(len(peaks)):
        out[i] = peaks[i]
    res: Dict[str, np.ndarray[float, 1]] = {}
    for k, v in props.items():
        res[k] = _farr(v)
    return out, res

def _argrel(data, cmp: int, order: int, mode: str):
    a = np.asarray(data)
    if a.ndim != 1:
        compile_error('argrelextrema: only 1-D input is supported natively')
    n = a.size
    res: List[int] = []
    for i in range(n):
        ok = True
        for s in range(1, order + 1):
            for j in (i + s, i - s):
                jj = j
                if mode == 'clip':
                    jj = min(max(j, 0), n - 1)
                elif mode == 'wrap':
                    jj = j % n
                c = (a[i] > a[jj]) if cmp == 0 else (a[i] < a[jj])
                if not c:
                    ok = False
        if ok:
            res.append(i)
    out = np.empty(len(res), dtype=int)
    for i in range(len(res)):
        out[i] = res[i]
    return (out,)

def argrelmax(data, axis: int = 0, order: int = 1, mode: str = 'clip'):
    return _argrel(data, 0, order, mode)

def argrelmin(data, axis: int = 0, order: int = 1, mode: str = 'clip'):
    return _argrel(data, 1, order, mode)

def argrelextrema(data, comparator, axis: int = 0, order: int = 1, mode: str = 'clip'):
    a = np.asarray(data)
    n = a.size
    res: List[int] = []
    for i in range(n):
        ok = True
        for s in range(1, order + 1):
            for j in (i + s, i - s):
                jj = min(max(j, 0), n - 1) if mode == 'clip' else j % n
                if not comparator(a[i], a[jj]):
                    ok = False
        if ok:
            res.append(i)
    out = np.empty(len(res), dtype=int)
    for i in range(len(res)):
        out[i] = res[i]
    return (out,)
