# Ported from pandas 3.1.1 (BSD-3-Clause)

# Column storage and the numeric kernels (ports of pandas/_libs/algos.pyx,
# groupby.pyx, window/aggregations.pyx and core/nanops.py). All kernels
# reproduce the exact floating point operation order of pandas/numpy.

import numpy as np
from numpy import ndarray
from .scalar import Scalar, K_FLOAT, K_INT, K_BOOL, K_STR, K_NONE, K_DT, K_TD, NA_STR, _isnan
from ._dt import NAT, Timestamp, Timedelta, fmt_ts_full, fmt_td, parse_ts, parse_td

NAN = float("nan")

# The reference pandas build (clang, -ffp-contract=on) fuses `a += b * c`
# in its Cython kernels into a fused multiply-add. Typhon does not contract,
# so the kernels call fma() explicitly to reproduce pandas bit-for-bit.
@pure
@llvm
def fma(a: float, b: float, c: float) -> float:
    declare double @llvm.fma.f64(double, double, double)
    %r = call double @llvm.fma.f64(double %a, double %b, double %c)
    ret double %r

def _dtype_name(kind: int) -> str:
    if kind == K_FLOAT:
        return "float64"
    if kind == K_INT:
        return "int64"
    if kind == K_BOOL:
        return "bool"
    if kind == K_DT:
        return "datetime64[us]"
    if kind == K_TD:
        return "timedelta64[ns]"
    return "str"

def _kind_of_dtype(dtype) -> int:
    if isinstance(dtype, str):
        if dtype in ("float", "float64", "f8", "double"):
            return K_FLOAT
        if dtype in ("int", "int64", "i8"):
            return K_INT
        if dtype in ("bool", "boolean"):
            return K_BOOL
        if dtype in ("str", "string", "object"):
            return K_STR
        if dtype in ("datetime64[ns]", "datetime64", "datetime64[us]", "M8[ns]"):
            return K_DT
        if dtype in ("timedelta64[ns]", "timedelta64", "m8[ns]"):
            return K_TD
        raise TypeError(f"data type '{dtype}' not understood")
    elif dtype is float:
        return K_FLOAT
    elif dtype is int:
        return K_INT
    elif dtype is bool:
        return K_BOOL
    elif dtype is str:
        return K_STR
    else:
        compile_error("pandas: unsupported dtype")

# ---------------------------------------------------------------------------
# numpy-compatible pairwise summation (numpy/_core/src/umath/loops_utils.h)

def _pairwise_sum(a: Ptr[float], n: int, stride: int) -> float:
    if n < 8:
        res = -0.0
        for i in range(n):
            res += a[i * stride]
        return res
    elif n <= 128:
        r0 = a[0 * stride]
        r1 = a[1 * stride]
        r2 = a[2 * stride]
        r3 = a[3 * stride]
        r4 = a[4 * stride]
        r5 = a[5 * stride]
        r6 = a[6 * stride]
        r7 = a[7 * stride]
        i = 8
        m = n - (n % 8)
        while i < m:
            r0 += a[(i + 0) * stride]
            r1 += a[(i + 1) * stride]
            r2 += a[(i + 2) * stride]
            r3 += a[(i + 3) * stride]
            r4 += a[(i + 4) * stride]
            r5 += a[(i + 5) * stride]
            r6 += a[(i + 6) * stride]
            r7 += a[(i + 7) * stride]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[i * stride]
            i += 1
        return res
    else:
        n2 = n // 2
        n2 -= n2 % 8
        return _pairwise_sum(a, n2, stride) + _pairwise_sum(a + n2 * stride, n - n2, stride)

def npsum(v: List[float]) -> float:
    if len(v) == 0:
        return 0.0
    return _pairwise_sum(v._ptr, len(v), 1)

# ---------------------------------------------------------------------------
# Column: one runtime-typed buffer

class Column:
    kind: int
    f: List[float]
    i: List[int]      # int64 and bool values
    s: List[str]

    def __init__(self, kind: int, f: List[float], i: List[int], s: List[str]):
        self.kind = kind
        self.f = f
        self.i = i
        self.s = s

    @staticmethod
    def floats(f: List[float]) -> Column:
        return Column(K_FLOAT, f, List[int](), List[str]())

    @staticmethod
    def ints(i: List[int]) -> Column:
        return Column(K_INT, List[float](), i, List[str]())

    @staticmethod
    def bools(b: List[bool]) -> Column:
        return Column(K_BOOL, List[float](), [1 if x else 0 for x in b], List[str]())

    @staticmethod
    def strs(s: List[str]) -> Column:
        return Column(K_STR, List[float](), List[int](), s)

    @staticmethod
    def empty(kind: int, n: int = 0) -> Column:
        c = Column(kind, List[float](n), List[int](n), List[str](n))
        return c

    @staticmethod
    def full(kind: int, n: int, v: Scalar) -> Column:
        if kind == K_FLOAT:
            x = v.num()
            return Column.floats([x for _ in range(n)])
        if kind == K_STR:
            return Column.strs([v.s for _ in range(n)])
        return Column(kind, List[float](), [v.i for _ in range(n)], List[str]())

    @staticmethod
    def from_scalars(v: List[Scalar]) -> Column:
        kind = K_NONE
        for x in v:
            if x.kind == K_NONE:
                continue
            if kind == K_NONE:
                kind = x.kind
            elif kind != x.kind:
                if (kind == K_INT or kind == K_FLOAT or kind == K_BOOL) and (x.kind == K_INT or x.kind == K_FLOAT):
                    kind = K_FLOAT
                elif kind == K_INT and x.kind == K_BOOL:
                    pass
                else:
                    kind = K_STR
        if kind == K_NONE:
            kind = K_FLOAT
        has_none = False
        for x in v:
            if x.kind == K_NONE:
                has_none = True
        if has_none and (kind == K_INT or kind == K_BOOL):
            kind = K_FLOAT
        c = Column.empty(kind)
        for x in v:
            c.append(x)
        return c

    @staticmethod
    def of(data) -> Column:
        if isinstance(data, Column):
            return data.copy()
        elif isinstance(data, ndarray):
            if static.len(data.shape) != 1:
                compile_error("pandas: expected 1-D array")
            return Column._of_list([data[k] for k in range(data.size)])
        elif isinstance(data, List):
            return Column._of_list(data)
        elif hasattr(data, "_col"):
            return data._col.copy()
        elif hasattr(data, "__iter__"):
            return Column._of_list(list(data))
        else:
            compile_error("pandas: unsupported column data")

    @staticmethod
    def _of_list(data: List[T], T: type) -> Column:
        if T is float:
            return Column.floats(data.copy())
        elif T is bool:
            return Column.bools(data)
        elif T is int:
            return Column.ints(data.copy())
        elif T is str:
            return Column.strs(data.copy())
        elif T is Scalar:
            return Column.from_scalars(data)
        elif T is Timestamp:
            return Column(K_DT, List[float](), [x.value for x in data], List[str]())
        elif T is Timedelta:
            return Column(K_TD, List[float](), [x.value for x in data], List[str]())
        elif isinstance(T, Optional):
            return Column.from_scalars([Scalar.of(x) for x in data])
        elif hasattr(T, "__index__"):
            return Column.ints([int(x) for x in data])
        elif hasattr(T, "__float__"):
            return Column.floats([float(x) for x in data])
        else:
            compile_error("pandas: unsupported element type")

    def __len__(self) -> int:
        if self.kind == K_FLOAT:
            return len(self.f)
        if self.kind == K_STR:
            return len(self.s)
        return len(self.i)

    @property
    def dtype(self) -> str:
        return _dtype_name(self.kind)

    @property
    def is_numeric(self) -> bool:
        return self.kind != K_STR and self.kind != K_DT and self.kind != K_TD

    def copy(self) -> Column:
        return Column(self.kind, self.f.copy(), self.i.copy(), self.s.copy())

    def get(self, k: int) -> Scalar:
        if self.kind == K_FLOAT:
            return Scalar(K_FLOAT, self.f[k], 0, "")
        if self.kind == K_STR:
            return Scalar(K_STR, 0.0, 0, self.s[k])
        return Scalar(self.kind, 0.0, self.i[k], "")

    def getf(self, k: int) -> float:
        if self.kind == K_FLOAT:
            return self.f[k]
        if self.kind == K_STR:
            if self.s[k] == NA_STR:
                return NAN
            raise TypeError("cannot convert str column to float")
        if (self.kind == K_DT or self.kind == K_TD) and self.i[k] == NAT:
            return NAN
        return float(self.i[k])

    def isna_at(self, k: int) -> bool:
        if self.kind == K_DT or self.kind == K_TD:
            return self.i[k] == NAT
        if self.kind == K_FLOAT:
            return _isnan(self.f[k])
        if self.kind == K_STR:
            return self.s[k] == NA_STR
        return False

    def append(self, v: Scalar):
        if self.kind == K_FLOAT:
            self.f.append(v.num())
        elif self.kind == K_STR:
            if v.kind == K_STR:
                self.s.append(v.s)
            elif v.kind == K_NONE or (v.kind == K_FLOAT and _isnan(v.f)):
                self.s.append(NA_STR)
            else:
                self.s.append(str(v))
        elif self.kind == K_DT or self.kind == K_TD:
            self.i.append(NAT if v.isna() else v.i)
        else:
            if v.kind == K_FLOAT:
                self.i.append(int(v.f))
            else:
                self.i.append(v.i)

    def set(self, k: int, v: Scalar):
        # upcasts like pandas when needed (int <- float or NaN)
        if self.kind == K_FLOAT:
            if v.kind == K_STR:
                self._to_str()
                self.s[k] = v.s
            else:
                self.f[k] = v.num()
        elif self.kind == K_STR:
            if v.kind == K_STR:
                self.s[k] = v.s
            elif v.isna():
                self.s[k] = NA_STR
            else:
                self.s[k] = str(v)
        elif self.kind == K_DT or self.kind == K_TD:
            self.i[k] = NAT if v.isna() else v.i
        else:
            if v.kind == K_FLOAT or v.kind == K_NONE:
                self.astype_inplace(K_FLOAT)
                self.f[k] = v.num()
            elif v.kind == K_STR:
                self._to_str()
                self.s[k] = v.s
            else:
                if self.kind == K_BOOL and v.kind == K_INT:
                    self.kind = K_INT
                self.i[k] = v.i

    def _to_str(self):
        n = len(self)
        self.s = [str(self.get(k)) if not self.isna_at(k) else NA_STR for k in range(n)]
        self.f = List[float]()
        self.i = List[int]()
        self.kind = K_STR

    def astype_inplace(self, kind: int):
        c = self.astype(kind)
        self.kind = c.kind
        self.f = c.f
        self.i = c.i
        self.s = c.s

    def astype(self, kind: int) -> Column:
        if kind == self.kind:
            return self.copy()
        n = len(self)
        if kind == K_DT or kind == K_TD:
            if self.kind == K_STR:
                return Column(kind, List[float](), [NAT if self.s[k] == NA_STR else _parse_dt(self.s[k], kind) for k in range(n)], List[str]())
            if self.kind == K_FLOAT:
                return Column(kind, List[float](), [NAT if _isnan(x) else int(x) for x in self.f], List[str]())
            return Column(kind, List[float](), self.i.copy(), List[str]())
        if self.kind == K_DT or self.kind == K_TD:
            if kind == K_FLOAT:
                return Column.floats([NAN if self.i[k] == NAT else float(self.i[k]) for k in range(n)])
            if kind == K_INT:
                return Column.ints(self.i.copy())
        if kind == K_FLOAT:
            if self.kind == K_STR:
                return Column.floats([NAN if self.s[k] == NA_STR else float(self.s[k]) for k in range(n)])
            return Column.floats([float(self.i[k]) for k in range(n)])
        if kind == K_INT:
            if self.kind == K_FLOAT:
                for x in self.f:
                    if _isnan(x):
                        raise ValueError("Cannot convert non-finite values (NA or inf) to integer")
                return Column.ints([int(x) for x in self.f])
            if self.kind == K_STR:
                return Column.ints([int(x) for x in self.s])
            return Column.ints(self.i.copy())
        if kind == K_BOOL:
            if self.kind == K_FLOAT:
                return Column(K_BOOL, List[float](), [1 if x != 0.0 else 0 for x in self.f], List[str]())
            if self.kind == K_STR:
                return Column(K_BOOL, List[float](), [1 if len(x) > 0 else 0 for x in self.s], List[str]())
            return Column(K_BOOL, List[float](), [1 if x != 0 else 0 for x in self.i], List[str]())
        c = self.copy()
        c._to_str()
        return c

    def take(self, idx: List[int]) -> Column:
        # idx entries of -1 produce missing values (upcasting int to float)
        has_missing = False
        for k in idx:
            if k < 0:
                has_missing = True
        if self.kind == K_FLOAT:
            return Column.floats([self.f[k] if k >= 0 else NAN for k in idx])
        if self.kind == K_STR:
            return Column.strs([self.s[k] if k >= 0 else NA_STR for k in idx])
        if self.kind == K_DT or self.kind == K_TD:
            return Column(self.kind, List[float](), [self.i[k] if k >= 0 else NAT for k in idx], List[str]())
        if has_missing:
            return Column.floats([float(self.i[k]) if k >= 0 else NAN for k in idx])
        return Column(self.kind, List[float](), [self.i[k] for k in idx], List[str]())

    def slice(self, a: int, b: int) -> Column:
        return self.take(list(range(a, b)))

    def extend(self, o: Column) -> Column:
        # concatenation with dtype promotion
        a = self
        b = o
        kind = a.kind
        if a.kind != b.kind:
            if a.kind == K_DT or b.kind == K_DT or a.kind == K_TD or b.kind == K_TD:
                kind = K_STR
            elif a.kind == K_STR or b.kind == K_STR:
                kind = K_STR
            elif a.kind == K_FLOAT or b.kind == K_FLOAT:
                kind = K_FLOAT
            else:
                kind = K_INT
        x = a.astype(kind)
        y = b.astype(kind)
        x.f.extend(y.f)
        x.i.extend(y.i)
        x.s.extend(y.s)
        return x

    def to_floats(self) -> List[float]:
        n = len(self)
        if self.kind == K_FLOAT:
            return self.f.copy()
        if self.kind == K_STR:
            return [self.getf(k) for k in range(n)]
        if self.kind == K_DT or self.kind == K_TD:
            return [NAN if x == NAT else float(x) for x in self.i]
        return [float(x) for x in self.i]

    def scalars(self) -> List[Scalar]:
        return [self.get(k) for k in range(len(self))]

    def hash_key(self, k: int) -> Scalar:
        return self.get(k)

    # formatting for CSV (pandas uses repr of floats, str of ints)
    def fmt_csv(self, k: int, float_format: Optional[str], na_rep: str, decimal: str) -> str:
        if self.kind == K_FLOAT:
            x = self.f[k]
            if _isnan(x):
                return na_rep
            if float_format is not None:
                r = float_format % x
            else:
                r = _float_repr(x)
            if decimal != ".":
                r = r.replace(".", decimal)
            return r
        if self.kind == K_INT:
            return str(self.i[k])
        if self.kind == K_DT:
            return na_rep if self.i[k] == NAT else fmt_ts_full(self.i[k])
        if self.kind == K_TD:
            return na_rep if self.i[k] == NAT else fmt_td(self.i[k])
        if self.kind == K_BOOL:
            return "True" if self.i[k] else "False"
        if self.s[k] == NA_STR:
            return na_rep
        return self.s[k]

def _parse_dt(s: str, kind: int) -> int:
    return parse_ts(s) if kind == K_DT else parse_td(s)

def _float_repr(x: float) -> str:
    if x == float("inf"):
        return "inf"
    if x == -float("inf"):
        return "-inf"
    return repr(x)

import internal.static as static

# ---------------------------------------------------------------------------
# nanops

def nansum(c: Column, skipna: bool = True, min_count: int = 0) -> float:
    vals = List[float](len(c))
    cnt = 0
    for k in range(len(c)):
        x = c.getf(k)
        if _isnan(x):
            if not skipna:
                return NAN
            vals.append(0.0)
        else:
            vals.append(x)
            cnt += 1
    if cnt < min_count:
        return NAN
    return npsum(vals) + 0.0

def nancount(c: Column) -> int:
    n = 0
    for k in range(len(c)):
        if not c.isna_at(k):
            n += 1
    return n

def nanmean(c: Column, skipna: bool = True) -> float:
    vals = List[float](len(c))
    cnt = 0
    for k in range(len(c)):
        x = c.getf(k)
        if _isnan(x):
            if not skipna:
                return NAN
            vals.append(0.0)
        else:
            vals.append(x)
            cnt += 1
    if cnt == 0:
        return NAN
    return npsum(vals) / float(cnt)

def nanvar(c: Column, skipna: bool = True, ddof: int = 1) -> float:
    n = len(c)
    vals = List[float](n)
    mask = List[bool](n)
    cnt = 0
    for k in range(n):
        x = c.getf(k)
        if _isnan(x):
            if not skipna:
                return NAN
            vals.append(0.0)
            mask.append(True)
        else:
            vals.append(x)
            mask.append(False)
            cnt += 1
    if cnt == 0:
        return NAN
    d = float(cnt - ddof)
    avg = npsum(vals) / float(cnt)
    sq = List[float](n)
    for k in range(n):
        if mask[k]:
            sq.append(0.0)
        else:
            t = avg - vals[k]
            sq.append(t * t)
    if d <= 0.0:
        return NAN
    return npsum(sq) / d

def nanstd(c: Column, skipna: bool = True, ddof: int = 1) -> float:
    from math import sqrt
    return sqrt(nanvar(c, skipna, ddof))

def nanprod(c: Column, skipna: bool = True) -> float:
    r = 1.0
    for k in range(len(c)):
        x = c.getf(k)
        if _isnan(x):
            if not skipna:
                return NAN
        else:
            r *= x
    return r

def nanminmax(c: Column, skipna: bool, is_max: bool) -> Scalar:
    best = Scalar.none()
    for k in range(len(c)):
        if c.isna_at(k):
            if not skipna:
                return Scalar(K_FLOAT, NAN, 0, "")
            continue
        v = c.get(k)
        if best.kind == K_NONE or (is_max and v > best) or (not is_max and v < best):
            best = v
    if best.kind == K_NONE:
        return Scalar(K_FLOAT, NAN, 0, "")
    return best

def _sorted_valid(c: Column) -> List[float]:
    v = [x for x in c.to_floats() if not _isnan(x)]
    v.sort()
    return v

def _lerp_quantile(a: List[float], q: float) -> float:
    n = len(a)
    if n == 0:
        return NAN
    vi = float(n - 1) * q
    import math
    p = int(math.floor(vi))
    if p < 0:
        p = 0
    nx = p + 1
    if nx > n - 1:
        nx = n - 1
    if p > n - 1:
        p = n - 1
    g = vi - float(p)
    A = a[p]
    B = a[nx]
    d = B - A
    if g >= 0.5:
        return B - d * (1.0 - g)
    return A + d * g

def nanquantile(c: Column, q: float) -> float:
    if not (0.0 <= q <= 1.0):
        raise ValueError("percentiles should all be in the interval [0, 1]")
    return _lerp_quantile(_sorted_valid(c), q)

def nanmedian(c: Column, skipna: bool = True) -> float:
    a = _sorted_valid(c)
    if not skipna and len(a) != len(c):
        return NAN
    n = len(a)
    if n == 0:
        return NAN
    if n % 2 == 1:
        return a[n // 2]
    # numpy median: mean of the two middle values
    return (a[n // 2 - 1] + a[n // 2]) / 2.0

# ---------------------------------------------------------------------------
# correlation (algos.nancorr / nancorr_spearman)

def nancorr(cols: List[List[float]], cov: bool, minp: int) -> List[List[float]]:
    K = len(cols)
    N = len(cols[0]) if K > 0 else 0
    res = [[0.0 for _ in range(K)] for _ in range(K)]
    no_nans = True
    for c in cols:
        for x in c:
            if not (x - x == 0.0):
                no_nans = False
    from math import sqrt
    for xi in range(K):
        for yi in range(xi + 1):
            nobs = 0
            ssqdmx = 0.0
            ssqdmy = 0.0
            covxy = 0.0
            meanx = 0.0
            meany = 0.0
            X = cols[xi]
            Y = cols[yi]
            for i in range(N):
                vx = X[i]
                vy = Y[i]
                if not no_nans and (not (vx - vx == 0.0) or not (vy - vy == 0.0)):
                    continue
                nobs += 1
                dx = vx - meanx
                dy = vy - meany
                meanx = fma(1. / float(nobs), dx, meanx)
                meany = fma(1. / float(nobs), dy, meany)
                ssqdmx = fma(vx - meanx, dx, ssqdmx)
                ssqdmy = fma(vy - meany, dy, ssqdmy)
                covxy = fma(vx - meanx, dy, covxy)
            val = NAN
            if nobs >= minp:
                divisor = (float(nobs) - 1.0) if cov else sqrt(ssqdmx * ssqdmy)
                if divisor != 0.0:
                    val = covxy / divisor
                    if not cov:
                        if val > 1.0:
                            val = 1.0
                        elif val < -1.0:
                            val = -1.0
            res[xi][yi] = val
            res[yi][xi] = val
    return res

def rank_average(x: List[float]) -> List[float]:
    # pandas rank_1d, method="average", na_option="keep", ascending
    n = len(x)
    idx = [k for k in range(n) if not _isnan(x[k])]
    idx.sort(key=lambda k: x[k])
    # stable sort: python sort is stable (pandas uses mergesort)
    out = [NAN for _ in range(n)]
    i = 0
    m = len(idx)
    while i < m:
        j = i
        while j + 1 < m and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        dups = j - i + 1
        sum_ranks = 0
        for t in range(i, j + 1):
            sum_ranks += t + 1
        r = float(sum_ranks) / float(dups)
        for t in range(i, j + 1):
            out[idx[t]] = r
        i = j + 1
    return out

def nancorr_spearman(cols: List[List[float]], minp: int) -> List[List[float]]:
    K = len(cols)
    N = len(cols[0]) if K > 0 else 0
    if N < minp:
        return [[NAN for _ in range(K)] for _ in range(K)]
    from math import sqrt
    ranked = [rank_average(c) for c in cols]
    finite = [[x - x == 0.0 for x in c] for c in cols]
    no_nans = True
    for f in finite:
        for b in f:
            if not b:
                no_nans = False
    res = [[0.0 for _ in range(K)] for _ in range(K)]
    for xi in range(K):
        for yi in range(xi + 1):
            sumx = 0.0
            sumxx = 0.0
            sumyy = 0.0
            if no_nans:
                mean = float(N + 1) / 2.
                for i in range(N):
                    vx = ranked[xi][i] - mean
                    vy = ranked[yi][i] - mean
                    sumx = fma(vx, vy, sumx)
                    sumxx = fma(vx, vx, sumxx)
                    sumyy = fma(vy, vy, sumyy)
            else:
                mx = List[float]()
                my = List[float]()
                all_ranks = True
                for i in range(N):
                    a = finite[xi][i]
                    b = finite[yi][i]
                    if a != b:
                        all_ranks = False
                    if a and b:
                        mx.append(ranked[xi][i])
                        my.append(ranked[yi][i])
                nobs = len(mx)
                if nobs < minp:
                    res[xi][yi] = NAN
                    res[yi][xi] = NAN
                    continue
                if not all_ranks:
                    mx = rank_average(mx)
                    my = rank_average(my)
                mean = float(nobs + 1) / 2.
                for i in range(nobs):
                    vx = mx[i] - mean
                    vy = my[i] - mean
                    sumx = fma(vx, vy, sumx)
                    sumxx = fma(vx, vx, sumxx)
                    sumyy = fma(vy, vy, sumyy)
            divisor = sqrt(sumxx * sumyy)
            v = sumx / divisor if divisor != 0.0 else NAN
            res[xi][yi] = v
            res[yi][xi] = v
    return res

# ---------------------------------------------------------------------------
# groupby kernels (groupby.pyx); labels are group codes 0..ngroups-1

def group_sum(vals: List[float], labels: List[int], ngroups: int, min_count: int = 0) -> List[float]:
    sumx = [0.0 for _ in range(ngroups)]
    comp = [0.0 for _ in range(ngroups)]
    nobs = [0 for _ in range(ngroups)]
    for i in range(len(vals)):
        lab = labels[i]
        if lab < 0:
            continue
        val = vals[i]
        if not _isnan(val):
            nobs[lab] += 1
            y = val - comp[lab]
            t = sumx[lab] + y
            comp[lab] = t - sumx[lab] - y
            if comp[lab] != comp[lab]:
                comp[lab] = 0.
            sumx[lab] = t
    return [sumx[g] if nobs[g] >= min_count else NAN for g in range(ngroups)]

def group_mean(vals: List[float], labels: List[int], ngroups: int) -> List[float]:
    sumx = [0.0 for _ in range(ngroups)]
    comp = [0.0 for _ in range(ngroups)]
    nobs = [0 for _ in range(ngroups)]
    for i in range(len(vals)):
        lab = labels[i]
        if lab < 0:
            continue
        val = vals[i]
        if not _isnan(val):
            nobs[lab] += 1
            y = val - comp[lab]
            t = sumx[lab] + y
            comp[lab] = t - sumx[lab] - y
            if comp[lab] != comp[lab]:
                comp[lab] = 0.
            sumx[lab] = t
    return [sumx[g] / float(nobs[g]) if nobs[g] > 0 else NAN for g in range(ngroups)]

def group_var(vals: List[float], labels: List[int], ngroups: int, ddof: int, is_std: bool) -> List[float]:
    from math import sqrt
    out = [0.0 for _ in range(ngroups)]
    mean = [0.0 for _ in range(ngroups)]
    nobs = [0 for _ in range(ngroups)]
    for i in range(len(vals)):
        lab = labels[i]
        if lab < 0:
            continue
        val = vals[i]
        if not _isnan(val):
            nobs[lab] += 1
            oldmean = mean[lab]
            mean[lab] += (val - oldmean) / float(nobs[lab])
            out[lab] = fma(val - mean[lab], val - oldmean, out[lab])
    res = List[float](ngroups)
    for g in range(ngroups):
        ct = nobs[g]
        if ct <= ddof:
            res.append(NAN)
        elif is_std:
            res.append(sqrt(out[g] / float(ct - ddof)))
        else:
            res.append(out[g] / float(ct - ddof))
    return res

def group_prod(vals: List[float], labels: List[int], ngroups: int) -> List[float]:
    out = [1.0 for _ in range(ngroups)]
    for i in range(len(vals)):
        lab = labels[i]
        if lab >= 0 and not _isnan(vals[i]):
            out[lab] *= vals[i]
    return out

# ---------------------------------------------------------------------------
# rolling kernels (window/aggregations.pyx), fixed windows

def _fixed_bounds(n: int, window: int, center: bool):
    start = List[int](n)
    end = List[int](n)
    offset = (window - 1) // 2 if center else 0
    for i in range(n):
        e = i + 1 + offset
        s = e - window
        if e > n:
            e = n
        if s < 0:
            s = 0
        if s > n:
            s = n
        start.append(s)
        end.append(e)
    return start, end

def _is_mono(start: List[int], end: List[int]) -> bool:
    for i in range(1, len(start)):
        if start[i] < start[i - 1] or end[i] < end[i - 1]:
            return False
    return True

def roll_sum(values: List[float], start: List[int], end: List[int], minp: int) -> List[float]:
    N = len(start)
    out = List[float](N)
    mono = _is_mono(start, end)
    sum_x = 0.0
    comp_add = 0.0
    comp_rem = 0.0
    nobs = 0
    ncons = 0
    prev = 0.0
    for i in range(N):
        s = start[i]
        e = end[i]
        if i == 0 or not mono or s >= end[i - 1]:
            prev = values[s] if s < len(values) else 0.0
            ncons = 0
            sum_x = 0.0
            comp_add = 0.0
            comp_rem = 0.0
            nobs = 0
            for j in range(s, e):
                val = values[j]
                if val == val:
                    nobs += 1
                    y = val - comp_add
                    t = sum_x + y
                    comp_add = t - sum_x - y
                    sum_x = t
                    if val == prev:
                        ncons += 1
                    else:
                        ncons = 1
                    prev = val
        else:
            for j in range(start[i - 1], s):
                val = values[j]
                if val == val:
                    nobs -= 1
                    y = -val - comp_rem
                    t = sum_x + y
                    comp_rem = t - sum_x - y
                    sum_x = t
            for j in range(end[i - 1], e):
                val = values[j]
                if val == val:
                    nobs += 1
                    y = val - comp_add
                    t = sum_x + y
                    comp_add = t - sum_x - y
                    sum_x = t
                    if val == prev:
                        ncons += 1
                    else:
                        ncons = 1
                    prev = val
        r = NAN
        if nobs == 0 and minp == 0:
            r = 0.0
        elif nobs >= minp:
            r = prev * float(nobs) if ncons >= nobs else sum_x
        out.append(r)
    return out

def roll_mean(values: List[float], start: List[int], end: List[int], minp: int) -> List[float]:
    from math import copysign
    N = len(start)
    out = List[float](N)
    mono = _is_mono(start, end)
    sum_x = 0.0
    comp_add = 0.0
    comp_rem = 0.0
    nobs = 0
    neg = 0
    ncons = 0
    prev = 0.0
    for i in range(N):
        s = start[i]
        e = end[i]
        if i == 0 or not mono or s >= end[i - 1]:
            comp_add = 0.0
            comp_rem = 0.0
            sum_x = 0.0
            nobs = 0
            neg = 0
            prev = values[s] if s < len(values) else 0.0
            ncons = 0
            for j in range(s, e):
                val = values[j]
                if val == val:
                    nobs += 1
                    y = val - comp_add
                    t = sum_x + y
                    comp_add = t - sum_x - y
                    sum_x = t
                    if copysign(1.0, val) < 0.0:
                        neg += 1
                    if val == prev:
                        ncons += 1
                    else:
                        ncons = 1
                    prev = val
        else:
            for j in range(start[i - 1], s):
                val = values[j]
                if val == val:
                    nobs -= 1
                    y = -val - comp_rem
                    t = sum_x + y
                    comp_rem = t - sum_x - y
                    sum_x = t
                    if copysign(1.0, val) < 0.0:
                        neg -= 1
            for j in range(end[i - 1], e):
                val = values[j]
                if val == val:
                    nobs += 1
                    y = val - comp_add
                    t = sum_x + y
                    comp_add = t - sum_x - y
                    sum_x = t
                    if copysign(1.0, val) < 0.0:
                        neg += 1
                    if val == prev:
                        ncons += 1
                    else:
                        ncons = 1
                    prev = val
        r = NAN
        if nobs >= minp and nobs > 0:
            r = sum_x / float(nobs)
            if ncons >= nobs:
                r = prev
            elif neg == 0 and r < 0:
                r = 0.0
            elif neg == nobs and r > 0:
                r = 0.0
        out.append(r)
    return out

def _add_var(val: float, st: List[float]):
    # st = [nobs, mean_x, ssqdm_x, compensation, unstable]
    if val != val:
        return
    prev_m2 = st[2]
    st[0] = st[0] + 1
    prev_mean = st[1] - st[3]
    y = val - st[3]
    t = y - st[1]
    st[3] = t + st[1] - y
    delta = t
    if st[0] != 0.0:
        st[1] = st[1] + delta / st[0]
    else:
        st[1] = 0.0
    st[2] = fma(val - prev_mean, val - st[1], st[2])
    if prev_m2 * (2.220446049250313e-16 * 1e3) > st[2]:
        st[4] = 1.0

def _remove_var(val: float, st: List[float]):
    # st = [nobs, mean_x, ssqdm_x, compensation_remove, unstable]
    if val == val:
        prev_m2 = st[2]
        st[0] = st[0] - 1
        if st[0] != 0.0:
            prev_mean = st[1] - st[3]
            y = val - st[3]
            t = y - st[1]
            st[3] = t + st[1] - y
            delta = t
            st[1] = st[1] - delta / st[0]
            st[2] = fma(-(val - prev_mean), val - st[1], st[2])
            if prev_m2 * (2.220446049250313e-16 * 1e3) > st[2]:
                st[4] = 1.0
        else:
            st[1] = 0.0
            st[2] = 0.0
            st[4] = 0.0

def roll_var(values: List[float], start: List[int], end: List[int], minp: int, ddof: int = 1) -> List[float]:
    if minp < 1:
        minp = 1
    N = len(start)
    out = List[float](N)
    mono = _is_mono(start, end)
    # nobs, mean, ssqdm, comp, unstable shared; compensation for add/remove separate
    nobs = 0.0
    mean_x = 0.0
    ssqdm = 0.0
    comp_add = 0.0
    comp_rem = 0.0
    unstable = False
    for i in range(N):
        s = start[i]
        e = end[i]
        recompute = i == 0 or not mono or s >= end[i - 1]
        if not recompute:
            for j in range(start[i - 1], s):
                st = [nobs, mean_x, ssqdm, comp_rem, 1.0 if unstable else 0.0]
                _remove_var(values[j], st)
                nobs, mean_x, ssqdm, comp_rem, unstable = st[0], st[1], st[2], st[3], st[4] != 0.0
            for j in range(end[i - 1], e):
                st = [nobs, mean_x, ssqdm, comp_add, 1.0 if unstable else 0.0]
                _add_var(values[j], st)
                nobs, mean_x, ssqdm, comp_add, unstable = st[0], st[1], st[2], st[3], st[4] != 0.0
        if recompute or unstable:
            mean_x = 0.0
            ssqdm = 0.0
            nobs = 0.0
            comp_add = 0.0
            comp_rem = 0.0
            for j in range(s, e):
                st = [nobs, mean_x, ssqdm, comp_add, 1.0 if unstable else 0.0]
                _add_var(values[j], st)
                nobs, mean_x, ssqdm, comp_add, unstable = st[0], st[1], st[2], st[3], st[4] != 0.0
            unstable = False
        if nobs >= float(minp) and nobs > float(ddof):
            out.append(ssqdm / (nobs - float(ddof)))
        else:
            out.append(NAN)
    return out

def roll_minmax(values: List[float], start: List[int], end: List[int], minp: int, is_max: bool) -> List[float]:
    N = len(start)
    out = List[float](N)
    for i in range(N):
        nobs = 0
        best = NAN
        for j in range(start[i], end[i]):
            v = values[j]
            if v == v:
                if nobs == 0 or (is_max and v > best) or (not is_max and v < best):
                    best = v
                nobs += 1
        out.append(best if nobs >= minp and nobs > 0 else NAN)
    return out

def roll_median(values: List[float], start: List[int], end: List[int], minp: int) -> List[float]:
    N = len(start)
    out = List[float](N)
    for i in range(N):
        w = [values[j] for j in range(start[i], end[i]) if values[j] == values[j]]
        if len(w) >= minp and len(w) > 0:
            w.sort()
            m = len(w)
            out.append(w[m // 2] if m % 2 == 1 else (w[m // 2 - 1] + w[m // 2]) / 2.0)
        else:
            out.append(NAN)
    return out

def rank_1d(x: List[float], method: str = "average", ascending: bool = True,
            na_option: str = "keep", pct: bool = False) -> List[float]:
    # pandas rank_1d for float data (NaN is missing)
    if method not in ("average", "min", "max", "first", "dense"):
        raise KeyError(method)
    if na_option not in ("keep", "top", "bottom"):
        raise ValueError("na_option must be one of 'keep', 'top', or 'bottom'")
    n = len(x)
    valid = [k for k in range(n) if not _isnan(x[k])]
    nas = [k for k in range(n) if _isnan(x[k])]
    if ascending:
        valid.sort(key=lambda k: x[k])
    else:
        valid.sort(key=lambda k: -x[k])
    # groups of tied positions in rank order
    groups = List[List[int]]()
    if na_option == "top" and len(nas) > 0:
        groups.append(nas)
    i = 0
    m = len(valid)
    while i < m:
        j = i
        while j + 1 < m and x[valid[j + 1]] == x[valid[i]]:
            j += 1
        groups.append(valid[i:j + 1])
        i = j + 1
    if na_option == "bottom" and len(nas) > 0:
        groups.append(nas)
    out = [NAN for _ in range(n)]
    pos = 0
    dense = 0
    for g in groups:
        dense += 1
        d = len(g)
        for t in range(d):
            r = 0.0
            if method == "average":
                r = float(pos + 1 + pos + d) / 2.0
            elif method == "min":
                r = float(pos + 1)
            elif method == "max":
                r = float(pos + d)
            elif method == "first":
                r = float(pos + t + 1)
            else:
                r = float(dense)
            out[g[t]] = r
        pos += d
    if pct:
        denom = float(dense) if method == "dense" else float(pos)
        for k in range(n):
            if not _isnan(out[k]):
                out[k] = out[k] / denom
    return out
