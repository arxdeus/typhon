# Ported from pandas 3.1.1 (BSD-3-Clause)

# Series and DataFrame. Values live in runtime-typed Columns (float64,
# int64, bool, str), so a DataFrame can hold columns of different dtypes
# while staying a single static Typhon type.

import numpy as np
from numpy import ndarray
import internal.static as static
from .scalar import Scalar, K_FLOAT, K_INT, K_BOOL, K_STR, K_NONE, K_DT, K_TD, NA_STR, _isnan
from . import _dt as DT
from ._dt import Timestamp, Timedelta
from .core import *
from .core import _dtype_name, _kind_of_dtype, _fixed_bounds, _lerp_quantile, _sorted_valid
from .index import Index
from . import strings as S
from numpy.random.legacy import check_random_state
from .nanalgos import nanskew_list, nankurt_list, quantile_sorted, mode_values, interpolate_linear, round_frac, infer_precision, linspace_list
from .window import ewm_kernel, ewmcov_kernel, roll_cov, roll_corr, roll_quantile, get_center_of_mass
from .fmt import frame_repr, series_repr, format_column, adjoin

def _as_label(x) -> Scalar:
    return Scalar.of(x)

def _opt_name(x) -> Optional[str]:
    if x is None:
        return None
    elif isinstance(x, str):
        return x
    elif isinstance(x, Optional[str]):
        return x
    else:
        return str(x)

def _label_name(s: Scalar) -> Optional[str]:
    if s.kind == K_NONE:
        return None
    return str(s)

def _make_index(index, n: int) -> Index:
    if index is None:
        return Index.range(n)
    elif isinstance(index, Index):
        if len(index) != n:
            raise ValueError(f"Length of values ({n}) does not match length of index ({len(index)})")
        return index.copy()
    else:
        idx = Index(index)
        if len(idx) != n:
            raise ValueError(f"Length of values ({n}) does not match length of index ({len(idx)})")
        return idx

def _norm_pos(k: int, n: int) -> int:
    if k < 0:
        k += n
    if k < 0 or k >= n:
        raise IndexError("single positional indexer is out-of-bounds")
    return k

def _mask_positions(mask) -> List[int]:
    res = List[int]()
    if isinstance(mask, Series):
        for k in range(len(mask)):
            if mask._col.isna_at(k):
                continue
            if mask._col.get(k).__bool__():
                res.append(k)
    else:
        k = 0
        for b in mask:
            if b:
                res.append(k)
            k += 1
    return res


def _csv_quote(s: str, sep: str) -> str:
    if sep in s or '"' in s or "\n" in s or "\r" in s:
        return '"' + s.replace('"', '""') + '"'
    return s

def _csv_q(s: str, sep: str, quoting: int) -> str:
    if quoting == 1:
        return '"' + s.replace('"', '""') + '"'
    if quoting == 3:
        return s
    return _csv_quote(s, sep)

# ---------------------------------------------------------------------------
# Series

class Series:
    _col: Column
    index: Index
    name: Optional[str]

    def __init__(self, data = None, index = None, dtype = None, name = None, copy: bool = False):
        if data is None:
            self._col = Column.floats(List[float]())
            self.index = _make_index(index, 0)
        elif isinstance(data, Series):
            self._col = data._col.copy()
            self.index = data.index.copy() if index is None else _make_index(index, len(data._col))
        elif isinstance(data, Dict):
            keys = list(data.keys())
            self._col = Column.of(list(data.values()))
            self.index = Index(keys) if index is None else _make_index(index, len(keys))
        elif isinstance(data, ndarray) or isinstance(data, List) or isinstance(data, Column):
            self._col = Column.of(data)
            self.index = _make_index(index, len(self._col))
        elif hasattr(data, "__iter__"):
            self._col = Column.of(list(data))
            self.index = _make_index(index, len(self._col))
        else:
            # scalar broadcast
            n = len(index) if index is not None else 1
            sc = Scalar.of(data)
            self._col = Column.full(sc.kind, n, sc)
            self.index = _make_index(index, n)
        if dtype is not None:
            self._col = self._col.astype(_kind_of_dtype(dtype))
        self.name = _opt_name(name)

    @staticmethod
    def _new(col: Column, index: Index, name: Optional[str]) -> Series:
        s = Series.__new__()
        s._col = col
        s.index = index
        s.name = name
        return s

    def _with(self, col: Column) -> Series:
        return Series._new(col, self.index, self.name)

    # -- attributes
    def __len__(self) -> int:
        return len(self._col)

    @property
    def size(self) -> int:
        return len(self._col)

    @property
    def shape(self) -> Tuple[int]:
        return (len(self._col),)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def dtype(self) -> str:
        return self._col.dtype

    @property
    def empty(self) -> bool:
        return len(self._col) == 0

    @property
    def values(self) -> ndarray[float, 1]:
        return self.to_numpy()

    def to_numpy(self, dtype: type = float) -> ndarray[dtype, 1]:
        n = len(self._col)
        a = np.empty(n, dtype=dtype)
        for k in range(n):
            if dtype is float:
                a[k] = self._col.getf(k)
            elif dtype is str:
                a[k] = str(self._col.get(k))
            elif dtype is bool:
                a[k] = self._col.get(k).__bool__()
            else:
                a[k] = dtype(int(self._col.get(k)))
        return a

    def tolist(self) -> List[Scalar]:
        return self._col.scalars()

    def to_list(self) -> List[Scalar]:
        return self.tolist()

    def to_dict(self) -> Dict[Scalar, Scalar]:
        d = Dict[Scalar, Scalar]()
        for k in range(len(self)):
            d[self.index[k]] = self._col.get(k)
        return d

    def copy(self, deep: bool = True) -> Series:
        return Series._new(self._col.copy(), self.index.copy(), self.name)

    def __iter__(self):
        for k in range(len(self._col)):
            yield self._col.get(k)

    def items(self):
        for k in range(len(self._col)):
            yield (self.index[k], self._col.get(k))

    def keys(self) -> Index:
        return self.index

    def __contains__(self, key) -> bool:
        return key in self.index

    # -- indexing
    def __getitem__(self, key):
        if isinstance(key, Slice):
            start, stop, step, _ = key.adjust_indices(len(self))
            return self._take(list(range(start, stop, step)))
        elif isinstance(key, Series) or (isinstance(key, ndarray[bool, 1]) or isinstance(key, List[bool])):
            return self._take(_mask_positions(key))
        elif isinstance(key, List):
            return self._take([self.index.get_loc(x) for x in key])
        elif isinstance(key, int) and not self._int_labels():
            return self._col.get(_norm_pos(key, len(self)))
        else:
            return self._col.get(self.index.get_loc(key))

    def _int_labels(self) -> bool:
        return self.index._col.kind == K_INT

    def __setitem__(self, key, value):
        if isinstance(key, Series) or (isinstance(key, ndarray[bool, 1]) or isinstance(key, List[bool])):
            v = Scalar.of(value)
            for k in _mask_positions(key):
                self._col.set(k, v)
        elif isinstance(key, Slice):
            start, stop, step, _ = key.adjust_indices(len(self))
            for k in range(start, stop, step):
                self._col.set(k, Scalar.of(value))
        else:
            k = self.index.get_loc_opt(Scalar.of(key))
            if k < 0:
                if isinstance(key, int) and not self._int_labels():
                    k = _norm_pos(key, len(self))
                else:
                    raise KeyError(str(key))
            self._col.set(k, Scalar.of(value))

    def _take(self, pos: List[int]) -> Series:
        return Series._new(self._col.take(pos), self.index.take(pos), self.name)

    @property
    def str(self):
        sm = StringMethods.__new__()
        sm.s = self
        return sm

    @property
    def dt(self):
        dp = DatetimeProperties.__new__()
        dp.s = self
        return dp

    def resample(self, rule: str):
        r = Resampler.__new__()
        r.s = self
        r.freq = rule
        return r

    @property
    def loc(self):
        return _SeriesLoc(self)

    @property
    def iloc(self):
        return _SeriesILoc(self)

    @property
    def at(self):
        return _SeriesLoc(self)

    @property
    def iat(self):
        return _SeriesILoc(self)

    def head(self, n: int = 5) -> Series:
        m = len(self)
        if n < 0:
            n = max(m + n, 0)
        return self._take(list(range(min(n, m))))

    def tail(self, n: int = 5) -> Series:
        m = len(self)
        if n < 0:
            n = max(m + n, 0)
        return self._take(list(range(max(m - n, 0), m)))

    def get(self, key, default = None):
        k = self.index.get_loc_opt(Scalar.of(key))
        if k < 0:
            return Scalar.of(default)
        return self._col.get(k)

    # -- conversion
    def astype(self, dtype) -> Series:
        return self._with(self._col.astype(_kind_of_dtype(dtype)))

    def rename(self, name) -> Series:
        return Series._new(self._col.copy(), self.index.copy(), _opt_name(name))

    def to_frame(self, name = None):
        nm = _opt_name(name)
        if nm is None and self.name is None:
            return DataFrame._new([self._col.copy()], Index([0]), self.index.copy())
        label = nm.__val__() if nm is not None else self.name.__val__()
        return DataFrame._new([self._col.copy()], Index.strs([label]), self.index.copy())

    def reset_index(self, drop: bool = False):
        f = self.to_frame()
        return f.reset_index(drop=drop)

    # -- reductions
    def sum(self, skipna: bool = True, min_count: int = 0) -> Scalar:
        if self._col.kind == K_INT or self._col.kind == K_BOOL:
            t = 0
            for x in self._col.i:
                t += x
            return Scalar(K_INT, 0.0, t, "")
        if self._col.kind == K_STR:
            return Scalar(K_STR, 0.0, 0, "".join(self._col.s))
        return Scalar(K_FLOAT, nansum(self._col, skipna, min_count), 0, "")

    def prod(self, skipna: bool = True) -> Scalar:
        if self._col.kind == K_INT or self._col.kind == K_BOOL:
            t = 1
            for x in self._col.i:
                t *= x
            return Scalar(K_INT, 0.0, t, "")
        return Scalar(K_FLOAT, nanprod(self._col, skipna), 0, "")

    def product(self, skipna: bool = True) -> Scalar:
        return self.prod(skipna)

    def mean(self, skipna: bool = True) -> float:
        return nanmean(self._col, skipna)

    def var(self, ddof: int = 1, skipna: bool = True) -> float:
        return nanvar(self._col, skipna, ddof)

    def std(self, ddof: int = 1, skipna: bool = True) -> float:
        return nanstd(self._col, skipna, ddof)

    def sem(self, ddof: int = 1, skipna: bool = True) -> float:
        from math import sqrt
        return nanstd(self._col, skipna, ddof) / sqrt(float(nancount(self._col)))

    def min(self, skipna: bool = True) -> Scalar:
        return nanminmax(self._col, skipna, False)

    def max(self, skipna: bool = True) -> Scalar:
        return nanminmax(self._col, skipna, True)

    def idxmin(self, skipna: bool = True) -> Scalar:
        return self._argext(False)

    def idxmax(self, skipna: bool = True) -> Scalar:
        return self._argext(True)

    def argmin(self) -> int:
        return self.index.get_loc(self._argext(False))

    def argmax(self) -> int:
        return self.index.get_loc(self._argext(True))

    def _argext(self, is_max: bool) -> Scalar:
        best = -1
        for k in range(len(self)):
            if self._col.isna_at(k):
                continue
            if best < 0:
                best = k
            else:
                v = self._col.get(k)
                b = self._col.get(best)
                if (is_max and v > b) or (not is_max and v < b):
                    best = k
        if best < 0:
            raise ValueError("attempt to get argmax of an empty sequence")
        return self.index[best]

    def median(self, skipna: bool = True) -> float:
        return nanmedian(self._col, skipna)

    def quantile(self, q: float = 0.5) -> float:
        return nanquantile(self._col, q)

    def quantile(self, q: List[float]) -> Series:
        return Series._new(Column.floats([nanquantile(self._col, x) for x in q]), Index(q), self.name)

    def count(self) -> int:
        return nancount(self._col)

    def nunique(self, dropna: bool = True) -> int:
        seen = Set[Scalar]()
        hasna = False
        for k in range(len(self)):
            if self._col.isna_at(k):
                hasna = True
            else:
                seen.add(self._col.get(k))
        return len(seen) + (1 if hasna and not dropna else 0)

    def unique(self) -> List[Scalar]:
        seen = Set[Scalar]()
        out = List[Scalar]()
        nan_seen = False
        for k in range(len(self)):
            v = self._col.get(k)
            if self._col.isna_at(k):
                if not nan_seen:
                    nan_seen = True
                    out.append(v)
            elif v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False, dropna: bool = True, bins: Optional[int] = None) -> Series:
        if bins is not None:
            # pandas: cut into equal-width bins, count per interval (all intervals kept)
            bl = _nbins_to_bins(self, bins, True)
            nbn = len(bl) - 1
            codes_s = _cut_core(self, bl, True, None, 3, False, "drop", True)
            mids = List[float]()
            for k in range(nbn):
                mids.append((bl[k] + bl[k + 1]) / 2.0)
            labs_s = _cut_core(Series(mids), bl, True, None, 3, True, "drop", False)
            cnts = [0 for _ in range(nbn)]
            for i in range(len(codes_s)):
                cv = codes_s._col.getf(i)
                if cv == cv:
                    cnts[int(cv)] += 1
            tot_n = 0
            for c in cnts:
                tot_n += c
            ordp = list(range(nbn))
            if sort:
                if ascending:
                    ordp.sort(key=lambda p: cnts[p])
                else:
                    ordp.sort(key=lambda p: -cnts[p])
            lab_keys = [labs_s._col.get(p) for p in ordp]
            idx_b = Index._wrap(Column.from_scalars(lab_keys), self.name)
            if normalize:
                return Series._new(Column.floats([float(cnts[p]) / float(len(self)) for p in ordp]), idx_b, "proportion")
            return Series._new(Column.ints([cnts[p] for p in ordp]), idx_b, "count")
        order = List[Scalar]()
        counts = Dict[Scalar, int]()
        for k in range(len(self)):
            if self._col.isna_at(k):
                continue
            v = self._col.get(k)
            if v not in counts:
                order.append(v)
                counts[v] = 0
            counts[v] += 1
        pos = list(range(len(order)))
        if sort:
            # stable sort by count (pandas keeps first-seen order for ties)
            if ascending:
                pos.sort(key=lambda p: counts[order[p]])
            else:
                pos.sort(key=lambda p: -counts[order[p]])
        keys = [order[p] for p in pos]
        cnt = [counts[order[p]] for p in pos]
        idx = Index._wrap(Column.from_scalars(keys), self.name)
        if normalize:
            tot = 0
            for c in cnt:
                tot += c
            return Series._new(Column.floats([float(c) / float(tot) for c in cnt]), idx, "proportion")
        return Series._new(Column.ints(cnt), idx, "count")

    def describe(self, percentiles: Optional[List[float]] = None) -> Series:
        if self._col.kind == K_STR:
            vc = self.value_counts()
            labels = ["count", "unique", "top", "freq"]
            vals = [Scalar.of(self.count()), Scalar.of(self.nunique()), vc.index[0], vc._col.get(0)]
            return Series._new(Column.from_scalars(vals).astype(K_STR), Index.strs(labels), self.name)
        pct = [0.25, 0.5, 0.75] if percentiles is None else percentiles
        a = _sorted_valid(self._col)
        vals = [float(nancount(self._col)), nanmean(self._col), nanstd(self._col), a[0] if len(a) else NAN]
        for p in pct:
            vals.append(_lerp_quantile(a, p))
        vals.append(a[-1] if len(a) else NAN)
        labels = ["count", "mean", "std", "min"] + [_pct_label(p) for p in pct] + ["max"]
        return Series._new(Column.floats(vals), Index.strs(labels), self.name)

    def _agg_name(self, f: str) -> Scalar:
        if f == "sum":
            return self.sum()
        elif f == "mean":
            return Scalar.of(self.mean())
        elif f == "min":
            return self.min()
        elif f == "max":
            return self.max()
        elif f == "std":
            return Scalar.of(self.std())
        elif f == "var":
            return Scalar.of(self.var())
        elif f == "median":
            return Scalar.of(self.median())
        elif f == "count":
            return Scalar.of(self.count())
        elif f == "nunique":
            return Scalar.of(self.nunique())
        elif f == "prod":
            return self.prod()
        elif f == "sem":
            return Scalar.of(self.sem())
        elif f == "skew":
            return Scalar.of(self.skew())
        elif f == "kurt":
            return Scalar.of(self.kurt())
        elif f == "first":
            return self._col.get(0)
        elif f == "last":
            return self._col.get(len(self) - 1)
        elif f == "size":
            return Scalar.of(len(self))
        raise AttributeError(f"'{f}' is not a valid function for 'Series' object")

    def agg(self, func: str) -> Scalar:
        return self._agg_name(func)

    def agg(self, func: List[str]) -> Series:
        out = List[Scalar]()
        for f in func:
            out.append(self._agg_name(f))
        return Series._new(Column.from_scalars(out), Index.strs(func), self.name)

    def aggregate(self, func):
        return self.agg(func)

    def sample(self, n: Optional[int] = None, frac: Optional[float] = None, replace: bool = False,
               random_state: Optional[int] = None):
        return self._take(_sample_pos(len(self), n, frac, replace, random_state))

    def unstack(self, level: int = -1):
        lv = self.index._lv
        if len(lv) != 2:
            raise ValueError("unstack requires a two-level MultiIndex")
        rl = 0 if level == 1 or level == -1 else 1
        cl = 1 - rl
        rg = _Grouper(lv[rl], self.index._lnames[rl])
        cg = _Grouper(lv[cl], self.index._lnames[cl])
        G = rg.ngroups
        H = cg.ngroups
        vals = [Scalar.none() for _ in range(G * H)]
        for i in range(len(self)):
            if rg.codes[i] >= 0 and cg.codes[i] >= 0:
                vals[cg.codes[i] * G + rg.codes[i]] = self._col.get(i)
        cols = List[Column]()
        for h in range(H):
            cols.append(Column.from_scalars(vals[h * G:(h + 1) * G]))
        return DataFrame._new(cols, Index._wrap(cg.key_col.copy(), self.index._lnames[cl]), rg.index())

    def all(self) -> bool:
        for x in self:
            if not x.__bool__():
                return False
        return True

    def any(self) -> bool:
        for x in self:
            if x.__bool__():
                return True
        return False

    # -- elementwise
    def _mapf(self, f) -> Series:
        return self._with(Column.floats([f(self._col.getf(k)) for k in range(len(self))]))

    def abs(self) -> Series:
        if self._col.kind == K_INT:
            return self._with(Column.ints([abs(x) for x in self._col.i]))
        return self._mapf(lambda x: abs(x))

    def round(self, decimals: int = 0) -> Series:
        if self._col.kind != K_FLOAT:
            return self.copy()
        a = np.round(np.array(self._col.f), decimals)
        return self._with(Column.floats([a[k] for k in range(len(a))]))

    def clip(self, lower = None, upper = None) -> Series:
        c = self._col.copy()
        for k in range(len(c)):
            v = c.get(k)
            if v.isna():
                continue
            if lower is not None:
                if v < Scalar.of(lower):
                    v = Scalar.of(lower)
            if upper is not None:
                if v > Scalar.of(upper):
                    v = Scalar.of(upper)
            c.set(k, v)
        return self._with(c)

    def isna(self) -> Series:
        return self._with(Column.bools([self._col.isna_at(k) for k in range(len(self))]))

    def isnull(self) -> Series:
        return self.isna()

    def notna(self) -> Series:
        return self._with(Column.bools([not self._col.isna_at(k) for k in range(len(self))]))

    def notnull(self) -> Series:
        return self.notna()

    def fillna(self, value) -> Series:
        c = self._col.copy()
        v = Scalar.of(value)
        for k in range(len(c)):
            if c.isna_at(k):
                c.set(k, v)
        if c.kind == K_FLOAT and v.kind == K_INT:
            pass
        return self._with(c)

    def ffill(self) -> Series:
        c = self._col.copy()
        for k in range(1, len(c)):
            if c.isna_at(k) and not c.isna_at(k - 1):
                c.set(k, c.get(k - 1))
        return self._with(c)

    def bfill(self) -> Series:
        c = self._col.copy()
        for k in range(len(c) - 2, -1, -1):
            if c.isna_at(k) and not c.isna_at(k + 1):
                c.set(k, c.get(k + 1))
        return self._with(c)

    def dropna(self) -> Series:
        return self._take([k for k in range(len(self)) if not self._col.isna_at(k)])

    def apply(self, func) -> Series:
        return self._with(Column.of([func(x) for x in self]))

    def map(self, arg) -> Series:
        if isinstance(arg, Dict):
            out = List[Scalar](len(self))
            for x in self:
                found = False
                for kk, vv in arg.items():
                    if Scalar.of(kk) == x:
                        out.append(Scalar.of(vv))
                        found = True
                        break
                if not found:
                    out.append(Scalar.none())
            return self._with(Column.from_scalars(out))
        elif isinstance(arg, Series):
            out = List[Scalar](len(self))
            for x in self:
                k = arg.index.get_loc_opt(x)
                out.append(arg._col.get(k) if k >= 0 else Scalar.none())
            return self._with(Column.from_scalars(out))
        else:
            return self.apply(arg)

    def replace(self, to_replace, value) -> Series:
        c = self._col.copy()
        a = Scalar.of(to_replace)
        b = Scalar.of(value)
        for k in range(len(c)):
            if c.get(k) == a:
                c.set(k, b)
        return self._with(c)

    def isin(self, values) -> Series:
        vs = Set[Scalar]()
        for v in values:
            vs.add(Scalar.of(v))
        return self._with(Column.bools([self._col.get(k) in vs for k in range(len(self))]))

    def between(self, left, right, inclusive: str = "both") -> Series:
        l = Scalar.of(left)
        r = Scalar.of(right)
        out = List[bool](len(self))
        for x in self:
            a = x >= l if inclusive in ("both", "left") else x > l
            b = x <= r if inclusive in ("both", "right") else x < r
            out.append(a and b)
        return self._with(Column.bools(out))

    def where(self, cond, other = None) -> Series:
        c = self._col.copy()
        pos = Set[int](_mask_positions(cond))
        o = Scalar.of(other)
        for k in range(len(c)):
            if k not in pos:
                c.set(k, o if other is not None else Scalar(K_FLOAT, NAN, 0, ""))
        return self._with(c)

    def mask(self, cond, other = None) -> Series:
        c = self._col.copy()
        o = Scalar.of(other)
        for k in _mask_positions(cond):
            c.set(k, o if other is not None else Scalar(K_FLOAT, NAN, 0, ""))
        return self._with(c)

    def cumsum(self, skipna: bool = True) -> Series:
        if self._col.kind == K_INT or self._col.kind == K_BOOL:
            out = List[int](len(self))
            t = 0
            for x in self._col.i:
                t += x
                out.append(t)
            return self._with(Column.ints(out))
        out = List[float](len(self))
        t = 0.0
        for x in self._col.f:
            if _isnan(x):
                out.append(NAN)
            else:
                t += x
                out.append(t)
        return self._with(Column.floats(out))

    def cumprod(self, skipna: bool = True) -> Series:
        if self._col.kind == K_INT or self._col.kind == K_BOOL:
            out = List[int](len(self))
            t = 1
            for x in self._col.i:
                t *= x
                out.append(t)
            return self._with(Column.ints(out))
        out = List[float](len(self))
        t = 1.0
        for x in self._col.f:
            if _isnan(x):
                out.append(NAN)
            else:
                t *= x
                out.append(t)
        return self._with(Column.floats(out))

    def _cumext(self, is_max: bool) -> Series:
        c = self._col.copy()
        best = Scalar.none()
        for k in range(len(c)):
            if c.isna_at(k):
                continue
            v = c.get(k)
            if best.kind == K_NONE or (is_max and v > best) or (not is_max and v < best):
                best = v
            c.set(k, best)
        return self._with(c)

    def cummax(self) -> Series:
        return self._cumext(True)

    def cummin(self) -> Series:
        return self._cumext(False)

    def diff(self, periods: int = 1) -> Series:
        n = len(self)
        out = List[float](n)
        for k in range(n):
            j = k - periods
            if j < 0 or j >= n:
                out.append(NAN)
            else:
                out.append(self._col.getf(k) - self._col.getf(j))
        return self._with(Column.floats(out))

    def shift(self, periods: int = 1, fill_value = None) -> Series:
        n = len(self)
        pos = [k - periods if 0 <= k - periods < n else -1 for k in range(n)]
        if fill_value is None:
            return self._with(self._col.take(pos))
        c = self._col.take([p if p >= 0 else 0 for p in pos])
        for k in range(n):
            if pos[k] < 0:
                c.set(k, Scalar.of(fill_value))
        return self._with(c)

    def pct_change(self, periods: int = 1) -> Series:
        n = len(self)
        out = List[float](n)
        for k in range(n):
            j = k - periods
            if j < 0 or j >= n:
                out.append(NAN)
            else:
                x = self._col.getf(k)
                y = self._col.getf(j)
                if y == 0.0:
                    out.append(NAN if x == 0.0 or x != x else (float("inf") if x > 0 else -float("inf")))
                else:
                    out.append(x / y - 1.0)
        return self._with(Column.floats(out))

    def rank(self, method: str = "average", ascending: bool = True, na_option: str = "keep", pct: bool = False) -> Series:
        return self._with(Column.floats(rank_1d(self._col.to_floats(), method, ascending, na_option, pct)))

    def sort_values(self, ascending: bool = True, na_position: str = "last") -> Series:
        return self._take(_argsort_col(self._col, ascending, na_position))

    def sort_index(self, ascending: bool = True) -> Series:
        return self._take(_argsort_col(self.index._col, ascending, "last"))

    def nlargest(self, n: int = 5) -> Series:
        return self.sort_values(ascending=False).head(n)

    def nsmallest(self, n: int = 5) -> Series:
        return self.sort_values().head(n)

    def drop(self, labels = None, index = None) -> Series:
        key = labels if labels is not None else index
        drop = Set[int]()
        if isinstance(key, List):
            for x in key:
                drop.add(self.index.get_loc(x))
        else:
            drop.add(self.index.get_loc(key))
        return self._take([k for k in range(len(self)) if k not in drop])

    def drop_duplicates(self, keep = "first", ignore_index: bool = False) -> Series:
        d = _dup_mask([repr(self._col.get(k)) for k in range(len(self))], keep)
        r = self._take([k for k in range(len(self)) if not d[k]])
        if ignore_index:
            r.index = Index.range(len(r))
        return r

    def duplicated(self, keep = "first") -> Series:
        return self._with(Column.bools(_dup_mask([repr(self._col.get(k)) for k in range(len(self))], keep)))

    def corr(self, other: Series, method: str = "pearson") -> float:
        a = self._col.to_floats()
        b = other._col.to_floats()
        if method == "spearman":
            return nancorr_spearman([a, b], 1)[0][1]
        return nancorr([a, b], False, 1)[0][1]

    def cov(self, other: Series, ddof: int = 1) -> float:
        return nancorr([self._col.to_floats(), other._col.to_floats()], True, 1)[0][1]

    def rolling(self, window: int, min_periods: Optional[int] = None, center: bool = False):
        return Rolling(self.to_frame(), window, min_periods if min_periods is not None else window, center, False, True)

    def expanding(self, min_periods: int = 1):
        return Rolling(self.to_frame(), 0, min_periods, False, True, True)

    def ewm(self, com: Optional[float] = None, span: Optional[float] = None, halflife: Optional[float] = None,
            alpha: Optional[float] = None, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False):
        return Ewm(self.to_frame(), get_center_of_mass(com, span, halflife, alpha), adjust, ignore_na, min_periods)

    def groupby(self, by):
        f = self.to_frame("__value__")
        keys = Series(by) if not isinstance(by, Series) else by
        return SeriesGroupBy(_Grouper(keys._col, keys.name), f._cols[0], self.name)

    # -- additional statistics
    def skew(self, skipna: bool = True) -> float:
        return nanskew_list(self._col.to_floats(), skipna)

    def kurt(self, skipna: bool = True) -> float:
        return nankurt_list(self._col.to_floats(), skipna)

    def kurtosis(self, skipna: bool = True) -> float:
        return self.kurt(skipna)

    def mode(self) -> Series:
        m = mode_values(self._col.to_floats())
        return Series._new(Column.floats(m), Index.range(len(m)), self.name)

    def quantile(self, q: float, interpolation: str) -> float:
        a = _sorted_valid(self._col)
        return quantile_sorted(a, q, interpolation)

    def interpolate(self, method: str = "linear", limit: int = -1, limit_direction: str = "forward",
                    limit_area: str = "") -> Series:
        if method != "linear" and method != "index" and method != "values":
            raise NotImplementedError("interpolate: only linear/index/values methods are supported")
        if method == "linear":
            return self._with(Column.floats(interpolate_linear(self._col.to_floats(), limit_direction, limit, limit_area)))
        # index/values: interpolate on the numeric index values
        v = self._col.to_floats()
        xs = List[float](len(v))
        for k in range(len(v)):
            xs.append(self.index._col.getf(k))
        return self._with(Column.floats(_interp_xy(v, xs, limit_direction, limit)))

    def autocorr(self, lag: int = 1) -> float:
        return self.corr(self.shift(lag))

    @property
    def hasnans(self) -> bool:
        for k in range(len(self)):
            if self._col.isna_at(k):
                return True
        return False

    @property
    def is_monotonic_increasing(self) -> bool:
        for k in range(1, len(self)):
            if self._col.get(k) < self._col.get(k - 1):
                return False
        return True

    @property
    def is_monotonic_decreasing(self) -> bool:
        for k in range(1, len(self)):
            if self._col.get(k) > self._col.get(k - 1):
                return False
        return True

    def argsort(self) -> Series:
        return self._with(Column.ints(_argsort_col(self._col, True, "last")))

    def searchsorted(self, value, side: str = "left") -> int:
        v = Scalar.of(value)
        lo = 0
        hi = len(self)
        while lo < hi:
            mid = (lo + hi) // 2
            x = self._col.get(mid)
            go_right = x < v if side == "left" else not (v < x)
            if go_right:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def repeat(self, repeats: int) -> Series:
        pos = List[int]()
        for k in range(len(self)):
            for _ in range(repeats):
                pos.append(k)
        return self._take(pos)

    def squeeze(self):
        return self

    def item(self) -> Scalar:
        if len(self) != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
        return self._col.get(0)

    def first_valid_index(self) -> Scalar:
        for k in range(len(self)):
            if not self._col.isna_at(k):
                return self.index[k]
        return Scalar.none()

    def last_valid_index(self) -> Scalar:
        for k in range(len(self) - 1, -1, -1):
            if not self._col.isna_at(k):
                return self.index[k]
        return Scalar.none()

    def add_prefix(self, prefix: str) -> Series:
        return self._with_index(Index.strs([prefix + str(self.index[k]) for k in range(len(self))], self.index.name))

    def add_suffix(self, suffix: str) -> Series:
        return self._with_index(Index.strs([str(self.index[k]) + suffix for k in range(len(self))], self.index.name))

    def _with_index(self, idx: Index) -> Series:
        return Series._new(self._col, idx, self.name)

    def pipe(self, func):
        return func(self)

    def combine_first(self, other: Series) -> Series:
        idx = _union_index(self.index, other.index)
        out = List[Scalar](len(idx))
        for k in range(len(idx)):
            a = self.index.get_loc_opt(idx[k])
            if a >= 0 and not self._col.isna_at(a):
                out.append(self._col.get(a))
            else:
                b = other.index.get_loc_opt(idx[k])
                out.append(other._col.get(b) if b >= 0 else Scalar.none())
        return Series._new(Column.from_scalars(out), idx, self.name)

    def update(self, other: Series):
        for k in range(len(self)):
            b = other.index.get_loc_opt(self.index[k])
            if b >= 0 and not other._col.isna_at(b):
                self._col.set(k, other._col.get(b))

    def eq(self, o) -> Series:
        return self._binop(o, 10)
    def ne(self, o) -> Series:
        return self._binop(o, 11)
    def lt(self, o) -> Series:
        return self._binop(o, 12)
    def le(self, o) -> Series:
        return self._binop(o, 13)
    def gt(self, o) -> Series:
        return self._binop(o, 14)
    def ge(self, o) -> Series:
        return self._binop(o, 15)
    def truediv(self, o) -> Series:
        return self._binop(o, 3)
    def divide(self, o) -> Series:
        return self._binop(o, 3)
    def multiply(self, o) -> Series:
        return self._binop(o, 2)
    def subtract(self, o) -> Series:
        return self._binop(o, 1)
    def floordiv(self, o) -> Series:
        return self._binop(o, 4)
    def mod(self, o) -> Series:
        return self._binop(o, 5)
    def pow(self, o) -> Series:
        return self._binop(o, 6)
    def radd(self, o) -> Series:
        return self._rbinop(o, 0)
    def rsub(self, o) -> Series:
        return self._rbinop(o, 1)
    def rmul(self, o) -> Series:
        return self._rbinop(o, 2)
    def rdiv(self, o) -> Series:
        return self._rbinop(o, 3)
    def rtruediv(self, o) -> Series:
        return self._rbinop(o, 3)
    def rpow(self, o) -> Series:
        return self._rbinop(o, 6)
    def __rfloordiv__(self, o) -> Series:
        return self._rbinop(o, 4)
    def __rmod__(self, o) -> Series:
        return self._rbinop(o, 5)
    def rfloordiv(self, o) -> Series:
        return self._rbinop(o, 4)
    def rmod(self, o) -> Series:
        return self._rbinop(o, 5)

    # -- arithmetic
    def _binop(self, other, op: int) -> Series:
        if isinstance(other, Series):
            if self.index.equals(other.index):
                return Series._new(_col_binop(self._col, other._col, op), self.index, self.name if self.name == other.name else None)
            idx = _union_index(self.index, other.index)
            a = self._col.take([self.index.get_loc_opt(idx[k]) for k in range(len(idx))])
            b = other._col.take([other.index.get_loc_opt(idx[k]) for k in range(len(idx))])
            return Series._new(_col_binop(a, b, op), idx, self.name if self.name == other.name else None)
        elif isinstance(other, ndarray) or isinstance(other, List):
            return Series._new(_col_binop(self._col, Column.of(other), op), self.index, self.name)
        else:
            return Series._new(_col_binop(self._col, Column.full(Scalar.of(other).kind, len(self), Scalar.of(other)), op), self.index, self.name)

    def _rbinop(self, other, op: int) -> Series:
        sc = Scalar.of(other)
        return Series._new(_col_binop(Column.full(sc.kind, len(self), sc), self._col, op), self.index, self.name)

    def __add__(self, o) -> Series:
        return self._binop(o, 0)
    def __sub__(self, o) -> Series:
        return self._binop(o, 1)
    def __mul__(self, o) -> Series:
        return self._binop(o, 2)
    def __truediv__(self, o) -> Series:
        return self._binop(o, 3)
    def __floordiv__(self, o) -> Series:
        return self._binop(o, 4)
    def __mod__(self, o) -> Series:
        return self._binop(o, 5)
    def __pow__(self, o) -> Series:
        return self._binop(o, 6)
    def __radd__(self, o) -> Series:
        return self._rbinop(o, 0)
    def __rsub__(self, o) -> Series:
        return self._rbinop(o, 1)
    def __rmul__(self, o) -> Series:
        return self._rbinop(o, 2)
    def __rtruediv__(self, o) -> Series:
        return self._rbinop(o, 3)
    def __rpow__(self, o) -> Series:
        return self._rbinop(o, 6)
    def add(self, o) -> Series:
        return self._binop(o, 0)
    def sub(self, o) -> Series:
        return self._binop(o, 1)
    def mul(self, o) -> Series:
        return self._binop(o, 2)
    def div(self, o) -> Series:
        return self._binop(o, 3)
    def __eq__(self, o) -> Series:
        return self._binop(o, 10)
    def __ne__(self, o) -> Series:
        return self._binop(o, 11)
    def __lt__(self, o) -> Series:
        return self._binop(o, 12)
    def __le__(self, o) -> Series:
        return self._binop(o, 13)
    def __gt__(self, o) -> Series:
        return self._binop(o, 14)
    def __ge__(self, o) -> Series:
        return self._binop(o, 15)
    def __and__(self, o) -> Series:
        return self._binop(o, 20)
    def __or__(self, o) -> Series:
        return self._binop(o, 21)
    def __xor__(self, o) -> Series:
        return self._binop(o, 22)
    def __invert__(self) -> Series:
        if self._col.kind == K_BOOL:
            return self._with(Column.bools([x == 0 for x in self._col.i]))
        return self._with(Column.ints([~x for x in self._col.i]))
    def __neg__(self) -> Series:
        if self._col.kind == K_FLOAT:
            return self._with(Column.floats([-x for x in self._col.f]))
        return self._with(Column.ints([-x for x in self._col.i]))
    def __abs__(self) -> Series:
        return self.abs()

    def equals(self, other: Series) -> bool:
        if len(self) != len(other) or self._col.kind != other._col.kind or not self.index.equals(other.index):
            return False
        for k in range(len(self)):
            if self._col.isna_at(k) and other._col.isna_at(k):
                continue
            if self._col.get(k) != other._col.get(k):
                return False
        return True

    # -- output
    def to_csv(self, path_or_buf: Optional[str] = None, sep: str = ",", na_rep: str = "",
               float_format: Optional[str] = None, header: bool = True, index: bool = True,
               index_label: Optional[str] = None, lineterminator: str = "\n", decimal: str = ".") -> str:
        nm = self.name if self.name is not None else "0"
        return self.to_frame(nm).to_csv(path_or_buf, sep, na_rep, float_format, None, header, index, index_label, lineterminator, decimal)

    def to_string(self) -> str:
        r = self.__repr__()
        return r[:r.rfind("\n")]

    def __repr__(self) -> str:
        if len(self) == 0:
            nm = f"Name: {self.name.__val__()}, " if self.name is not None else ""
            return f"Series([], {nm}dtype: {self.dtype})"
        return series_repr(self._col, self.index._col, self.index.name, self.name, self.dtype, True, self.index._lv, self.index._lnames)

    def __str__(self) -> str:
        return self.__repr__()

class _SeriesLoc:
    s: Series

    def __getitem__(self, key):
        if isinstance(key, Slice):
            return self.s._take(_label_slice(self.s.index, key))
        elif isinstance(key, Series) or (isinstance(key, ndarray[bool, 1]) or isinstance(key, List[bool])):
            return self.s._take(_mask_positions(key))
        elif isinstance(key, List):
            return self.s._take([self.s.index.get_loc(x) for x in key])
        else:
            return self.s._col.get(self.s.index.get_loc(key))

    def __setitem__(self, key, value):
        k = self.s.index.get_loc_opt(Scalar.of(key))
        if k < 0:
            # enlargement
            self.s.index = self.s.index.append(Index([key]))
            self.s._col = self.s._col.extend(Column.full(Scalar.of(value).kind, 1, Scalar.of(value)))
        else:
            self.s._col.set(k, Scalar.of(value))

class _SeriesILoc:
    s: Series

    def __getitem__(self, key):
        n = len(self.s)
        if isinstance(key, int):
            return self.s._col.get(_norm_pos(key, n))
        elif isinstance(key, Slice):
            start, stop, step, _ = key.adjust_indices(n)
            return self.s._take(list(range(start, stop, step)))
        elif (isinstance(key, ndarray[bool, 1]) or isinstance(key, List[bool])):
            return self.s._take(_mask_positions(key))
        else:
            return self.s._take([_norm_pos(int(k), n) for k in key])

    def __setitem__(self, key: int, value):
        self.s._col.set(_norm_pos(key, len(self.s)), Scalar.of(value))

def _label_slice(idx: Index, key: Slice) -> List[int]:
    n = len(idx)
    a = 0
    b = n
    if key.start is not None:
        a = idx.get_loc(key.start)
    if key.stop is not None:
        b = idx.get_loc(key.stop) + 1
    step = 1
    if key.step is not None:
        step = key.step
    return list(range(a, b, step))

def _union_index(a: Index, b: Index) -> Index:
    keys = a.tolist()
    seen = Set[Scalar](keys)
    for k in range(len(b)):
        if b[k] not in seen:
            keys.append(b[k])
            seen.add(b[k])
    keys.sort()
    return Index._wrap(Column.from_scalars(keys), a.name if a.name == b.name else None)

def _dup_mask(keys: List[str], keep) -> List[bool]:
    # pandas duplicated: keep='first' | 'last' | False
    mode = 0
    if isinstance(keep, bool):
        if keep:
            raise ValueError("keep must be either \"first\", \"last\" or False")
        mode = 2
    elif keep == "first":
        mode = 0
    elif keep == "last":
        mode = 1
    else:
        raise ValueError("keep must be either \"first\", \"last\" or False")
    n = len(keys)
    out = [False for _ in range(n)]
    if mode == 2:
        cnt = Dict[str, int]()
        for k in keys:
            cnt[k] = cnt.get(k, 0) + 1
        return [cnt[k] > 1 for k in keys]
    seen = Set[str]()
    order = list(range(n)) if mode == 0 else list(range(n - 1, -1, -1))
    for i in order:
        out[i] = keys[i] in seen
        seen.add(keys[i])
    return out

def _interp_xy(v: List[float], xs: List[float], limit_direction: str, limit: int) -> List[float]:
    n = len(v)
    out = [x for x in v]
    valid = [i for i in range(n) if v[i] == v[i]]
    if len(valid) == 0:
        return out
    first = valid[0]
    last = valid[len(valid) - 1]
    for i in range(n):
        if v[i] == v[i]:
            continue
        if i < first:
            if limit_direction == "forward":
                continue
            out[i] = v[first]
            continue
        if i > last:
            if limit_direction == "backward":
                continue
            out[i] = v[last]
            continue
        lo = i - 1
        while lo >= 0 and v[lo] != v[lo]:
            lo -= 1
        hi = i + 1
        while hi < n and v[hi] != v[hi]:
            hi += 1
        out[i] = v[lo] + (v[hi] - v[lo]) / (xs[hi] - xs[lo]) * (xs[i] - xs[lo])
    return out

def _col_binop(a: Column, b: Column, op: int) -> Column:
    n = len(a)
    if len(b) != n:
        raise ValueError("Lengths must match")
    if op >= 20:
        out = List[bool](n)
        for k in range(n):
            x = a.get(k).__bool__() and not a.isna_at(k)
            y = b.get(k).__bool__() and not b.isna_at(k)
            if op == 20:
                out.append(x and y)
            elif op == 21:
                out.append(x or y)
            else:
                out.append(x != y)
        return Column.bools(out)
    if op >= 10:
        out = List[bool](n)
        for k in range(n):
            x = a.get(k)
            y = b.get(k)
            r = False
            if op == 10:
                r = x == y
            elif op == 11:
                r = not (x == y)
            elif op == 12:
                r = x < y
            elif op == 13:
                r = x <= y
            elif op == 14:
                r = x > y
            else:
                r = x >= y
            out.append(r)
        return Column.bools(out)
    if a.kind == K_DT or a.kind == K_TD or b.kind == K_DT or b.kind == K_TD:
        sc = List[Scalar](n)
        for k in range(n):
            sc.append(a.get(k)._arith(b.get(k), op) if not (a.isna_at(k) or b.isna_at(k)) else Scalar.none())
        r = Column.from_scalars(sc)
        if r.kind == K_FLOAT and (a.kind == K_DT or a.kind == K_TD or b.kind == K_DT or b.kind == K_TD):
            # all-NaT result keeps a datetime-like dtype
            pass
        return r
    if a.kind == K_STR or b.kind == K_STR:
        if op == 0 and a.kind == K_STR and b.kind == K_STR:
            return Column.strs([a.s[k] + b.s[k] if a.s[k] != NA_STR and b.s[k] != NA_STR else NA_STR for k in range(n)])
        raise TypeError("unsupported operand type(s) for str column")
    if a.kind != K_FLOAT and b.kind != K_FLOAT and op != 3 and op != 6 and not ((op == 4 or op == 5) and 0 in b.i):
        out = List[int](n)
        for k in range(n):
            x = a.i[k]
            y = b.i[k]
            if op == 0:
                out.append(x + y)
            elif op == 1:
                out.append(x - y)
            elif op == 2:
                out.append(x * y)
            elif op == 4:
                out.append(x // y)
            else:
                out.append(x % y)
        return Column.ints(out)
    if a.kind != K_FLOAT and b.kind != K_FLOAT and op == 6:
        ok = True
        for y in b.i:
            if y < 0:
                ok = False
        if ok:
            return Column.ints([a.i[k] ** b.i[k] for k in range(n)])
    out = List[float](n)
    for k in range(n):
        x = a.getf(k)
        y = b.getf(k)
        r = 0.0
        if op == 0:
            r = x + y
        elif op == 1:
            r = x - y
        elif op == 2:
            r = x * y
        elif op == 3:
            if y == 0.0:
                r = NAN if x == 0.0 or x != x else (float("inf") if x > 0 else -float("inf"))
            else:
                r = x / y
        elif op == 4:
            r = np.floor_divide(x, y) if y != 0.0 else (NAN if x == 0.0 or x != x else (float("inf") if x > 0 else -float("inf")))
        elif op == 5:
            r = np.mod(x, y) if y != 0.0 else NAN
        else:
            r = x ** y
        out.append(r)
    return Column.floats(out)

def _argsort_col(c: Column, ascending: bool, na_position: str) -> List[int]:
    n = len(c)
    valid = [k for k in range(n) if not c.isna_at(k)]
    nas = [k for k in range(n) if c.isna_at(k)]
    if c.kind == K_FLOAT:
        valid.sort(key=lambda k: c.f[k])
    elif c.kind == K_STR:
        valid.sort(key=lambda k: c.s[k])
    else:
        valid.sort(key=lambda k: c.i[k])
    if not ascending:
        # stable descending, like pandas (equal keys keep original order)
        valid = _stable_reverse(valid, c)
    if na_position == "first":
        return nas + valid
    return valid + nas


def _sample_pos(length: int, n: Optional[int], frac: Optional[float], replace: bool,
                random_state: Optional[int]) -> List[int]:
    # pandas: rs.choice(axis_length, size=n, replace=replace) on a RandomState
    if n is not None and frac is not None:
        raise ValueError("Please enter a value for `frac` OR `n`, not both")
    k = 1
    if frac is not None:
        f = frac.__val__()
        if f > 1.0 and not replace:
            raise ValueError("Replace has to be set to `True` when upsampling the population `frac` > 1.")
        k = int(round(f * length))
    elif n is not None:
        k = n.__val__()
    if k < 0:
        raise ValueError("A negative number of rows requested. Please provide positive value.")
    rs = check_random_state(random_state)
    idx = rs.choice(length, size=k, replace=replace)
    return [int(idx[i]) for i in range(k)]

def _stable_reverse(v: List[int], c: Column) -> List[int]:
    out = List[int](len(v))
    i = len(v) - 1
    while i >= 0:
        j = i
        while j - 1 >= 0 and c.get(v[j - 1]) == c.get(v[i]):
            j -= 1
        for t in range(j, i + 1):
            out.append(v[t])
        i = j - 1
    return out

# ---------------------------------------------------------------------------
# DataFrame

class DataFrame:
    _cols: List[Column]
    columns: Index
    index: Index

    def __init__(self, data = None, index = None, columns = None, dtype = None, copy: bool = False):
        self._cols = List[Column]()
        if data is None:
            self.columns = Index.strs(List[str]())
            if columns is not None:
                self.columns = Index(columns)
                for _ in range(len(self.columns)):
                    self._cols.append(Column.floats(List[float]()))
            self.index = _make_index(index, 0)
        elif isinstance(data, DataFrame):
            self._cols = [c.copy() for c in data._cols]
            self.columns = data.columns.copy()
            self.index = data.index.copy()
        elif isinstance(data, ndarray):
            if static.len(data.shape) != 2:
                compile_error("pandas: DataFrame needs a 2-D array")
            n, m = data.shape
            for j in range(m):
                self._cols.append(Column.of(data[:, j]))
            self.columns = Index.range(m) if columns is None else Index(columns)
            if len(self.columns) != m:
                raise ValueError(f"Shape of passed values is ({n}, {m}), indices imply ({n}, {len(self.columns)})")
            self.index = _make_index(index, n)
        elif isinstance(data, Dict):
            names = List[str]()
            n = -1
            for k, v in data.items():
                c = Column.of(v)
                if n >= 0 and len(c) != n:
                    raise ValueError("All arrays must be of the same length")
                n = len(c)
                names.append(str(k))
                self._cols.append(c)
            idx_from = None
            for k, v in data.items():
                if isinstance(v, Series):
                    idx_from = v.index
                break
            self.columns = Index.strs(names)
            if n < 0:
                n = 0
            if index is None and idx_from is not None:
                self.index = idx_from.copy()
            else:
                self.index = _make_index(index, n)
            if columns is not None:
                sel = Index(columns)
                self._cols = [self._cols[self.columns.get_loc(sel[j])] for j in range(len(sel))]
                self.columns = sel
        elif isinstance(data, List):
            n = len(data)
            m = 0
            if n > 0:
                m = len(data[0])
            for j in range(m):
                self._cols.append(Column.of([row[j] for row in data]))
            self.columns = Index.range(m) if columns is None else Index(columns)
            self.index = _make_index(index, n)
        else:
            compile_error("pandas: unsupported DataFrame data")
        if dtype is not None:
            kind = _kind_of_dtype(dtype)
            self._cols = [c.astype(kind) for c in self._cols]

    @staticmethod
    def _new(cols: List[Column], columns: Index, index: Index):
        df = DataFrame.__new__()
        df._cols = cols
        df.columns = columns
        df.index = index
        return df

    # -- attributes
    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.index), len(self._cols))

    def __len__(self) -> int:
        return len(self.index)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return len(self.index) * len(self._cols)

    @property
    def empty(self) -> bool:
        return len(self.index) == 0 or len(self._cols) == 0

    @property
    def dtypes(self) -> Series:
        return Series._new(Column.strs([c.dtype for c in self._cols]), self.columns.copy(), None)

    @property
    def values(self) -> ndarray[float, 2]:
        return self.to_numpy()

    @property
    def T(self):
        return self.transpose()

    @property
    def loc(self):
        return _FrameLoc(self)

    @property
    def iloc(self):
        return _FrameILoc(self)

    @property
    def at(self):
        return _FrameLoc(self)

    @property
    def iat(self):
        return _FrameILoc(self)

    def keys(self) -> Index:
        return self.columns

    def __contains__(self, key) -> bool:
        return key in self.columns

    def __iter__(self):
        for k in range(len(self.columns)):
            yield self.columns[k]

    def to_numpy(self, dtype: type = float) -> ndarray[dtype, 2]:
        n, m = self.shape
        a = np.empty((n, m), dtype=dtype)
        for j in range(m):
            c = self._cols[j]
            for i in range(n):
                if dtype is float:
                    a[i, j] = c.getf(i)
                elif dtype is str:
                    a[i, j] = str(c.get(i))
                elif dtype is bool:
                    a[i, j] = c.get(i).__bool__()
                else:
                    a[i, j] = dtype(int(c.get(i)))
        return a

    def copy(self, deep: bool = True):
        return DataFrame._new([c.copy() for c in self._cols], self.columns.copy(), self.index.copy())

    def _col_pos(self, key) -> int:
        return self.columns.get_loc(key)

    def _series(self, j: int) -> Series:
        return Series._new(self._cols[j], self.index, _label_name(self.columns[j]))

    def _row(self, i: int) -> Series:
        return Series._new(Column.from_scalars([c.get(i) for c in self._cols]), self.columns, _label_name(self.index[i]))

    def _take_rows(self, pos: List[int]):
        return DataFrame._new([c.take(pos) for c in self._cols], self.columns.copy(), self.index.take(pos))

    def _take_cols(self, pos: List[int]):
        return DataFrame._new([self._cols[j].copy() for j in pos], self.columns.take(pos), self.index.copy())

    # -- item access
    def __getitem__(self, key):
        if isinstance(key, Optional):
            return self[key.__val__()]
        elif isinstance(key, Slice):
            start, stop, step, _ = key.adjust_indices(len(self))
            return self._take_rows(list(range(start, stop, step)))
        elif isinstance(key, Series) or (isinstance(key, ndarray[bool, 1]) or isinstance(key, List[bool])):
            return self._take_rows(_mask_positions(key))
        elif isinstance(key, List) or isinstance(key, Index):
            return self._take_cols([self._col_pos(k) for k in key])
        else:
            return self._series(self._col_pos(key))

    def __setitem__(self, key, value):
        n = len(self.index)
        col = Column.floats(List[float]())
        if isinstance(value, Series):
            if value.index.equals(self.index):
                col = value._col.copy()
            else:
                col = value._col.take([value.index.get_loc_opt(self.index[k]) for k in range(n)])
        elif isinstance(value, ndarray) or isinstance(value, List) or isinstance(value, Column):
            col = Column.of(value)
            if len(col) != n:
                raise ValueError(f"Length of values ({len(col)}) does not match length of index ({n})")
        else:
            sc = Scalar.of(value)
            col = Column.full(sc.kind, n, sc)
        k = self.columns.get_loc_opt(Scalar.of(key))
        if k >= 0:
            self._cols[k] = col
        else:
            if len(self._cols) == 0 and n == 0:
                self.index = Index.range(len(col))
            self._cols.append(col)
            self.columns = self.columns.append(Index([key]))

    def __delitem__(self, key):
        k = self._col_pos(key)
        self._cols.pop(k)
        self.columns = self.columns.take([j for j in range(len(self.columns)) if j != k])

    def __getattr__(self, name: Literal[str]):
        return self[name]

    def get(self, key, default = None):
        if key in self.columns:
            return self[key]
        return default

    def pop(self, key) -> Series:
        s = self[key]
        self.__delitem__(key)
        return s

    def insert(self, loc: int, column, value):
        self[column] = value
        k = len(self._cols) - 1
        c = self._cols.pop(k)
        self._cols.insert(loc, c)
        order = list(range(len(self.columns) - 1))
        order.insert(loc, k)
        self.columns = self.columns.take(order)

    def head(self, n: int = 5):
        m = len(self)
        if n < 0:
            n = max(m + n, 0)
        return self._take_rows(list(range(min(n, m))))

    def tail(self, n: int = 5):
        m = len(self)
        if n < 0:
            n = max(m + n, 0)
        return self._take_rows(list(range(max(m - n, 0), m)))

    def iterrows(self):
        for i in range(len(self.index)):
            yield (self.index[i], self._row(i))

    def items(self):
        for j in range(len(self._cols)):
            yield (self.columns[j], self._series(j))

    def itertuples(self, index: bool = True):
        for i in range(len(self.index)):
            r = List[Scalar]()
            if index:
                r.append(self.index[i])
            for c in self._cols:
                r.append(c.get(i))
            yield r

    # -- structure
    def drop(self, labels = None, axis: int = 0, index = None, columns = None):
        if labels is not None:
            if axis == 1:
                return self.drop(columns=labels)
            return self.drop(index=labels)
        df = self
        if columns is not None:
            dropc = Set[int]()
            if isinstance(columns, List):
                for x in columns:
                    dropc.add(self._col_pos(x))
            else:
                dropc.add(self._col_pos(columns))
            df = df._take_cols([j for j in range(len(df._cols)) if j not in dropc])
        if index is not None:
            dropr = Set[int]()
            if isinstance(index, List):
                for x in index:
                    for p in df.index.get_locs(x):
                        dropr.add(p)
            else:
                for p in df.index.get_locs(index):
                    dropr.add(p)
            df = df._take_rows([i for i in range(len(df)) if i not in dropr])
        if columns is None and index is None:
            return self.copy()
        return df

    def rename(self, columns = None, index = None):
        df = self.copy()
        if columns is not None:
            labels = List[Scalar]()
            for j in range(len(df.columns)):
                lab = df.columns[j]
                newl = lab
                for k, v in columns.items():
                    if Scalar.of(k) == lab:
                        newl = Scalar.of(v)
                labels.append(newl)
            df.columns = Index._wrap(Column.from_scalars(labels), df.columns.name)
        if index is not None:
            labels = List[Scalar]()
            for j in range(len(df.index)):
                lab = df.index[j]
                newl = lab
                for k, v in index.items():
                    if Scalar.of(k) == lab:
                        newl = Scalar.of(v)
                labels.append(newl)
            df.index = Index._wrap(Column.from_scalars(labels), df.index.name)
        return df

    def astype(self, dtype):
        if isinstance(dtype, Dict):
            df = self.copy()
            for k, v in dtype.items():
                j = df._col_pos(k)
                df._cols[j] = df._cols[j].astype(_kind_of_dtype(v))
            return df
        kind = _kind_of_dtype(dtype)
        return DataFrame._new([c.astype(kind) for c in self._cols], self.columns.copy(), self.index.copy())

    def reset_index(self, drop: bool = False):
        if drop:
            return DataFrame._new([c.copy() for c in self._cols], self.columns.copy(), Index.range(len(self)))
        if len(self.index._lv) > 0:
            cols = [c.copy() for c in self.index._lv] + [c.copy() for c in self._cols]
            labels = List[Scalar]()
            for j in range(len(self.index._lv)):
                nm = self.index._lnames[j]
                labels.append(Scalar.of(nm.__val__() if nm is not None else f"level_{j}"))
            labels = labels + self.columns.tolist()
            return DataFrame._new(cols, Index._wrap(Column.from_scalars(labels)), Index.range(len(self)))
        name = self.index.name if self.index.name is not None else "index"
        cols = [self.index._col.copy()] + [c.copy() for c in self._cols]
        labels = [Scalar.of(name)] + self.columns.tolist()
        return DataFrame._new(cols, Index._wrap(Column.from_scalars(labels)), Index.range(len(self)))

    def set_index(self, keys, drop: bool = True):
        j = self._col_pos(keys)
        idx = Index._wrap(self._cols[j].copy(), _label_name(self.columns[j]))
        if drop:
            df = self._take_cols([k for k in range(len(self._cols)) if k != j])
            df.index = idx
            return df
        df = self.copy()
        df.index = idx
        return df

    def set_index(self, keys: List[str], drop: bool = True):
        if len(keys) == 1:
            return self.set_index(keys[0], drop)
        pos = [self._col_pos(k) for k in keys]
        idx = Index._multi([self._cols[j].copy() for j in pos], [_label_name(self.columns[j]) for j in pos])
        if drop:
            df = self._take_cols([k for k in range(len(self._cols)) if k not in pos])
            df.index = idx
            return df
        df = self.copy()
        df.index = idx
        return df

    def sort_values(self, by, ascending = True, na_position: str = "last"):
        keys = List[int]()
        if isinstance(by, List):
            for b in by:
                keys.append(self._col_pos(b))
        else:
            keys.append(self._col_pos(by))
        asc = List[bool]()
        if isinstance(ascending, List):
            asc = ascending
        else:
            asc = [ascending for _ in keys]
        pos = list(range(len(self)))
        # successive stable sorts from the last key
        for t in range(len(keys) - 1, -1, -1):
            c = self._cols[keys[t]].take(pos)
            order = _argsort_col(c, asc[t], na_position)
            pos = [pos[o] for o in order]
        return self._take_rows(pos)

    def sort_index(self, ascending: bool = True, axis: int = 0):
        if axis == 1:
            return self._take_cols(_argsort_col(self.columns._col, ascending, "last"))
        return self._take_rows(_argsort_col(self.index._col, ascending, "last"))

    def transpose(self):
        n, m = self.shape
        cols = [Column.from_scalars([self._cols[j].get(i) for j in range(m)]) for i in range(n)]
        return DataFrame._new(cols, self.index.copy(), self.columns.copy())

    def select_dtypes(self, include = None, exclude = None):
        def matches(kind: int, spec) -> bool:
            if isinstance(spec, List):
                for s in spec:
                    if matches(kind, s):
                        return True
                return False
            elif isinstance(spec, str):
                if spec == "number":
                    return kind == K_FLOAT or kind == K_INT
                return _kind_of_dtype(spec) == kind
            else:
                return _kind_of_dtype(spec) == kind
        keep = List[int]()
        for j in range(len(self._cols)):
            k = self._cols[j].kind
            ok = True
            if include is not None:
                ok = matches(k, include)
            if exclude is not None and matches(k, exclude):
                ok = False
            if ok:
                keep.append(j)
        return self._take_cols(keep)

    # -- missing values
    def isna(self):
        return self._map_cols(lambda c: Column.bools([c.isna_at(k) for k in range(len(c))]))

    def isnull(self):
        return self.isna()

    def notna(self):
        return self._map_cols(lambda c: Column.bools([not c.isna_at(k) for k in range(len(c))]))

    def notnull(self):
        return self.notna()

    def fillna(self, value):
        df = self.copy()
        for j in range(len(df._cols)):
            v = Scalar.of(value)
            if isinstance(value, Dict):
                found = False
                for k, x in value.items():
                    if Scalar.of(k) == df.columns[j]:
                        v = Scalar.of(x)
                        found = True
                if not found:
                    continue
            c = df._cols[j]
            for i in range(len(c)):
                if c.isna_at(i):
                    c.set(i, v)
        return df

    def dropna(self, axis: int = 0, how: str = "any", subset: Optional[List[str]] = None):
        if axis == 1:
            keep = List[int]()
            for j in range(len(self._cols)):
                c = self._cols[j]
                cnt = nancount(c)
                if (how == "any" and cnt == len(c)) or (how == "all" and cnt > 0):
                    keep.append(j)
            return self._take_cols(keep)
        cols = list(range(len(self._cols)))
        if subset is not None:
            cols = [self._col_pos(s) for s in subset]
        keep = List[int]()
        for i in range(len(self)):
            nna = 0
            for j in cols:
                if self._cols[j].isna_at(i):
                    nna += 1
            if (how == "any" and nna == 0) or (how == "all" and nna < len(cols)):
                keep.append(i)
        return self._take_rows(keep)

    def _map_cols(self, f):
        return DataFrame._new([f(c) for c in self._cols], self.columns.copy(), self.index.copy())

    def _map_series(self, f):
        return DataFrame._new([f(self._series(j))._col for j in range(len(self._cols))], self.columns.copy(), self.index.copy())

    # -- reductions
    def _numeric(self, numeric_only: bool) -> List[int]:
        res = List[int]()
        for j in range(len(self._cols)):
            if self._cols[j].kind != K_STR:
                res.append(j)
            elif not numeric_only:
                raise TypeError(f"could not convert column '{self.columns[j]}' of dtype str to numeric")
        return res

    def _reduce_f(self, f, axis: int, numeric_only: bool) -> Series:
        cols = self._numeric(numeric_only)
        if axis == 1:
            out = List[float](len(self))
            for i in range(len(self)):
                row = Column.floats([self._cols[j].getf(i) for j in cols])
                out.append(f(row))
            return Series._new(Column.floats(out), self.index.copy(), None)
        return Series._new(Column.floats([f(self._cols[j]) for j in cols]), self.columns.take(cols), None)

    def _reduce_s(self, f, axis: int, numeric_only: bool) -> Series:
        cols = self._numeric(numeric_only)
        if axis == 1:
            out = List[Scalar](len(self))
            for i in range(len(self)):
                row = Column.from_scalars([self._cols[j].get(i) for j in cols])
                out.append(f(row))
            return Series._new(Column.from_scalars(out), self.index.copy(), None)
        return Series._new(Column.from_scalars([f(self._cols[j]) for j in cols]), self.columns.take(cols), None)

    def sum(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, min_count: int = 0) -> Series:
        return self._reduce_s(lambda c: Series._new(c, Index.range(len(c)), None).sum(skipna, min_count), axis, numeric_only)

    def prod(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return self._reduce_s(lambda c: Series._new(c, Index.range(len(c)), None).prod(skipna), axis, numeric_only)

    def product(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return self.prod(axis, skipna, numeric_only)

    def mean(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return self._reduce_f(lambda c: nanmean(c, skipna), axis, numeric_only)

    def median(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return self._reduce_f(lambda c: nanmedian(c, skipna), axis, numeric_only)

    def var(self, axis: int = 0, skipna: bool = True, ddof: int = 1, numeric_only: bool = False) -> Series:
        return self._reduce_f(lambda c: nanvar(c, skipna, ddof), axis, numeric_only)

    def std(self, axis: int = 0, skipna: bool = True, ddof: int = 1, numeric_only: bool = False) -> Series:
        return self._reduce_f(lambda c: nanstd(c, skipna, ddof), axis, numeric_only)

    def sem(self, axis: int = 0, skipna: bool = True, ddof: int = 1, numeric_only: bool = False) -> Series:
        from math import sqrt
        return self._reduce_f(lambda c: nanstd(c, skipna, ddof) / sqrt(float(nancount(c))), axis, numeric_only)

    def min(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return self._reduce_s(lambda c: nanminmax(c, skipna, False), axis, numeric_only)

    def max(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return self._reduce_s(lambda c: nanminmax(c, skipna, True), axis, numeric_only)

    def quantile(self, q: float = 0.5, axis: int = 0, numeric_only: bool = False) -> Series:
        s = self._reduce_f(lambda c: nanquantile(c, q), axis, numeric_only)
        s.name = str(q)
        return s

    def count(self, axis: int = 0) -> Series:
        if axis == 1:
            return Series._new(Column.ints([len(self._cols) - sum(1 for c in self._cols if c.isna_at(i)) for i in range(len(self))]), self.index.copy(), None)
        return Series._new(Column.ints([nancount(c) for c in self._cols]), self.columns.copy(), None)

    def nunique(self) -> Series:
        return Series._new(Column.ints([self._series(j).nunique() for j in range(len(self._cols))]), self.columns.copy(), None)

    def idxmax(self) -> Series:
        return Series._new(Column.from_scalars([self._series(j).idxmax() for j in range(len(self._cols))]), self.columns.copy(), None)

    def idxmin(self) -> Series:
        return Series._new(Column.from_scalars([self._series(j).idxmin() for j in range(len(self._cols))]), self.columns.copy(), None)

    def all(self) -> Series:
        return Series._new(Column.bools([self._series(j).all() for j in range(len(self._cols))]), self.columns.copy(), None)

    def any(self) -> Series:
        return Series._new(Column.bools([self._series(j).any() for j in range(len(self._cols))]), self.columns.copy(), None)


    # -- additional statistics / aliases
    def skew(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return self._reduce_f(lambda c: nanskew_list(c.to_floats(), skipna), axis, numeric_only)

    def kurt(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return self._reduce_f(lambda c: nankurt_list(c.to_floats(), skipna), axis, numeric_only)

    def kurtosis(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return self.kurt(axis, skipna, numeric_only)

    def quantile(self, q: float, interpolation: str, axis: int = 0, numeric_only: bool = False) -> Series:
        s = self._reduce_f(lambda c: quantile_sorted(_sorted_valid(c), q, interpolation), axis, numeric_only)
        s.name = str(q)
        return s

    def mode(self):
        cols = List[Column]()
        mx = 0
        modes = List[List[float]]()
        for c in self._cols:
            m = mode_values(c.to_floats())
            modes.append(m)
            if len(m) > mx:
                mx = len(m)
        for m in modes:
            out = List[float](mx)
            for k in range(mx):
                out.append(m[k] if k < len(m) else NAN)
            cols.append(Column.floats(out))
        return DataFrame._new(cols, self.columns.copy(), Index.range(mx))

    def interpolate(self, method: str = "linear", limit: int = -1, limit_direction: str = "forward", limit_area: str = ""):
        return self._map_series(lambda s: s.interpolate(method, limit, limit_direction, limit_area))

    def eq(self, o):
        return self._binop(o, 10)
    def ne(self, o):
        return self._binop(o, 11)
    def lt(self, o):
        return self._binop(o, 12)
    def le(self, o):
        return self._binop(o, 13)
    def gt(self, o):
        return self._binop(o, 14)
    def ge(self, o):
        return self._binop(o, 15)
    def truediv(self, o):
        return self._binop(o, 3)
    def divide(self, o):
        return self._binop(o, 3)
    def multiply(self, o):
        return self._binop(o, 2)
    def subtract(self, o):
        return self._binop(o, 1)
    def floordiv(self, o):
        return self._binop(o, 4)
    def mod(self, o):
        return self._binop(o, 5)
    def pow(self, o):
        return self._binop(o, 6)
    def radd(self, o):
        return self._rbinop(o, 0)
    def rsub(self, o):
        return self._rbinop(o, 1)
    def rmul(self, o):
        return self._rbinop(o, 2)
    def rtruediv(self, o):
        return self._rbinop(o, 3)
    def rdiv(self, o):
        return self._rbinop(o, 3)
    def rpow(self, o):
        return self._rbinop(o, 6)

    def add_prefix(self, prefix: str):
        df = self.copy()
        df.columns = Index.strs([prefix + str(self.columns[j]) for j in range(len(self._cols))], self.columns.name)
        return df

    def add_suffix(self, suffix: str):
        df = self.copy()
        df.columns = Index.strs([str(self.columns[j]) + suffix for j in range(len(self._cols))], self.columns.name)
        return df

    def pipe(self, func):
        return func(self)

    def assign(self, **kwargs):
        df = self.copy()
        for k, v in kwargs.items():
            df[k] = v
        return df

    def first_valid_index(self) -> Scalar:
        for i in range(len(self)):
            for c in self._cols:
                if not c.isna_at(i):
                    return self.index[i]
        return Scalar.none()

    def last_valid_index(self) -> Scalar:
        for i in range(len(self) - 1, -1, -1):
            for c in self._cols:
                if not c.isna_at(i):
                    return self.index[i]
        return Scalar.none()

    def squeeze(self):
        return self

    def melt(self, id_vars: Optional[List[str]] = None, value_vars: Optional[List[str]] = None,
             var_name: str = "variable", value_name: str = "value", ignore_index: bool = True):
        ids = id_vars if id_vars is not None else List[str]()
        vals = List[str]()
        if value_vars is not None:
            vals = value_vars
        else:
            for j in range(len(self._cols)):
                nm = str(self.columns[j])
                if nm not in ids:
                    vals.append(nm)
        n = len(self)
        out_cols = List[Column]()
        labels = List[str]()
        pos = List[int]()
        for _ in vals:
            for i in range(n):
                pos.append(i)
        for k in ids:
            out_cols.append(self._cols[self._col_pos(k)].take(pos))
            labels.append(k)
        names = List[str]()
        for v in vals:
            for _ in range(n):
                names.append(v)
        out_cols.append(Column.strs(names))
        labels.append(var_name)
        allv = List[Scalar]()
        for v in vals:
            c = self._cols[self._col_pos(v)]
            for i in range(n):
                allv.append(c.get(i))
        out_cols.append(Column.from_scalars(allv))
        labels.append(value_name)
        return DataFrame._new(out_cols, Index.strs(labels), Index.range(len(pos)))

    def pivot(self, index: str, columns: str, values: str):
        return pivot_table(self, values, index, columns, "last", False)

    def pivot_table(self, values: str, index: str, columns: Optional[str] = None, aggfunc: str = "mean"):
        return pivot_table(self, values, index, columns, aggfunc, True)

    def cumsum(self):
        return self._map_series(lambda s: s.cumsum())

    def cumprod(self):
        return self._map_series(lambda s: s.cumprod())

    def cummax(self):
        return self._map_series(lambda s: s.cummax())

    def cummin(self):
        return self._map_series(lambda s: s.cummin())

    def abs(self):
        return self._map_series(lambda s: s.abs())

    def round(self, decimals: int = 0):
        return self._map_series(lambda s: s.round(decimals))

    def clip(self, lower = None, upper = None):
        return self._map_series(lambda s: s.clip(lower, upper))

    def diff(self, periods: int = 1):
        return self._map_series(lambda s: s.diff(periods))

    def shift(self, periods: int = 1, fill_value = None):
        return self._map_series(lambda s: s.shift(periods, fill_value))

    def pct_change(self, periods: int = 1):
        return self._map_series(lambda s: s.pct_change(periods))

    def rank(self, axis: int = 0, method: str = "average", ascending: bool = True, na_option: str = "keep", pct: bool = False):
        if axis == 1:
            n = len(self)
            m = len(self._cols)
            rows = [rank_1d([c.getf(i) for c in self._cols], method, ascending, na_option, pct) for i in range(n)]
            return DataFrame._new([Column.floats([rows[i][j] for i in range(n)]) for j in range(m)], self.columns.copy(), self.index.copy())
        if axis != 0:
            raise ValueError(f"No axis named {axis} for object type DataFrame")
        return self._map_series(lambda s: s.rank(method=method, ascending=ascending, na_option=na_option, pct=pct))

    def ffill(self):
        return self._map_series(lambda s: s.ffill())

    def bfill(self):
        return self._map_series(lambda s: s.bfill())

    def replace(self, to_replace, value):
        return self._map_series(lambda s: s.replace(to_replace, value))

    def isin(self, values):
        return self._map_series(lambda s: s.isin(values))

    def where(self, cond: DataFrame, other = None):
        return DataFrame._new([self._series(j).where(cond._series(j), other)._col for j in range(len(self._cols))], self.columns.copy(), self.index.copy())

    def mask(self, cond: DataFrame, other = None):
        return DataFrame._new([self._series(j).mask(cond._series(j), other)._col for j in range(len(self._cols))], self.columns.copy(), self.index.copy())

    def duplicated(self, subset = None, keep = "first") -> Series:
        cols = List[int]()
        if subset is None:
            cols = list(range(len(self._cols)))
        elif isinstance(subset, List):
            for x in subset:
                cols.append(self.columns.get_loc(x))
        else:
            cols.append(self.columns.get_loc(subset))
        keys = ["\x01".join([repr(self._cols[j].get(i)) for j in cols]) for i in range(len(self))]
        return Series._new(Column.bools(_dup_mask(keys, keep)), self.index.copy(), None)

    def drop_duplicates(self, subset = None, keep = "first", ignore_index: bool = False):
        d = self.duplicated(subset, keep)
        r = self._take_rows([i for i in range(len(self)) if not d._col.i[i]])
        if ignore_index:
            r.index = Index.range(len(r))
        return r

    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False, dropna: bool = True) -> Series:
        if len(self._cols) != 1:
            raise NotImplementedError("DataFrame.value_counts is only supported for one column")
        s = self._series(0).value_counts(normalize, sort, ascending, dropna)
        s.index.name = _label_name(self.columns[0])
        return s

    def apply(self, func, axis: int = 0):
        # func receives a Series (column for axis=0, row for axis=1) and
        # returns a scalar (result: Series) or a Series (result: DataFrame)
        if axis == 1:
            rows = [func(self._row(i)) for i in range(len(self))]
            if isinstance(rows, List[Series]):
                return concat_series_axis1(rows).transpose()._set_index(self.index)
            else:
                return Series._new(Column.of(rows), self.index.copy(), None)
        res = [func(self._series(j)) for j in range(len(self._cols))]
        if isinstance(res, List[Series]):
            return DataFrame._new([r._col for r in res], self.columns.copy(), res[0].index.copy() if len(res) else Index.range(0))
        else:
            return Series._new(Column.of(res), self.columns.copy(), None)

    def _set_index(self, idx: Index):
        self.index = idx.copy()
        return self

    def map(self, func):
        return self._map_series(lambda s: s.apply(func))

    def applymap(self, func):
        return self.map(func)

    def describe(self, percentiles: Optional[List[float]] = None, include: Optional[str] = None):
        pct = [0.25, 0.5, 0.75] if percentiles is None else percentiles
        want_obj = include is not None and (include.__val__() == "all" or include.__val__() == "object")
        want_num = include is None or include.__val__() == "all" or include.__val__() == "number"
        cols = List[int]()
        for j in range(len(self._cols)):
            k = self._cols[j].kind
            is_num = k == K_FLOAT or k == K_INT
            if (is_num and want_num) or (k == K_STR and want_obj):
                cols.append(j)
        any_num = False
        any_str = False
        for j in cols:
            if self._cols[j].kind == K_STR:
                any_str = True
            else:
                any_num = True
        labels = List[str]()
        if any_str:
            labels.extend(["count", "unique", "top", "freq"])
        if any_num or not any_str:
            labels.extend(["count", "mean", "std", "min"] + [_pct_label(p) for p in pct] + ["max"])
        # union in pandas order: count unique top freq mean std min pcts max
        order = List[str]()
        for l in labels:
            if l not in order:
                order.append(l)
        out = List[Column]()
        for j in cols:
            c = self._cols[j]
            sc = Series._new(c, Index.range(len(c)), None).describe(pct)
            vals = List[Scalar]()
            for l in order:
                p = sc.index.get_loc_opt(Scalar.of(l))
                vals.append(sc._col.get(p) if p >= 0 else Scalar.none())
            out.append(Column.from_scalars(vals))
        return DataFrame._new(out, self.columns.take(cols), Index.strs(order))

    def stack(self):
        # DataFrame -> Series with a (row, column) MultiIndex; missing values are dropped
        n = len(self)
        m = len(self._cols)
        rows = List[int]()
        cs = List[int]()
        vals = List[Scalar]()
        for i in range(n):
            for j in range(m):
                if not self._cols[j].isna_at(i):
                    rows.append(i)
                    cs.append(j)
                    vals.append(self._cols[j].get(i))
        idx = Index._multi([self.index._col.take(rows), self.columns._col.take(cs)], [self.index.name, self.columns.name])
        return Series._new(Column.from_scalars(vals), idx, None)

    def nlargest(self, n: int, columns, keep: str = "first"):
        asc = False
        return self.sort_values(columns, ascending=asc).head(n)

    def nsmallest(self, n: int, columns, keep: str = "first"):
        return self.sort_values(columns).head(n)

    def agg(self, func: str):
        return self._reduce_s(lambda c: Series._new(c, Index.range(len(c)), None).agg(func), 0, True)

    def agg(self, func: List[str]):
        cols = self._numeric(True)
        out = List[Column]()
        for j in cols:
            sc = Series._new(self._cols[j], Index.range(len(self._cols[j])), None)
            out.append(sc.agg(func)._col)
        return DataFrame._new(out, self.columns.take(cols), Index.strs(func))

    def aggregate(self, func):
        return self.agg(func)

    def sample(self, n: Optional[int] = None, frac: Optional[float] = None, replace: bool = False,
               random_state: Optional[int] = None):
        return self._take_rows(_sample_pos(len(self), n, frac, replace, random_state))

    def corr(self, method: str = "pearson", min_periods: int = 1, numeric_only: bool = False):
        cols = self._numeric(numeric_only)
        mats = [self._cols[j].to_floats() for j in cols]
        r = List[List[float]]()
        if method == "pearson":
            r = nancorr(mats, False, min_periods)
        elif method == "spearman":
            r = nancorr_spearman(mats, min_periods)
        else:
            raise ValueError(f"method must be either 'pearson', 'spearman', or a callable, '{method}' was supplied")
        lab = self.columns.take(cols)
        return DataFrame._new([Column.floats(r[j]) for j in range(len(cols))], lab, lab.copy())

    def cov(self, min_periods: Optional[int] = None, ddof: int = 1, numeric_only: bool = False):
        cols = self._numeric(numeric_only)
        mats = [self._cols[j].to_floats() for j in cols]
        lab = self.columns.take(cols)
        K = len(cols)
        no_nan = True
        for m in mats:
            for x in m:
                if _isnan(x):
                    no_nan = False
        res = List[List[float]]()
        if no_nan and min_periods is None:
            # np.cov path
            res = _np_cov(mats, ddof)
        else:
            res = nancorr(mats, True, 1 if min_periods is None else min_periods)
        return DataFrame._new([Column.floats(res[j]) for j in range(K)], lab, lab.copy())

    def corrwith(self, other: Series) -> Series:
        cols = self._numeric(True)
        return Series._new(Column.floats([self._series(j).corr(other) for j in cols]), self.columns.take(cols), None)

    def groupby(self, by, sort: bool = True, dropna: bool = True):
        if isinstance(by, Series):
            return DataFrameGroupBy(self, _Grouper(by._col, by.name, sort), -1)
        elif isinstance(by, List):
            if len(by) != 1:
                kc = List[Column]()
                kn = List[Optional[str]]()
                kp = List[int]()
                for b in by:
                    jj = self._col_pos(b)
                    kc.append(self._cols[jj])
                    kn.append(_label_name(self.columns[jj]))
                    kp.append(jj)
                gb = DataFrameGroupBy(self, _Grouper.multi(kc, kn, sort), -1)
                gb.key_cols = kp
                return gb
            j = self._col_pos(by[0])
            return DataFrameGroupBy(self, _Grouper(self._cols[j], _label_name(self.columns[j]), sort), j)
        else:
            j = self._col_pos(by)
            return DataFrameGroupBy(self, _Grouper(self._cols[j], _label_name(self.columns[j]), sort), j)

    def rolling(self, window: int, min_periods: Optional[int] = None, center: bool = False):
        return Rolling(self, window, min_periods if min_periods is not None else window, center, False, False)

    def expanding(self, min_periods: int = 1):
        return Rolling(self, 0, min_periods, False, True, False)

    def ewm(self, com: Optional[float] = None, span: Optional[float] = None, halflife: Optional[float] = None,
            alpha: Optional[float] = None, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False):
        return Ewm(self, get_center_of_mass(com, span, halflife, alpha), adjust, ignore_na, min_periods)

    def merge(self, right: DataFrame, how: str = "inner", on = None, left_on = None, right_on = None, suffixes: Tuple[str, str] = ("_x", "_y"), indicator: bool = False):
        return merge(self, right, how=how, on=on, left_on=left_on, right_on=right_on, suffixes=suffixes, indicator=indicator)

    def equals(self, other: DataFrame) -> bool:
        if self.shape != other.shape or not self.columns.equals(other.columns) or not self.index.equals(other.index):
            return False
        for j in range(len(self._cols)):
            if not self._series(j).equals(other._series(j)):
                return False
        return True

    # -- arithmetic
    def _binop(self, other, op: int):
        if isinstance(other, DataFrame):
            return DataFrame._new([_col_binop(self._cols[j], other._cols[other._col_pos(self.columns[j])], op) for j in range(len(self._cols))], self.columns.copy(), self.index.copy())
        elif isinstance(other, Series):
            # align on columns
            return DataFrame._new([_col_binop(self._cols[j], Column.full(other[self.columns[j]].kind, len(self), other.loc[self.columns[j]]), op) for j in range(len(self._cols))], self.columns.copy(), self.index.copy())
        else:
            sc = Scalar.of(other)
            return DataFrame._new([_col_binop(c, Column.full(sc.kind, len(c), sc), op) for c in self._cols], self.columns.copy(), self.index.copy())

    def _rbinop(self, other, op: int):
        sc = Scalar.of(other)
        return DataFrame._new([_col_binop(Column.full(sc.kind, len(c), sc), c, op) for c in self._cols], self.columns.copy(), self.index.copy())

    def __add__(self, o):
        return self._binop(o, 0)
    def __sub__(self, o):
        return self._binop(o, 1)
    def __mul__(self, o):
        return self._binop(o, 2)
    def __truediv__(self, o):
        return self._binop(o, 3)
    def __floordiv__(self, o):
        return self._binop(o, 4)
    def __mod__(self, o):
        return self._binop(o, 5)
    def __pow__(self, o):
        return self._binop(o, 6)
    def __radd__(self, o):
        return self._rbinop(o, 0)
    def __rsub__(self, o):
        return self._rbinop(o, 1)
    def __rmul__(self, o):
        return self._rbinop(o, 2)
    def __rtruediv__(self, o):
        return self._rbinop(o, 3)
    def add(self, o):
        return self._binop(o, 0)
    def sub(self, o):
        return self._binop(o, 1)
    def mul(self, o):
        return self._binop(o, 2)
    def div(self, o):
        return self._binop(o, 3)
    def __eq__(self, o):
        return self._binop(o, 10)
    def __ne__(self, o):
        return self._binop(o, 11)
    def __lt__(self, o):
        return self._binop(o, 12)
    def __le__(self, o):
        return self._binop(o, 13)
    def __gt__(self, o):
        return self._binop(o, 14)
    def __ge__(self, o):
        return self._binop(o, 15)
    def __and__(self, o):
        return self._binop(o, 20)
    def __or__(self, o):
        return self._binop(o, 21)
    def __neg__(self):
        return self._map_series(lambda s: -s)
    def __invert__(self):
        return self._map_series(lambda s: ~s)

    # -- output
    def to_csv(self, path_or_buf: Optional[str] = None, sep: str = ",", na_rep: str = "",
               float_format: Optional[str] = None, columns: Optional[List[str]] = None,
               header: bool = True, index: bool = True, index_label: Optional[str] = None,
               lineterminator: str = "\n", decimal: str = ".", mode: str = "w",
               quoting: int = 0, header_names: Optional[List[str]] = None) -> str:
        # quoting: 0 = QUOTE_MINIMAL, 1 = QUOTE_ALL, 3 = QUOTE_NONE; mode "a" appends
        df = self
        if columns is not None:
            df = self[columns.__val__()]
        lines = List[str]()
        if header:
            hdr = List[str]()
            if index:
                il = index_label if index_label is not None else (df.index.name if df.index.name is not None else "")
                hdr.append(_csv_q(il, sep, quoting))
            for j in range(len(df.columns)):
                hn = str(df.columns[j]) if header_names is None else header_names.__val__()[j]
                hdr.append(_csv_q(hn, sep, quoting))
            lines.append(sep.join(hdr))
        n = len(df)
        ready = [csv_ready(c, na_rep) for c in df._cols]
        iready = csv_ready(df.index._col, na_rep)
        for i in range(n):
            row = List[str](len(df._cols) + 1)
            if index:
                row.append(_csv_q(iready.fmt_csv(i, float_format, na_rep, decimal), sep, quoting))
            for c in ready:
                row.append(_csv_q(c.fmt_csv(i, float_format, na_rep, decimal), sep, quoting))
            lines.append(sep.join(row))
        out = lineterminator.join(lines) + lineterminator
        if path_or_buf is not None:
            with open(path_or_buf.__val__(), "a" if mode == "a" else "w") as f:
                f.write(out)
        return out

    def to_dict(self) -> Dict[str, Dict[Scalar, Scalar]]:
        d = Dict[str, Dict[Scalar, Scalar]]()
        for j in range(len(self._cols)):
            d[str(self.columns[j])] = self._series(j).to_dict()
        return d

    def to_string(self) -> str:
        return frame_repr(self._cols, [str(self.columns[j]) for j in range(len(self.columns))], self.index._col, self.index.name, len(self), False)

    def __repr__(self) -> str:
        n, m = self.shape
        if m == 0 or n == 0:
            cl = ", ".join([str(c) for c in self.columns])
            il = ", ".join([str(c) for c in self.index])
            return f"Empty DataFrame\nColumns: [{cl}]\nIndex: [{il}]"
        return frame_repr(self._cols, [str(self.columns[j]) for j in range(len(self.columns))], self.index._col, self.index.name, n, True, self.index._lv, self.index._lnames)

    def __str__(self) -> str:
        return self.__repr__()

    def info(self):
        print(f"<class 'pandas.DataFrame'>")
        print(f"RangeIndex: {len(self)} entries, 0 to {len(self) - 1}")
        print(f"Data columns (total {len(self._cols)} columns):")
        for j in range(len(self._cols)):
            print(f" {j}   {self.columns[j]}   {nancount(self._cols[j])} non-null   {self._cols[j].dtype}")

def _pct_label(p: float) -> str:
    v = p * 100.0
    if v == float(int(v)):
        return str(int(v)) + "%"
    s = str(v)
    return s + "%"

def _np_cov(mats: List[List[float]], ddof: int) -> List[List[float]]:
    K = len(mats)
    N = len(mats[0]) if K > 0 else 0
    X = np.empty((K, N), float)
    for j in range(K):
        for i in range(N):
            X[j, i] = mats[j][i]
    C = np.cov(X, ddof=ddof)
    return [[C[a, b] for b in range(K)] for a in range(K)]

class _FrameLoc:
    df: DataFrame

    def _rows(self, key) -> List[int]:
        if isinstance(key, Slice):
            return _label_slice(self.df.index, key)
        elif isinstance(key, Series) or (isinstance(key, ndarray[bool, 1]) or isinstance(key, List[bool])):
            return _mask_positions(key)
        elif isinstance(key, List):
            return [self.df.index.get_loc(x) for x in key]
        else:
            raise TypeError("unsupported row indexer")

    def _colsel(self, key) -> List[int]:
        if isinstance(key, Slice):
            return _label_slice(self.df.columns, key)
        elif isinstance(key, Series) or (isinstance(key, ndarray[bool, 1]) or isinstance(key, List[bool])):
            return _mask_positions(key)
        elif isinstance(key, List):
            return [self.df._col_pos(x) for x in key]
        else:
            raise TypeError("unsupported column indexer")

    def __getitem__(self, key):
        df = self.df
        if isinstance(key, Tuple):
            r = key[0]
            c = key[1]
            if (isinstance(r, Slice) or isinstance(r, List) or isinstance(r, Series) or (isinstance(r, ndarray[bool, 1]) or isinstance(r, List[bool]))):
                if isinstance(c, Slice) or isinstance(c, List):
                    return df._take_rows(self._rows(r))._take_cols(self._colsel(c))
                else:
                    return df._series(df._col_pos(c))._take(self._rows(r))
            else:
                i = df.index.get_loc(r)
                if isinstance(c, Slice) or isinstance(c, List):
                    sel = self._colsel(c)
                    return Series._new(Column.from_scalars([df._cols[j].get(i) for j in sel]), df.columns.take(sel), _label_name(df.index[i]))
                else:
                    return df._cols[df._col_pos(c)].get(i)
        elif isinstance(key, Slice) or isinstance(key, List) or isinstance(key, Series) or (isinstance(key, ndarray[bool, 1]) or isinstance(key, List[bool])):
            return df._take_rows(self._rows(key))
        else:
            return df._row(df.index.get_loc(key))

    def __setitem__(self, key, value):
        df = self.df
        if isinstance(key, Tuple):
            r = key[0]
            c = key[1]
            k = df.columns.get_loc_opt(Scalar.of(c))
            if k < 0:
                df[c] = Scalar(K_FLOAT, NAN, 0, "")
                k = len(df._cols) - 1
            if isinstance(r, Slice) or isinstance(r, List) or isinstance(r, Series) or (isinstance(r, ndarray[bool, 1]) or isinstance(r, List[bool])):
                for i in self._rows(r):
                    df._cols[k].set(i, Scalar.of(value))
            else:
                df._cols[k].set(df.index.get_loc(r), Scalar.of(value))
        else:
            i = df.index.get_loc(key)
            for j in range(len(df._cols)):
                df._cols[j].set(i, Scalar.of(value))

class _FrameILoc:
    df: DataFrame

    def _sel(self, key, n: int) -> List[int]:
        if isinstance(key, Slice):
            start, stop, step, _ = key.adjust_indices(n)
            return list(range(start, stop, step))
        elif (isinstance(key, ndarray[bool, 1]) or isinstance(key, List[bool])):
            return _mask_positions(key)
        elif isinstance(key, List) or isinstance(key, ndarray):
            return [_norm_pos(int(k), n) for k in key]
        else:
            raise TypeError("unsupported positional indexer")

    def __getitem__(self, key):
        df = self.df
        n, m = df.shape
        if isinstance(key, Tuple):
            r = key[0]
            c = key[1]
            if isinstance(r, int):
                i = _norm_pos(r, n)
                if isinstance(c, int):
                    return df._cols[_norm_pos(c, m)].get(i)
                else:
                    sel = self._sel(c, m)
                    return Series._new(Column.from_scalars([df._cols[j].get(i) for j in sel]), df.columns.take(sel), _label_name(df.index[i]))
            else:
                rows = self._sel(r, n)
                if isinstance(c, int):
                    return df._series(_norm_pos(c, m))._take(rows)
                else:
                    return df._take_rows(rows)._take_cols(self._sel(c, m))
        elif isinstance(key, int):
            return df._row(_norm_pos(key, n))
        else:
            return df._take_rows(self._sel(key, n))

    def __setitem__(self, key, value):
        df = self.df
        n, m = df.shape
        if isinstance(key, Tuple):
            df._cols[_norm_pos(key[1], m)].set(_norm_pos(key[0], n), Scalar.of(value))
        else:
            i = _norm_pos(key, n)
            for j in range(m):
                df._cols[j].set(i, Scalar.of(value))

# ---------------------------------------------------------------------------
# GroupBy

class _Grouper:
    codes: List[int]
    keys: List[Scalar]
    key_col: Column
    name: Optional[str]
    levels: List[Column]               # multi-key: one key column per level (group order)
    lnames: List[Optional[str]]

    def __init__(self, col: Column, name: Optional[str], sort: bool = True):
        first = Dict[Scalar, int]()
        order = List[Scalar]()
        n = len(col)
        for i in range(n):
            if col.isna_at(i):
                continue
            v = col.get(i)
            if v not in first:
                first[v] = len(order)
                order.append(v)
        keys = order.copy()
        if sort:
            keys.sort()
        pos = Dict[Scalar, int]()
        for g in range(len(keys)):
            pos[keys[g]] = g
        codes = List[int](n)
        for i in range(n):
            if col.isna_at(i):
                codes.append(-1)
            else:
                codes.append(pos[col.get(i)])
        self.codes = codes
        self.keys = keys
        self.key_col = Column.from_scalars(keys) if col.kind != K_STR else Column.strs([k.s for k in keys])
        if (col.kind == K_INT or col.kind == K_DT or col.kind == K_TD) and len(keys) > 0:
            self.key_col = Column(col.kind, List[float](), [k.i for k in keys], List[str]())
        self.name = name
        self.levels = List[Column]()
        self.lnames = List[Optional[str]]()

    @staticmethod
    def multi(cols: List[Column], names: List[Optional[str]], sort: bool = True) -> _Grouper:
        # group by several keys: combine the per-key group codes lexicographically
        n = len(cols[0])
        subs = List[_Grouper]()
        for c in cols:
            subs.append(_Grouper(c, None, sort))
        tup = Dict[str, int]()
        order = List[List[int]]()
        raw = List[int](n)
        for i in range(n):
            ok = True
            t = List[int]()
            for sg in subs:
                if sg.codes[i] < 0:
                    ok = False
                t.append(sg.codes[i])
            if not ok:
                raw.append(-1)
                continue
            key = ",".join([str(x) for x in t])
            if key not in tup:
                tup[key] = len(order)
                order.append(t)
            raw.append(tup[key])
        perm = list(range(len(order)))
        if sort:
            perm.sort(key=lambda g: order[g])
        newpos = [0 for _ in range(len(order))]
        for r in range(len(perm)):
            newpos[perm[r]] = r
        g = _Grouper.__new__()
        g.codes = [newpos[c] if c >= 0 else -1 for c in raw]
        g.name = None
        g.lnames = names
        g.levels = List[Column]()
        for j in range(len(subs)):
            pick = List[int]()
            for r in range(len(perm)):
                pick.append(order[perm[r]][j])
            g.levels.append(subs[j].key_col.take(pick))
        keys = List[Scalar]()
        for r in range(len(perm)):
            keys.append(Scalar.of("\x1f".join([str(lv.get(r)) for lv in g.levels])))
        g.keys = keys
        g.key_col = Column.from_scalars(keys)
        return g

    @property
    def ngroups(self) -> int:
        return len(self.keys)

    def index(self) -> Index:
        if len(self.levels) > 0:
            return Index._multi([c.copy() for c in self.levels], self.lnames.copy())
        return Index._wrap(self.key_col.copy(), self.name)

    def members(self) -> List[List[int]]:
        out = [List[int]() for _ in range(self.ngroups)]
        for i in range(len(self.codes)):
            if self.codes[i] >= 0:
                out[self.codes[i]].append(i)
        return out

def _agg_col(g: _Grouper, c: Column, how: str, ddof: int = 1) -> Column:
    G = g.ngroups
    if how == "count":
        cnt = [0 for _ in range(G)]
        for i in range(len(c)):
            if g.codes[i] >= 0 and not c.isna_at(i):
                cnt[g.codes[i]] += 1
        return Column.ints(cnt)
    if how == "size":
        cnt = [0 for _ in range(G)]
        for i in range(len(c)):
            if g.codes[i] >= 0:
                cnt[g.codes[i]] += 1
        return Column.ints(cnt)
    if how == "nunique":
        sets = [Set[Scalar]() for _ in range(G)]
        for i in range(len(c)):
            if g.codes[i] >= 0 and not c.isna_at(i):
                sets[g.codes[i]].add(c.get(i))
        return Column.ints([len(s) for s in sets])
    if how == "min" or how == "max" or how == "first" or how == "last":
        res = [Scalar.none() for _ in range(G)]
        for i in range(len(c)):
            gg = g.codes[i]
            if gg < 0 or c.isna_at(i):
                continue
            v = c.get(i)
            b = res[gg]
            if b.kind == K_NONE or how == "last" or (how == "max" and v > b) or (how == "min" and v < b):
                res[gg] = v
        out = Column.empty(c.kind)
        anynone = False
        for r in res:
            if r.kind == K_NONE:
                anynone = True
        if anynone and c.kind != K_STR:
            out = Column.empty(K_FLOAT)
        for r in res:
            out.append(r if r.kind != K_NONE else Scalar(K_FLOAT, NAN, 0, ""))
        return out
    if c.kind == K_STR:
        raise TypeError(f"agg function failed [how->{how},dtype->str]")
    vals = c.to_floats()
    if how == "sum":
        if c.kind == K_INT or c.kind == K_BOOL:
            out = [0 for _ in range(G)]
            for i in range(len(c)):
                if g.codes[i] >= 0:
                    out[g.codes[i]] += c.i[i]
            return Column.ints(out)
        return Column.floats(group_sum(vals, g.codes, G))
    if how == "prod":
        if c.kind == K_INT or c.kind == K_BOOL:
            out = [1 for _ in range(G)]
            for i in range(len(c)):
                if g.codes[i] >= 0:
                    out[g.codes[i]] *= c.i[i]
            return Column.ints(out)
        return Column.floats(group_prod(vals, g.codes, G))
    if how == "mean":
        return Column.floats(group_mean(vals, g.codes, G))
    if how == "var":
        return Column.floats(group_var(vals, g.codes, G, ddof, False))
    if how == "std":
        return Column.floats(group_var(vals, g.codes, G, ddof, True))
    if how == "median":
        mem = g.members()
        return Column.floats([nanmedian(Column.floats([vals[i] for i in m])) for m in mem])
    raise ValueError(f"unsupported aggregation '{how}'")

def _fdiv(x: float, y: float) -> float:
    # IEEE division (no ZeroDivisionError)
    if y == 0.0:
        if x == 0.0 or x != x:
            return NAN
        neg_zero = str(y).startswith("-")
        return float("inf") if (x > 0.0) != neg_zero else -float("inf")
    return x / y

def _g_cumcount(g: _Grouper, ascending: bool) -> List[int]:
    n = len(g.codes)
    out = [0 for _ in range(n)]
    mem = g.members()
    for m in mem:
        L = len(m)
        for t in range(L):
            out[m[t]] = t if ascending else L - 1 - t
    return out

def _shift_list(v: List[float], periods: int) -> List[float]:
    n = len(v)
    return [v[k - periods] if 0 <= k - periods < n else NAN for k in range(n)]

def _cum_list(v: List[float], kind: int) -> List[float]:
    out = List[float](len(v))
    acc = 0.0 if kind == 0 else 1.0
    best = NAN
    for x in v:
        if x != x:
            out.append(NAN)
            continue
        if kind == 0:
            acc += x
            out.append(acc)
        elif kind == 1:
            acc *= x
            out.append(acc)
        elif kind == 2:
            best = x if best != best or x > best else best
            out.append(best)
        else:
            best = x if best != best or x < best else best
            out.append(best)
    return out

def _sorted_floats(v: List[float]) -> List[float]:
    a = [x for x in v if x == x]
    a.sort()
    return a

def _sem_list(v: List[float]) -> float:
    c = Column.floats(v)
    n = nancount(c)
    if n == 0:
        return NAN
    return nanstd(c) / (float(n) ** 0.5)

class GroupRolling[GB]:
    gb: GB
    window: int
    min_periods: int

    def __init__(self, gb: GB, window: int, min_periods: int):
        self.gb = gb
        self.window = window
        self.min_periods = min_periods

    def _run(self, how: int):
        # results aligned to the original row order (groups' rows stay in place)
        gb = self.gb
        mem = gb.g.members()
        cols = List[Column]()
        labels = List[int]()
        for j in gb._value_cols():
            c = gb.df._cols[j]
            res = [NAN for _ in range(len(c))]
            for m in mem:
                v = c.take(m).to_floats()
                st, en = _fixed_bounds(len(v), self.window, False)
                if how == 0:
                    r = roll_mean(v, st, en, self.min_periods)
                elif how == 1:
                    r = roll_sum(v, st, en, self.min_periods)
                elif how == 2:
                    r = roll_var(v, st, en, self.min_periods, 1)
                elif how == 3:
                    r = [x ** 0.5 if x == x else x for x in roll_var(v, st, en, self.min_periods, 1)]
                elif how == 4:
                    r = roll_minmax(v, st, en, self.min_periods, False)
                else:
                    r = roll_minmax(v, st, en, self.min_periods, True)
                for t in range(len(m)):
                    res[m[t]] = r[t]
            cols.append(Column.floats(res))
            labels.append(j)
        return DataFrame._new(cols, gb.df.columns.take(labels), gb.df.index.copy())

    def mean(self):
        return self._run(0)
    def sum(self):
        return self._run(1)
    def var(self):
        return self._run(2)
    def std(self):
        return self._run(3)
    def min(self):
        return self._run(4)
    def max(self):
        return self._run(5)

class DataFrameGroupBy:
    df: DataFrame
    g: _Grouper
    key_pos: int
    sel: Optional[List[int]]
    key_cols: List[int]

    def __init__(self, df: DataFrame, g: _Grouper, key_pos: int, sel: Optional[List[int]] = None):
        self.df = df
        self.g = g
        self.key_pos = key_pos
        self.sel = sel
        self.key_cols = List[int]()

    def _value_cols(self) -> List[int]:
        if self.sel is not None:
            return self.sel
        return [j for j in range(len(self.df._cols)) if j != self.key_pos and j not in self.key_cols]

    def __getitem__(self, key):
        j = self.df._col_pos(key)
        return SeriesGroupBy(self.g, self.df._cols[j], _label_name(self.df.columns[j]))

    def __getitem__(self, key: List[str]):
        gb = DataFrameGroupBy(self.df, self.g, self.key_pos, [self.df._col_pos(k) for k in key])
        gb.key_cols = self.key_cols
        return gb

    def __getattr__(self, name: Literal[str]):
        return self[name]

    @property
    def ngroups(self) -> int:
        return self.g.ngroups

    @property
    def groups(self) -> Dict[Scalar, List[Scalar]]:
        d = Dict[Scalar, List[Scalar]]()
        mem = self.g.members()
        for k in range(self.g.ngroups):
            d[self.g.keys[k]] = [self.df.index[i] for i in mem[k]]
        return d

    def _agg(self, how: str, numeric_only: bool = False, ddof: int = 1):
        cols = List[int]()
        for j in self._value_cols():
            if self.df._cols[j].kind == K_STR and how not in ("count", "size", "nunique", "min", "max", "first", "last"):
                if numeric_only:
                    continue
            cols.append(j)
        return DataFrame._new([_agg_col(self.g, self.df._cols[j], how, ddof) for j in cols], self.df.columns.take(cols), self.g.index())

    def mean(self, numeric_only: bool = False):
        return self._agg("mean", numeric_only)
    def sum(self, numeric_only: bool = False):
        return self._agg("sum", numeric_only)
    def prod(self, numeric_only: bool = False):
        return self._agg("prod", numeric_only)
    def median(self, numeric_only: bool = False):
        return self._agg("median", numeric_only)
    def std(self, ddof: int = 1, numeric_only: bool = False):
        return self._agg("std", numeric_only, ddof)
    def var(self, ddof: int = 1, numeric_only: bool = False):
        return self._agg("var", numeric_only, ddof)
    def min(self, numeric_only: bool = False):
        return self._agg("min", numeric_only)
    def max(self, numeric_only: bool = False):
        return self._agg("max", numeric_only)
    def first(self):
        return self._agg("first")
    def last(self):
        return self._agg("last")
    def count(self):
        return self._agg("count")
    def nunique(self):
        return self._agg("nunique")

    def size(self) -> Series:
        return Series._new(_agg_col(self.g, Column.ints([0 for _ in range(len(self.g.codes))]), "size"), self.g.index(), None)

    def agg(self, func: str):
        return self._agg(func)

    def agg(self, func: List[str]):
        # pandas returns MultiIndex columns (col, func); flattened here as "col_func"
        cols = List[Column]()
        labels = List[str]()
        for j in self._value_cols():
            for f in func:
                cols.append(_agg_col(self.g, self.df._cols[j], f))
                labels.append(str(self.df.columns[j]) + "_" + f)
        return DataFrame._new(cols, Index.strs(labels), self.g.index())

    def agg(self, func: Dict[str, str]):
        cols = List[Column]()
        labels = List[str]()
        for k, f in func.items():
            j = self.df._col_pos(k)
            cols.append(_agg_col(self.g, self.df._cols[j], f))
            labels.append(k)
        return DataFrame._new(cols, Index.strs(labels), self.g.index())

    def aggregate(self, func):
        return self.agg(func)

    def apply(self, func) -> Series:
        # func: DataFrame -> scalar
        mem = self.g.members()
        out = [func(self.df._take_rows(m)) for m in mem]
        return Series._new(Column.of(out), self.g.index(), None)

    def transform(self, how: str):
        r = self._agg(how)
        pos = [c if c >= 0 else 0 for c in self.g.codes]
        return DataFrame._new([c.take(pos) for c in r._cols], r.columns.copy(), self.df.index.copy())

    def cumcount(self, ascending: bool = True) -> Series:
        return Series._new(Column.ints(_g_cumcount(self.g, ascending)), self.df.index.copy(), None)

    def ngroup(self) -> Series:
        return Series._new(Column.ints([c for c in self.g.codes]), self.df.index.copy(), None)

    def nth(self, n: int):
        mem = self.g.members()
        pos = List[int]()
        for m in mem:
            k = n if n >= 0 else len(m) + n
            if 0 <= k < len(m):
                pos.append(m[k])
        pos.sort()
        return self.df._take_rows(pos)

    def _per_group_cols(self, f):
        # f(Column of one group's values, List[int] positions) -> Column of same length
        cols = List[Column]()
        labels = List[int]()
        mem = self.g.members()
        for j in self._value_cols():
            c = self.df._cols[j]
            out = Column.empty(K_FLOAT, len(c))
            res = [NAN for _ in range(len(c))]
            for m in mem:
                r = f(c.take(m).to_floats())
                for t in range(len(m)):
                    res[m[t]] = r[t]
            cols.append(Column.floats(res))
            labels.append(j)
        return DataFrame._new(cols, self.df.columns.take(labels), self.df.index.copy())

    def shift(self, periods: int = 1):
        return self._per_group_cols(lambda v: _shift_list(v, periods))

    def diff(self, periods: int = 1):
        return self._per_group_cols(lambda v: [v[k] - v[k - periods] if 0 <= k - periods < len(v) else NAN for k in range(len(v))])

    def pct_change(self, periods: int = 1):
        return self._per_group_cols(lambda v: [_fdiv(v[k], v[k - periods]) - 1.0 if 0 <= k - periods < len(v) else NAN for k in range(len(v))])

    def cumsum(self):
        return self._per_group_cols(lambda v: _cum_list(v, 0))

    def cumprod(self):
        return self._per_group_cols(lambda v: _cum_list(v, 1))

    def cummax(self):
        return self._per_group_cols(lambda v: _cum_list(v, 2))

    def cummin(self):
        return self._per_group_cols(lambda v: _cum_list(v, 3))

    def rank(self, method: str = "average", ascending: bool = True, na_option: str = "keep", pct: bool = False):
        return self._per_group_cols(lambda v: rank_1d(v, method, ascending, na_option, pct))

    def skew(self):
        return self._agg_f(lambda v: nanskew_list(v, True))

    def kurt(self):
        return self._agg_f(lambda v: nankurt_list(v, True))

    def sem(self):
        return self._agg_f(lambda v: _sem_list(v))

    def quantile(self, q: float = 0.5, interpolation: str = "linear"):
        return self._agg_f(lambda v: quantile_sorted(_sorted_floats(v), q, interpolation))

    def _agg_f(self, f):
        mem = self.g.members()
        cols = List[Column]()
        labels = List[int]()
        for j in self._value_cols():
            c = self.df._cols[j]
            cols.append(Column.floats([f(c.take(m).to_floats()) for m in mem]))
            labels.append(j)
        return DataFrame._new(cols, self.df.columns.take(labels), self.g.index())

    def filter(self, func):
        mem = self.g.members()
        pos = List[int]()
        for m in mem:
            if func(self.df._take_rows(m)):
                for i in m:
                    pos.append(i)
        pos.sort()
        return self.df._take_rows(pos)

    def pipe(self, func):
        return func(self)

    def rolling(self, window: int, min_periods: Optional[int] = None):
        mp = min_periods if min_periods is not None else window
        return GroupRolling(self, window, mp)

    def __iter__(self):
        mem = self.g.members()
        for k in range(self.g.ngroups):
            yield (self.g.keys[k], self.df._take_rows(mem[k]))

    def get_group(self, key):
        sk = Scalar.of(key)
        mem = self.g.members()
        for k in range(self.g.ngroups):
            if self.g.keys[k] == sk:
                return self.df._take_rows(mem[k])
        raise KeyError(str(sk))

class SeriesGroupBy:
    g: _Grouper
    col: Column
    name: Optional[str]

    def __init__(self, g: _Grouper, col: Column, name: Optional[str]):
        self.g = g
        self.col = col
        self.name = name

    def _agg(self, how: str, ddof: int = 1) -> Series:
        return Series._new(_agg_col(self.g, self.col, how, ddof), self.g.index(), self.name)

    def mean(self) -> Series:
        return self._agg("mean")
    def sum(self) -> Series:
        return self._agg("sum")
    def prod(self) -> Series:
        return self._agg("prod")
    def median(self) -> Series:
        return self._agg("median")
    def std(self, ddof: int = 1) -> Series:
        return self._agg("std", ddof)
    def var(self, ddof: int = 1) -> Series:
        return self._agg("var", ddof)
    def min(self) -> Series:
        return self._agg("min")
    def max(self) -> Series:
        return self._agg("max")
    def first(self) -> Series:
        return self._agg("first")
    def last(self) -> Series:
        return self._agg("last")
    def count(self) -> Series:
        return self._agg("count")
    def nunique(self) -> Series:
        return self._agg("nunique")
    def size(self) -> Series:
        return Series._new(_agg_col(self.g, self.col, "size"), self.g.index(), None)

    def agg(self, func: str) -> Series:
        return self._agg(func)

    def agg(self, func: List[str]):
        return DataFrame._new([_agg_col(self.g, self.col, f) for f in func], Index.strs(func), self.g.index())

    def apply(self, func) -> Series:
        mem = self.g.members()
        out = [func(Series._new(self.col.take(m), Index.range(len(m)), self.name)) for m in mem]
        return Series._new(Column.of(out), self.g.index(), self.name)

    def transform(self, how: str) -> Series:
        r = _agg_col(self.g, self.col, how)
        pos = [c if c >= 0 else 0 for c in self.g.codes]
        return Series._new(r.take(pos), Index.range(len(pos)), self.name)

# ---------------------------------------------------------------------------
# Rolling / Expanding

class Rolling:
    df: DataFrame
    window: int
    min_periods: int
    center: bool
    expanding: bool
    series: bool

    def __init__(self, df: DataFrame, window: int, min_periods: int, center: bool, expanding: bool, series: bool):
        self.df = df
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.expanding = expanding
        self.series = series

    def _bounds(self, n: int):
        if self.expanding:
            return [0 for _ in range(n)], [i + 1 for i in range(n)]
        return _fixed_bounds(n, self.window, self.center)

    def _apply(self, f):
        n = len(self.df)
        start, end = self._bounds(n)
        cols = List[Column]()
        for c in self.df._cols:
            if c.kind == K_STR:
                raise TypeError("rolling on str column")
            cols.append(Column.floats(f(c.to_floats(), start, end)))
        return DataFrame._new(cols, self.df.columns.copy(), self.df.index.copy())

    def _ret(self, f):
        return self._apply(f)

    def sum(self):
        mp = self.min_periods
        return self._apply(lambda v, s, e: roll_sum(v, s, e, mp))
    def mean(self):
        mp = self.min_periods
        return self._apply(lambda v, s, e: roll_mean(v, s, e, mp))
    def var(self, ddof: int = 1):
        mp = self.min_periods
        return self._apply(lambda v, s, e: roll_var(v, s, e, mp, ddof))
    def std(self, ddof: int = 1):
        mp = self.min_periods
        from math import sqrt
        return self._apply(lambda v, s, e: [sqrt(x) if x == x else x for x in roll_var(v, s, e, mp, ddof)])
    def min(self):
        mp = self.min_periods
        return self._apply(lambda v, s, e: roll_minmax(v, s, e, mp, False))
    def max(self):
        mp = self.min_periods
        return self._apply(lambda v, s, e: roll_minmax(v, s, e, mp, True))
    def median(self):
        mp = self.min_periods
        return self._apply(lambda v, s, e: roll_median(v, s, e, mp))
    def count(self):
        return self._apply(lambda v, s, e: [float(sum(1 for j in range(s[i], e[i]) if v[j] == v[j])) for i in range(len(s))])

    def skew(self):
        mp = self.min_periods
        return self._apply(lambda v, s, e: _roll_moment(v, s, e, mp, 3))

    def kurt(self):
        mp = self.min_periods
        return self._apply(lambda v, s, e: _roll_moment(v, s, e, mp, 4))

    def sem(self, ddof: int = 1):
        mp = self.min_periods
        from math import sqrt
        return self._apply(lambda v, s, e: _roll_sem(v, s, e, mp, ddof))

    def first(self):
        return self._apply(lambda v, s, e: _roll_edge(v, s, e, self.min_periods, True))

    def last(self):
        return self._apply(lambda v, s, e: _roll_edge(v, s, e, self.min_periods, False))

    def quantile(self, quantile: float, interpolation: str = "linear"):
        mp = self.min_periods
        return self._apply(lambda v, s, e: roll_quantile(v, s, e, mp, quantile, interpolation))

    def cov(self, other, ddof: int = 1):
        n = len(self.df)
        start, end = self._bounds(n)
        y = other._col.to_floats() if isinstance(other, Series) else other._cols[0].to_floats()
        cols = List[Column]()
        for c in self.df._cols:
            xx, yy = _prep_binary(c.to_floats(), y)
            cols.append(Column.floats(roll_cov(xx, yy, start, end, self.min_periods, ddof)))
        return DataFrame._new(cols, self.df.columns.copy(), self.df.index.copy())

    def corr(self, other):
        n = len(self.df)
        start, end = self._bounds(n)
        y = other._col.to_floats() if isinstance(other, Series) else other._cols[0].to_floats()
        cols = List[Column]()
        for c in self.df._cols:
            xx, yy = _prep_binary(c.to_floats(), y)
            cols.append(Column.floats(roll_corr(xx, yy, start, end, self.min_periods)))
        return DataFrame._new(cols, self.df.columns.copy(), self.df.index.copy())

    def apply(self, func):
        mp = self.min_periods
        def f(v: List[float], s: List[int], e: List[int]) -> List[float]:
            out = List[float](len(s))
            for i in range(len(s)):
                w = [v[j] for j in range(s[i], e[i])]
                cnt = sum(1 for x in w if x == x)
                out.append(float(func(Series(w))) if cnt >= mp else NAN)
            return out
        return self._apply(f)

def _roll_moment(v: List[float], s: List[int], e: List[int], mp: int, order: int) -> List[float]:
    out = List[float](len(s))
    for i in range(len(s)):
        w = List[float]()
        for j in range(s[i], e[i]):
            if v[j] == v[j]:
                w.append(v[j])
        if len(w) < mp:
            out.append(NAN)
        elif order == 3:
            out.append(nanskew_list(w))
        else:
            out.append(nankurt_list(w))
    return out

def _roll_sem(v: List[float], s: List[int], e: List[int], mp: int, ddof: int) -> List[float]:
    from math import sqrt
    vr = roll_var(v, s, e, mp, ddof)
    out = List[float](len(s))
    for i in range(len(s)):
        cnt = 0
        for j in range(s[i], e[i]):
            if v[j] == v[j]:
                cnt += 1
        out.append(sqrt(vr[i] / float(cnt)) if vr[i] == vr[i] and cnt > 0 else NAN)
    return out

def _roll_edge(v: List[float], s: List[int], e: List[int], mp: int, first: bool) -> List[float]:
    out = List[float](len(s))
    for i in range(len(s)):
        cnt = 0
        for j in range(s[i], e[i]):
            if v[j] == v[j]:
                cnt += 1
        if cnt < mp or e[i] <= s[i]:
            out.append(NAN)
        else:
            r = NAN
            if first:
                for j in range(s[i], e[i]):
                    if v[j] == v[j]:
                        r = v[j]
                        break
            else:
                for j in range(e[i] - 1, s[i] - 1, -1):
                    if v[j] == v[j]:
                        r = v[j]
                        break
            out.append(r)
    return out

def _prep_binary(x: List[float], y: List[float]):
    # pandas prep_binary: NaN in either input masks both
    n = len(x)
    xo = List[float](n)
    yo = List[float](n)
    for i in range(n):
        if x[i] != x[i] or y[i] != y[i]:
            xo.append(float("nan"))
            yo.append(float("nan"))
        else:
            xo.append(x[i])
            yo.append(y[i])
    return xo, yo

class Ewm:
    df: DataFrame
    com: float
    adjust: bool
    ignore_na: bool
    min_periods: int

    def __init__(self, df: DataFrame, com: float, adjust: bool, ignore_na: bool, min_periods: int):
        self.df = df
        self.com = com
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.min_periods = max(min_periods, 1)

    def _each(self, f):
        n = len(self.df)
        cols = List[Column]()
        for c in self.df._cols:
            if c.kind == K_STR:
                raise TypeError("ewm on str column")
            cols.append(Column.floats(f(c.to_floats(), [0], [n])))
        return DataFrame._new(cols, self.df.columns.copy(), self.df.index.copy())

    def mean(self):
        com = self.com
        adj = self.adjust
        ign = self.ignore_na
        mp = self.min_periods
        return self._each(lambda v, s, e: ewm_kernel(v, s, e, mp, com, adj, ign, True))

    def sum(self):
        if not self.adjust:
            raise NotImplementedError("sum is not implemented with adjust=False")
        com = self.com
        adj = self.adjust
        ign = self.ignore_na
        mp = self.min_periods
        return self._each(lambda v, s, e: ewm_kernel(v, s, e, mp, com, adj, ign, False))

    def var(self, bias: bool = False):
        com = self.com
        adj = self.adjust
        ign = self.ignore_na
        mp = self.min_periods
        return self._each(lambda v, s, e: ewmcov_kernel(v, s, e, mp, v, com, adj, ign, bias))

    def std(self, bias: bool = False):
        from math import sqrt
        com = self.com
        adj = self.adjust
        ign = self.ignore_na
        mp = self.min_periods
        return self._each(lambda v, s, e: [sqrt(x) if x == x else x for x in ewmcov_kernel(v, s, e, mp, v, com, adj, ign, bias)])

    def cov(self, other, bias: bool = False):
        n = len(self.df)
        y = other._col.to_floats() if isinstance(other, Series) else other._cols[0].to_floats()
        cols = List[Column]()
        for c in self.df._cols:
            xx, yy = _prep_binary(c.to_floats(), y)
            cols.append(Column.floats(ewmcov_kernel(xx, [0], [n], self.min_periods, yy, self.com, self.adjust, self.ignore_na, bias)))
        return DataFrame._new(cols, self.df.columns.copy(), self.df.index.copy())

    def corr(self, other):
        from math import sqrt
        n = len(self.df)
        y0 = other._col.to_floats() if isinstance(other, Series) else other._cols[0].to_floats()
        cols = List[Column]()
        for c in self.df._cols:
            x, y = _prep_binary(c.to_floats(), y0)
            cv = ewmcov_kernel(x, [0], [n], self.min_periods, y, self.com, self.adjust, self.ignore_na, True)
            xv = ewmcov_kernel(x, [0], [n], self.min_periods, x, self.com, self.adjust, self.ignore_na, True)
            yv = ewmcov_kernel(y, [0], [n], self.min_periods, y, self.com, self.adjust, self.ignore_na, True)
            r = List[float](n)
            for i in range(n):
                p = xv[i] * yv[i]
                dd = sqrt(p) if p >= 0.0 else float("nan")
                r.append(float("nan") if dd == 0.0 and cv[i] == 0.0 else (cv[i] / dd if dd != 0.0 else (float("inf") if cv[i] > 0.0 else (-float("inf") if cv[i] < 0.0 else float("nan")))))
            cols.append(Column.floats(r))
        return DataFrame._new(cols, self.df.columns.copy(), self.df.index.copy())

def _ser(r: DataFrame, series: bool) -> Series:
    s = r._series(0)
    return s


# ---------------------------------------------------------------------------
# reshape / binning

def pivot_table(df: DataFrame, values: str, index: str, columns: Optional[str] = None,
                aggfunc: str = "mean", dropna_all: bool = True):
    # Result: sorted unique `index` labels as rows, sorted unique `columns`
    # labels as columns (or a single column named after `values`).
    if columns is None:
        g = df.groupby(index)[values]
        r = g.agg(aggfunc)
        return r.to_frame(values)
    ic = df._cols[df._col_pos(index)]
    cc = df._cols[df._col_pos(columns)]
    vc = df._cols[df._col_pos(values)]
    rg = _Grouper(ic, index)
    cg = _Grouper(cc, columns)
    R = rg.ngroups
    C = cg.ngroups
    cells = List[List[int]]()
    for _ in range(R * C):
        cells.append(List[int]())
    for i in range(len(df)):
        r = rg.codes[i]
        c = cg.codes[i]
        if r >= 0 and c >= 0:
            cells[r * C + c].append(i)
    out = List[Column]()
    for c in range(C):
        col = List[float](R)
        for r in range(R):
            m = cells[r * C + c]
            if len(m) == 0:
                col.append(NAN)
            else:
                sub = vc.take(m)
                col.append(_agg_scalar(sub, aggfunc))
        out.append(Column.floats(col))
    return DataFrame._new(out, Index._wrap(cg.key_col.copy(), columns), rg.index())

def _agg_scalar(c: Column, how: str) -> float:
    if how == "mean":
        return nanmean(c)
    elif how == "sum":
        return nansum(c)
    elif how == "min":
        return nanminmax(c, True, False).f if c.kind == K_FLOAT else float(nanminmax(c, True, False).i)
    elif how == "max":
        return nanminmax(c, True, True).f if c.kind == K_FLOAT else float(nanminmax(c, True, True).i)
    elif how == "count":
        return float(nancount(c))
    elif how == "median":
        return nanmedian(c)
    elif how == "std":
        return nanstd(c)
    elif how == "var":
        return nanvar(c)
    elif how == "first":
        return c.getf(0)
    elif how == "last":
        return c.getf(len(c) - 1)
    raise ValueError("unsupported aggfunc " + how)

def get_dummies(data, prefix: Optional[str] = None, prefix_sep: str = "_", dummy_na: bool = False,
                drop_first: bool = False):
    # Series -> frame of uint8-like bool columns, sorted category order.
    if isinstance(data, Series):
        s = data
        g = _Grouper(s._col, s.name)
        cols = List[Column]()
        labels = List[str]()
        start = 1 if drop_first else 0
        for k in range(start, g.ngroups):
            cols.append(Column.bools([g.codes[i] == k for i in range(len(s))]))
            nm = str(g.keys[k])
            labels.append(nm if prefix is None else prefix + prefix_sep + nm)
        if dummy_na:
            cols.append(Column.bools([g.codes[i] < 0 for i in range(len(s))]))
            labels.append("nan" if prefix is None else prefix + prefix_sep + "nan")
        return DataFrame._new(cols, Index.strs(labels), s.index.copy())
    raise NotImplementedError("get_dummies: only Series input is supported")

class Interval:
    left: float
    right: float
    closed: str

    def __init__(self, left: float, right: float, closed: str = "right"):
        self.left = left
        self.right = right
        self.closed = closed

    def __contains__(self, x: float) -> bool:
        if self.closed == "right":
            return self.left < x <= self.right
        elif self.closed == "left":
            return self.left <= x < self.right
        elif self.closed == "both":
            return self.left <= x <= self.right
        return self.left < x < self.right

def _fmt_num(x: float) -> str:
    return repr_float(x)

def repr_float(x: float) -> str:
    s = f"{x:.15g}"
    if "e" not in s and "." not in s and "n" not in s:
        s = s + ".0"
    return s

def _cut_core(x: Series, bins: List[float], right: bool, labels: Optional[List[str]], precision: int,
              include_lowest: bool, duplicates: str, return_codes: bool) -> Series:
    ub = List[float]()
    for b in bins:
        if len(ub) == 0 or ub[len(ub) - 1] != b:
            ub.append(b)
    if len(ub) < len(bins) and len(bins) != 2:
        if duplicates == "raise":
            raise ValueError("Bin edges must be unique: " + str(bins) + ".\nYou can drop duplicate edges by setting the 'duplicates' kwarg")
        bins = ub
    nb = len(bins)
    n = len(x)
    codes = List[int](n)
    for i in range(n):
        v = x._col.getf(i)
        if v != v:
            codes.append(-1)
            continue
        # searchsorted
        lo = 0
        hi = nb
        while lo < hi:
            mid = (lo + hi) // 2
            go = bins[mid] < v if right else bins[mid] <= v
            if go:
                lo = mid + 1
            else:
                hi = mid
        ids = lo
        if include_lowest and v == bins[0]:
            ids = 1
        if ids == 0 or ids == nb:
            codes.append(-1)
        else:
            codes.append(ids - 1)
    if return_codes:
        has_na = False
        for c in codes:
            if c < 0:
                has_na = True
        if has_na:
            return Series._new(Column.floats([float(c) if c >= 0 else NAN for c in codes]), x.index.copy(), x.name)
        return Series._new(Column.ints(codes), x.index.copy(), x.name)
    labs = List[str]()
    if labels is not None:
        if len(labels) != nb - 1:
            raise ValueError("Bin labels must be one fewer than the number of bin edges")
        labs = labels
    else:
        prec = infer_precision(precision, bins)
        br = [round_frac(b, prec) for b in bins]
        if right and include_lowest:
            br[0] = br[0] - 10.0 ** (-prec)
        for k in range(nb - 1):
            if right:
                labs.append("(" + _fmt_num(br[k]) + ", " + _fmt_num(br[k + 1]) + "]")
            else:
                labs.append("[" + _fmt_num(br[k]) + ", " + _fmt_num(br[k + 1]) + ")")
    out = List[str](n)
    for i in range(n):
        out.append(labs[codes[i]] if codes[i] >= 0 else NA_STR)
    return Series._new(Column.strs(out), x.index.copy(), x.name)

def _nbins_to_bins(x: Series, nbins: int, right: bool) -> List[float]:
    if nbins < 1:
        raise ValueError("`bins` should be a positive integer.")
    if len(x) == 0:
        raise ValueError("Cannot cut empty array")
    mn = x.min().f if x._col.kind == K_FLOAT else float(x.min().i)
    mx = x.max().f if x._col.kind == K_FLOAT else float(x.max().i)
    if mn == float("inf") or mn == -float("inf") or mx == float("inf") or mx == -float("inf"):
        raise ValueError("cannot specify integer `bins` when input data contains infinity")
    if mn == mx:
        mn -= 0.001 * abs(mn) if mn != 0 else 0.001
        mx += 0.001 * abs(mx) if mx != 0 else 0.001
        return linspace_list(mn, mx, nbins + 1)
    bins = linspace_list(mn, mx, nbins + 1)
    adj = (mx - mn) * 0.001
    if right:
        bins[0] -= adj
    else:
        bins[nbins] += adj
    return bins

def cut(x: Series, bins, right: bool = True, labels = None, retbins: bool = False, precision: int = 3,
        include_lowest: bool = False, duplicates: str = "raise"):
    b = List[float]()
    if isinstance(bins, int):
        b = _nbins_to_bins(x, bins, right)
    else:
        for v in bins:
            b.append(float(v))
        for k in range(1, len(b)):
            if b[k] < b[k - 1]:
                raise ValueError("bins must increase monotonically.")
    if labels is None:
        return _cut_core(x, b, right, None, precision, include_lowest, duplicates, False)
    elif isinstance(labels, bool):
        return _cut_core(x, b, right, None, precision, include_lowest, duplicates, True)
    else:
        return _cut_core(x, b, right, [str(l) for l in labels], precision, include_lowest, duplicates, False)

def qcut(x: Series, q, labels = None, precision: int = 3, duplicates: str = "raise"):
    qs = List[float]()
    if isinstance(q, int):
        if q < 1:
            raise ValueError("`q` should be a positive integer.")
        qs = linspace_list(0.0, 1.0, q + 1)
        for k in range(q + 1):
            if float(q) * qs[k] != float(k):
                qs[k] = np.nextafter(qs[k], 1.0)
    else:
        for v in q:
            qs.append(float(v))
    a = _sorted_valid(x._col)
    bins = [quantile_sorted(a, v, "linear") for v in qs]
    if labels is None:
        return _cut_core(x, bins, True, None, precision, True, duplicates, False)
    elif isinstance(labels, bool):
        return _cut_core(x, bins, True, None, precision, True, duplicates, True)
    else:
        return _cut_core(x, bins, True, [str(l) for l in labels], precision, True, duplicates, False)

def crosstab(index: Series, columns: Series) -> DataFrame:
    rg = _Grouper(index._col, index.name)
    cg = _Grouper(columns._col, columns.name)
    R = rg.ngroups
    C = cg.ngroups
    cnt = [[0 for _ in range(C)] for _ in range(R)]
    for i in range(len(index)):
        if rg.codes[i] >= 0 and cg.codes[i] >= 0:
            cnt[rg.codes[i]][cg.codes[i]] += 1
    cols = List[Column]()
    for c in range(C):
        cols.append(Column.ints([cnt[r][c] for r in range(R)]))
    return DataFrame._new(cols, Index._wrap(cg.key_col.copy(), columns.name), rg.index())

def factorize(values: Series, sort: bool = False):
    first = Dict[Scalar, int]()
    order = List[Scalar]()
    codes = List[int](len(values))
    for i in range(len(values)):
        if values._col.isna_at(i):
            codes.append(-1)
            continue
        v = values._col.get(i)
        if v not in first:
            first[v] = len(order)
            order.append(v)
        codes.append(first[v])
    if sort:
        srt = order.copy()
        srt.sort()
        pos = Dict[Scalar, int]()
        for k in range(len(srt)):
            pos[srt[k]] = k
        for i in range(len(codes)):
            if codes[i] >= 0:
                codes[i] = pos[order[codes[i]]]
        order = srt
    return codes, Index._wrap(Column.from_scalars(order))

# ---------------------------------------------------------------------------
# module-level functions

def _concat_keys_cols(res: DataFrame, objs: List[DataFrame], keys: List[str]) -> DataFrame:
    # axis=1 with keys: MultiIndex columns (key, column)
    k0 = List[Scalar]()
    k1 = List[Scalar]()
    for t in range(len(objs)):
        for j in range(len(objs[t].columns)):
            k0.append(Scalar.of(keys[t]))
            k1.append(objs[t].columns[j])
    mi = Index._multi([Column.from_scalars(k0), Column.from_scalars(k1)], [None, None])
    return DataFrame._new([c.copy() for c in res._cols], mi, res.index.copy())

def _concat_keys_rows(res, objs, keys: List[str]):
    # axis=0 with keys: MultiIndex rows (key, original label)
    k0 = List[Scalar]()
    k1 = List[Scalar]()
    for t in range(len(objs)):
        ix = objs[t].index
        for i in range(len(ix)):
            k0.append(Scalar.of(keys[t]))
            k1.append(ix[i])
    mi = Index._multi([Column.from_scalars(k0), Column.from_scalars(k1)], [None, None])
    if isinstance(res, Series):
        return Series._new(res._col.copy(), mi, res.name)
    return DataFrame._new([c.copy() for c in res._cols], res.columns.copy(), mi)

def concat(objs, axis: int = 0, ignore_index: bool = False, keys: Optional[List[str]] = None):
    if len(objs) == 0:
        raise ValueError("No objects to concatenate")
    if keys is not None:
        kk = keys.__val__()
        if len(kk) != len(objs):
            raise ValueError("The length of the keys must match the number of objects")
        if isinstance(objs[0], Series):
            if axis == 1:
                s_cols = List[Column]()
                s_labels = List[Scalar]()
                for t in range(len(objs)):
                    s_cols.append(objs[t]._col.copy())
                    s_labels.append(Scalar.of(kk[t]))
                return DataFrame._new(s_cols, Index._wrap(Column.from_scalars(s_labels)), objs[0].index.copy())
            return _concat_keys_rows(_concat_series0(objs, False), objs, kk)
        res0 = _concat_frames(objs, axis, False)
        if axis == 1:
            return _concat_keys_cols(res0, objs, kk)
        return _concat_keys_rows(res0, objs, kk)
    if isinstance(objs[0], Series):
        if axis == 1:
            raise ValueError("pandas: concat(list of Series, axis=1) returns a DataFrame; use concat_series_axis1")
        return _concat_series0(objs, ignore_index)
    else:
        return _concat_frames(objs, axis, ignore_index)

def concat_series_axis1(objs: List[Series]) -> DataFrame:
    # Typhon: concat([s1, s2], axis=1) has a different result type than axis=0
    cols = [s._col.copy() for s in objs]
    labels = List[Scalar]()
    for k in range(len(objs)):
        labels.append(Scalar.of(objs[k].name) if objs[k].name is not None else Scalar.of(k))
    return DataFrame._new(cols, Index._wrap(Column.from_scalars(labels)), objs[0].index.copy())

def _concat_series0(objs: List[Series], ignore_index: bool) -> Series:
    c = objs[0]._col.copy()
    idx = objs[0].index
    for k in range(1, len(objs)):
        c = c.extend(objs[k]._col)
        idx = idx.append(objs[k].index)
    if ignore_index:
        idx = Index.range(len(c))
    return Series._new(c, idx, objs[0].name)

def _concat_frames(objs: List[DataFrame], axis: int, ignore_index: bool) -> DataFrame:
    if axis == 1:
        cols = List[Column]()
        labels = List[Scalar]()
        base = objs[0].index
        for df in objs:
            for j in range(len(df._cols)):
                if df.index.equals(base):
                    cols.append(df._cols[j].copy())
                else:
                    cols.append(df._cols[j].take([df.index.get_loc_opt(base[i]) for i in range(len(base))]))
                labels.append(df.columns[j])
        return DataFrame._new(cols, Index._wrap(Column.from_scalars(labels)), base.copy())
    # union of columns in order of appearance
    labels = List[Scalar]()
    seen = Set[Scalar]()
    for df in objs:
        for j in range(len(df.columns)):
            if df.columns[j] not in seen:
                seen.add(df.columns[j])
                labels.append(df.columns[j])
    cols = List[Column]()
    for lab in labels:
        c: Optional[Column] = None
        for df in objs:
            k = df.columns.get_loc_opt(lab)
            part = df._cols[k].copy() if k >= 0 else Column.floats([NAN for _ in range(len(df))])
            if c is None:
                c = part
            else:
                c = c.extend(part)
        cols.append(c)
    idx = objs[0].index.copy()
    for k in range(1, len(objs)):
        idx = idx.append(objs[k].index)
    if ignore_index:
        idx = Index.range(len(idx))
    return DataFrame._new(cols, Index._wrap(Column.from_scalars(labels)), idx)


def merge(left: DataFrame, right: DataFrame, how: str = "inner", on = None, left_on = None, right_on = None, suffixes: Tuple[str, str] = ("_x", "_y"), indicator: bool = False):
    lk = ""
    rk = ""
    if on is not None:
        lk = str(on)
        rk = str(on)
    else:
        lk = str(left_on)
        rk = str(right_on)
    lj = left._col_pos(lk)
    rj = right._col_pos(rk)
    lc = left._cols[lj]
    rc = right._cols[rj]
    rmap = Dict[Scalar, List[int]]()
    rorder = List[Scalar]()
    for i in range(len(rc)):
        v = rc.get(i)
        if v not in rmap:
            rmap[v] = List[int]()
            rorder.append(v)
        rmap[v].append(i)
    lpos = List[int]()
    rpos = List[int]()
    keys = List[Scalar]()
    used = Set[Scalar]()
    for i in range(len(lc)):
        v = lc.get(i)
        if v in rmap:
            used.add(v)
            for r in rmap[v]:
                lpos.append(i)
                rpos.append(r)
                keys.append(v)
        elif how == "left" or how == "outer":
            lpos.append(i)
            rpos.append(-1)
            keys.append(v)
    if how == "right" or how == "outer":
        if how == "right":
            lpos.clear()
            rpos.clear()
            keys.clear()
            lmap = Dict[Scalar, List[int]]()
            for i in range(len(lc)):
                v = lc.get(i)
                if v not in lmap:
                    lmap[v] = List[int]()
                lmap[v].append(i)
            for r in range(len(rc)):
                v = rc.get(r)
                if v in lmap:
                    for i in lmap[v]:
                        lpos.append(i)
                        rpos.append(r)
                        keys.append(v)
                else:
                    lpos.append(-1)
                    rpos.append(r)
                    keys.append(v)
        else:
            for r in range(len(rc)):
                v = rc.get(r)
                if v not in used:
                    lpos.append(-1)
                    rpos.append(r)
                    keys.append(v)
    if how == "outer":
        order = list(range(len(keys)))
        order.sort(key=lambda t: keys[t])
        lpos = [lpos[t] for t in order]
        rpos = [rpos[t] for t in order]
        keys = [keys[t] for t in order]
    lnames = Set[str]([str(left.columns[j]) for j in range(len(left.columns))])
    rnames = Set[str]([str(right.columns[j]) for j in range(len(right.columns))])
    cols = List[Column]()
    labels = List[str]()
    for j in range(len(left._cols)):
        name = str(left.columns[j])
        if j == lj and lk == rk:
            kc = Column.from_scalars(keys)
            if lc.kind == K_INT and not any(p < 0 for p in lpos) or (lc.kind == K_INT and how == "outer"):
                kc = kc.astype(lc.kind) if lc.kind != K_FLOAT else kc
            cols.append(kc)
            labels.append(name)
            continue
        cols.append(left._cols[j].take(lpos))
        labels.append(name + suffixes[0] if name in rnames and name != rk else name)
    for j in range(len(right._cols)):
        name = str(right.columns[j])
        if j == rj and lk == rk:
            continue
        cols.append(right._cols[j].take(rpos))
        labels.append(name + suffixes[1] if name in lnames and name != lk else name)
    if indicator:
        ind = List[str]()
        for t in range(len(lpos)):
            if lpos[t] >= 0 and rpos[t] >= 0:
                ind.append("both")
            elif lpos[t] >= 0:
                ind.append("left_only")
            else:
                ind.append("right_only")
        cols.append(Column.from_scalars([Scalar.of(v) for v in ind]).astype(K_STR))
        labels.append("_merge")
    return DataFrame._new(cols, Index.strs(labels), Index.range(len(keys)))

def isna(obj):
    if isinstance(obj, Series) or isinstance(obj, DataFrame):
        return obj.isna()
    else:
        return Scalar.of(obj).isna()

def isnull(obj):
    return isna(obj)

def notna(obj):
    if isinstance(obj, Series) or isinstance(obj, DataFrame):
        return obj.notna()
    else:
        return not Scalar.of(obj).isna()

def notnull(obj):
    return notna(obj)

def unique(values) -> List[Scalar]:
    return Series(values).unique()

def to_numeric(arg) -> Series:
    s = Series(arg)
    return s.astype(float) if s._col.kind == K_STR else s

# ---------------------------------------------------------------------------
# CSV reader

def _split_csv(text: str, sep: str) -> List[List[str]]:
    rows = List[List[str]]()
    row = List[str]()
    cur = List[str]()
    i = 0
    n = len(text)
    inq = False
    quoted = False
    while i < n:
        ch = text[i]
        if inq:
            if ch == '"':
                if i + 1 < n and text[i + 1] == '"':
                    cur.append('"')
                    i += 1
                else:
                    inq = False
            else:
                cur.append(ch)
        else:
            if ch == '"':
                inq = True
                quoted = True
            elif text[i:i + len(sep)] == sep:
                row.append("".join(cur))
                cur.clear()
                i += len(sep) - 1
            elif ch == "\n" or ch == "\r":
                if ch == "\r" and i + 1 < n and text[i + 1] == "\n":
                    i += 1
                row.append("".join(cur))
                cur.clear()
                if not (len(row) == 1 and row[0] == "" and not quoted):
                    rows.append(row)
                row = List[str]()
                quoted = False
            else:
                cur.append(ch)
        i += 1
    if len(cur) > 0 or len(row) > 0:
        row.append("".join(cur))
        rows.append(row)
    return rows

_NA_VALUES = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
              "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"]

def _infer_column(vals: List[str], extra_na: List[str] = List[str](), keep_default: bool = True) -> Column:
    nas = Set[str](_NA_VALUES) if keep_default else Set[str]()
    for e in extra_na:
        nas.add(e)
    all_int = True
    all_float = True
    all_bool = True
    any_na = False
    for v in vals:
        if v in nas:
            any_na = True
            continue
        if all_bool and v not in ("True", "False", "TRUE", "FALSE", "true", "false"):
            all_bool = False
        if all_int:
            try:
                int(v.strip())
            except:
                all_int = False
        if all_float and not all_int:
            try:
                float(v.strip())
            except:
                all_float = False
    nonna = 0
    for v in vals:
        if v not in nas:
            nonna += 1
    if nonna == 0:
        return Column.floats([NAN for _ in vals])
    if all_int and not any_na:
        return Column.ints([int(v.strip()) for v in vals])
    if all_int or all_float:
        return Column.floats([NAN if v in nas else float(v.strip()) for v in vals])
    if all_bool and not any_na:
        return Column.bools([v in ("True", "TRUE", "true") for v in vals])
    return Column.strs([NA_STR if v in nas else v for v in vals])

def read_csv(filepath_or_buffer: str, sep: str = ",", header: Optional[int] = 0, names: Optional[List[str]] = None,
             index_col = None, usecols: Optional[List[str]] = None, nrows: Optional[int] = None,
             skiprows: int = 0, dtype: Optional[Dict[str, str]] = None, na_values: Optional[List[str]] = None,
             parse_dates: Optional[List[str]] = None, keep_default_na: bool = True,
             skip_blank_lines: bool = True, comment: Optional[str] = None):
    text = ""
    with open(filepath_or_buffer) as f:
        text = f.read()
    return _read_csv_text(text, sep, header, names, index_col, usecols, nrows, skiprows, dtype, na_values,
                          parse_dates, keep_default_na, comment)

def read_csv_string(text: str, sep: str = ",", header: Optional[int] = 0, names: Optional[List[str]] = None,
                    index_col = None, usecols: Optional[List[str]] = None, nrows: Optional[int] = None,
                    skiprows: int = 0, dtype: Optional[Dict[str, str]] = None, na_values: Optional[List[str]] = None,
                    parse_dates: Optional[List[str]] = None, keep_default_na: bool = True,
                    skip_blank_lines: bool = True, comment: Optional[str] = None):
    # Typhon extension: parse CSV text directly (pandas: read_csv(io.StringIO(text)))
    return _read_csv_text(text, sep, header, names, index_col, usecols, nrows, skiprows, dtype, na_values,
                          parse_dates, keep_default_na, comment)

def _read_csv_text(text: str, sep: str, header: Optional[int], names: Optional[List[str]], index_col, usecols: Optional[List[str]], nrows: Optional[int], skiprows: int,
                   dtype: Optional[Dict[str, str]] = None, na_values: Optional[List[str]] = None, parse_dates: Optional[List[str]] = None,
                   keep_default_na: bool = True, comment: Optional[str] = None):
    rows = _split_csv(text, sep)
    rows = rows[skiprows:]
    if comment is not None:
        cm = comment.__val__()
        kept = List[List[str]]()
        for r in rows:
            if len(r) > 0 and r[0].startswith(cm):
                continue
            kept.append(r)
        rows = kept
    extra = na_values if na_values is not None else List[str]()
    hdr = List[str]()
    body = rows
    if header is not None:
        hdr = rows[header]
        body = rows[header + 1:]
    if names is not None:
        hdr = names
    if nrows is not None:
        body = body[:nrows]
    m = len(hdr) if len(hdr) > 0 else (len(body[0]) if len(body) > 0 else 0)
    if index_col is None and len(hdr) > 0 and len(body) > 0 and len(body[0]) == len(hdr) + 1:
        # pandas: one extra field per row means the first column is the index
        icol = _infer_column([r[0] if len(r) > 0 else "" for r in body], extra, keep_default_na)
        cols = List[Column]()
        for j in range(m):
            cols.append(_infer_column([r[j + 1] if j + 1 < len(r) else "" for r in body], extra, keep_default_na))
        hdr2 = [h if h != "" else f"Unnamed: {j}" for j, h in enumerate(hdr)]
        df = DataFrame._new(cols, Index.strs(hdr2), Index._wrap(icol, None))
        if usecols is not None:
            df = df[usecols]
        return df
    if len(hdr) == 0:
        hdr = [str(j) for j in range(m)]
    # unnamed header fields
    for j in range(len(hdr)):
        if hdr[j] == "":
            hdr[j] = f"Unnamed: {j}"
    cols = List[Column]()
    for j in range(m):
        cols.append(_infer_column([r[j] if j < len(r) else "" for r in body], extra, keep_default_na))
    labels = Index.strs(hdr)
    if header is None and names is None:
        labels = Index.range(m)
    if dtype is not None:
        for j in range(len(hdr)):
            if hdr[j] in dtype.__val__():
                dn = dtype.__val__()[hdr[j]]
                raw = [r[j] if j < len(r) else "" for r in body]
                k = _kind_of_dtype(dn)
                if k == K_STR:
                    cols[j] = Column.strs([NA_STR if v in Set[str](_NA_VALUES) or v in Set[str](extra) else v for v in raw])
                elif k == K_FLOAT:
                    cols[j] = Column.floats([NAN if (v in Set[str](_NA_VALUES) or v in Set[str](extra)) else float(v.strip()) for v in raw])
                else:
                    cols[j] = cols[j].astype(k)
    if parse_dates is not None:
        for j in range(len(hdr)):
            if hdr[j] in parse_dates.__val__():
                raw = [r[j] if j < len(r) else "" for r in body]
                cols[j] = Column(K_DT, List[float](), [DT.NAT if v == "" else DT.parse_ts(v) for v in raw], List[str]())
    df = DataFrame._new(cols, labels, Index.range(len(body)))
    if usecols is not None:
        df = df[usecols]
    if index_col is not None:
        if isinstance(index_col, int):
            j = index_col
            idx = Index._wrap(df._cols[j].copy(), None if hdr[j].startswith("Unnamed: ") else hdr[j])
            df = df._take_cols([k for k in range(len(df._cols)) if k != j])
            df.index = idx
        else:
            df = df.set_index(index_col)
    return df

# ---------------------------------------------------------------------------
# Series.str accessor

class StringMethods:
    s: Series

    def _chk(self) -> Column:
        if self.s._col.kind != K_STR:
            raise AttributeError("Can only use .str accessor with string values!")
        return self.s._col

    def _out(self, col: Column) -> Series:
        return self.s._with(col)

    def _frame(self, cols: List[Column]) -> DataFrame:
        return DataFrame._new(cols, Index.range(len(cols)), self.s.index)

    def len(self) -> Series:
        return self._out(S.smap_int(self._chk(), lambda x: len(x)))
    def lower(self) -> Series:
        return self._out(S.smap(self._chk(), lambda x: x.lower()))
    def upper(self) -> Series:
        return self._out(S.smap(self._chk(), lambda x: x.upper()))
    def title(self) -> Series:
        return self._out(S.smap(self._chk(), lambda x: x.title()))
    def capitalize(self) -> Series:
        return self._out(S.smap(self._chk(), lambda x: x.capitalize()))
    def swapcase(self) -> Series:
        return self._out(S.smap(self._chk(), lambda x: x.swapcase()))
    def casefold(self) -> Series:
        return self._out(S.smap(self._chk(), lambda x: x.casefold()))
    def strip(self, to_strip: Optional[str] = None) -> Series:
        return self._out(S.strip(self._chk(), to_strip, True, True))
    def lstrip(self, to_strip: Optional[str] = None) -> Series:
        return self._out(S.strip(self._chk(), to_strip, True, False))
    def rstrip(self, to_strip: Optional[str] = None) -> Series:
        return self._out(S.strip(self._chk(), to_strip, False, True))
    def isalpha(self) -> Series:
        return self._out(S.smap_bool(self._chk(), lambda x: x.isalpha()))
    def isdigit(self) -> Series:
        return self._out(S.smap_bool(self._chk(), lambda x: x.isdigit()))
    def isalnum(self) -> Series:
        return self._out(S.smap_bool(self._chk(), lambda x: x.isalnum()))
    def isspace(self) -> Series:
        return self._out(S.smap_bool(self._chk(), lambda x: x.isspace()))
    def islower(self) -> Series:
        return self._out(S.smap_bool(self._chk(), lambda x: x.islower()))
    def isupper(self) -> Series:
        return self._out(S.smap_bool(self._chk(), lambda x: x.isupper()))
    def isnumeric(self) -> Series:
        return self._out(S.smap_bool(self._chk(), lambda x: x.isnumeric()))
    def isdecimal(self) -> Series:
        return self._out(S.smap_bool(self._chk(), lambda x: x.isdecimal()))
    def istitle(self) -> Series:
        return self._out(S.smap_bool(self._chk(), lambda x: x.istitle()))
    def contains(self, pat: str, case: bool = True, regex: bool = True, na: Optional[bool] = None) -> Series:
        return self._out(S.contains(self._chk(), pat, case, regex, na))
    def match(self, pat: str, case: bool = True) -> Series:
        return self._out(S.match(self._chk(), pat, case, False))
    def fullmatch(self, pat: str, case: bool = True) -> Series:
        return self._out(S.match(self._chk(), pat, case, True))
    def startswith(self, pat: str) -> Series:
        return self._out(S.startswith(self._chk(), pat, False))
    def endswith(self, pat: str) -> Series:
        return self._out(S.startswith(self._chk(), pat, True))
    def replace(self, pat: str, repl: str, n: int = -1, case: bool = True, regex: bool = False) -> Series:
        return self._out(S.replace(self._chk(), pat, repl, n, case, regex))
    def find(self, sub: str) -> Series:
        return self._out(S.find(self._chk(), sub, False))
    def rfind(self, sub: str) -> Series:
        return self._out(S.find(self._chk(), sub, True))
    def count(self, pat: str) -> Series:
        return self._out(S.count(self._chk(), pat, 0))
    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> Series:
        return self._out(S.slice_(self._chk(), start, stop, step))
    def get(self, i: int) -> Series:
        return self._out(S.get(self._chk(), i))
    def pad(self, width: int, side: str = "left", fillchar: str = " ") -> Series:
        return self._out(S.pad(self._chk(), width, side, fillchar))
    def center(self, width: int, fillchar: str = " ") -> Series:
        return self._out(S.pad(self._chk(), width, "both", fillchar))
    def ljust(self, width: int, fillchar: str = " ") -> Series:
        return self._out(S.pad(self._chk(), width, "right", fillchar))
    def rjust(self, width: int, fillchar: str = " ") -> Series:
        return self._out(S.pad(self._chk(), width, "left", fillchar))
    def zfill(self, width: int) -> Series:
        return self._out(S.zfill(self._chk(), width))
    def repeat(self, repeats: int) -> Series:
        return self._out(S.repeat(self._chk(), repeats))
    def removeprefix(self, prefix: str) -> Series:
        return self._out(S.removeprefix(self._chk(), prefix, False))
    def removesuffix(self, suffix: str) -> Series:
        return self._out(S.removeprefix(self._chk(), suffix, True))
    def cat(self, others = None, sep: str = "", na_rep: Optional[str] = None):
        if others is None:
            return S.cat_all(self._chk(), sep, na_rep)
        o = Series(others)
        if len(o) != len(self.s):
            raise ValueError("Series length mismatch")
        return self._out(S.cat_other(self._chk(), o._col, sep, na_rep))

    def split(self, pat: Optional[str] = None, n: int = -1, expand: Literal[bool] = False, regex: Optional[bool] = None):
        parts = S.split_lists(self._chk(), pat, n, False, True if regex is None else regex)
        if expand:
            return self._frame(S.split_columns(parts))
        else:
            return [p for p in parts]

    def rsplit(self, pat: Optional[str] = None, n: int = -1, expand: Literal[bool] = False):
        parts = S.split_lists(self._chk(), pat, n, True, False)
        if expand:
            return self._frame(S.split_columns(parts))
        else:
            return [p for p in parts]

    def extract(self, pat: str, expand: Literal[bool] = True):
        cols = S.extract(self._chk(), pat)
        if expand:
            return self._frame(cols)
        else:
            return self._out(cols[0])

    def join(self, sep: str) -> Series:
        return self._out(S.smap(self._chk(), lambda x: sep.join([c for c in x])))


# ---------------------------------------------------------------------------
# datetime: to_datetime, date_range, Series.dt, resample, merge_asof

def to_datetime(arg, format: Optional[str] = None) -> Series:
    s = Series(arg)
    if s._col.kind == K_DT:
        return s
    return s._with(s._col.astype(K_DT))

def to_timedelta(arg) -> Series:
    s = Series(arg)
    return s._with(s._col.astype(K_TD))

def _dt_arg(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, Optional):
        return _dt_arg(x.__val__())
    if isinstance(x, str):
        return DT.parse_ts(x)
    if isinstance(x, Timestamp):
        return x.value
    return int(x)

def date_range(start = None, end = None, periods: Optional[int] = None, freq: str = "D",
               normalize: bool = False, name: Optional[str] = None) -> Index:
    vals = DT.date_range_ns(_dt_arg(start), _dt_arg(end), periods, freq, normalize)
    return Index._wrap(Column(K_DT, List[float](), vals, List[str]()), name, False)

def timedelta_range(start = None, end = None, periods: Optional[int] = None, freq: str = "D") -> Index:
    k, n, a = DT.parse_freq(freq)
    s0 = Timedelta(start).value if start is not None else 0
    out = List[int]()
    if periods is not None:
        for i in range(periods.__val__()):
            out.append(s0 + i * n)
    else:
        e0 = Timedelta(end).value
        cur = s0
        while cur <= e0:
            out.append(cur)
            cur += n
    return Index._wrap(Column(K_TD, List[float](), out, List[str]()), None, False)

class DatetimeProperties:
    s: Series

    def _f(self, f) -> Series:
        c = self.s._col
        return self.s._with(Column.ints([f(x) for x in c.i]))
    @property
    def days(self) -> Series:
        return self._f(lambda x: x // DT.NS_DAY)
    @property
    def seconds(self) -> Series:
        return self._f(lambda x: (x % DT.NS_DAY) // DT.NS_SEC)
    @property
    def microseconds(self) -> Series:
        return self._f(lambda x: (x % DT.NS_SEC) // 1000)
    @property
    def nanoseconds(self) -> Series:
        return self._f(lambda x: x % 1000)
    def total_seconds(self) -> Series:
        c = self.s._col
        return self.s._with(Column.floats([NAN if x == DT.NAT else float(x) / 1e9 for x in c.i]))


    def _c(self) -> Column:
        if self.s._col.kind != K_DT:
            raise AttributeError("Can only use .dt accessor with datetimelike values")
        return self.s._col

    def _field(self, f) -> Series:
        c = self._c()
        has_nat = False
        for x in c.i:
            if x == DT.NAT:
                has_nat = True
        if has_nat:
            return self.s._with(Column.floats([NAN if x == DT.NAT else float(f(x)) for x in c.i]))
        return self.s._with(Column.ints([f(x) for x in c.i]))

    @property
    def year(self) -> Series:
        return self._field(lambda x: DT.fields(x)[0])
    @property
    def month(self) -> Series:
        return self._field(lambda x: DT.fields(x)[1])
    @property
    def day(self) -> Series:
        return self._field(lambda x: DT.fields(x)[2])
    @property
    def hour(self) -> Series:
        return self._field(lambda x: DT.fields(x)[3])
    @property
    def minute(self) -> Series:
        return self._field(lambda x: DT.fields(x)[4])
    @property
    def second(self) -> Series:
        return self._field(lambda x: DT.fields(x)[5])
    @property
    def microsecond(self) -> Series:
        return self._field(lambda x: (x % DT.NS_SEC) // 1000)
    @property
    def nanosecond(self) -> Series:
        return self._field(lambda x: x % 1000)
    @property
    def dayofweek(self) -> Series:
        return self._field(lambda x: DT.dayofweek(x))
    @property
    def day_of_week(self) -> Series:
        return self.dayofweek
    @property
    def weekday(self) -> Series:
        return self.dayofweek
    @property
    def dayofyear(self) -> Series:
        return self._field(lambda x: DT.dayofyear(x))
    @property
    def day_of_year(self) -> Series:
        return self.dayofyear
    @property
    def quarter(self) -> Series:
        return self._field(lambda x: (DT.fields(x)[1] - 1) // 3 + 1)
    @property
    def days_in_month(self) -> Series:
        return self._field(lambda x: DT.days_in_month(DT.fields(x)[0], DT.fields(x)[1]))
    @property
    def is_leap_year(self) -> Series:
        c = self._c()
        return self.s._with(Column.bools([DT.is_leap(DT.fields(x)[0]) for x in c.i]))
    @property
    def is_month_start(self) -> Series:
        c = self._c()
        return self.s._with(Column.bools([DT.fields(x)[2] == 1 for x in c.i]))
    @property
    def is_month_end(self) -> Series:
        c = self._c()
        return self.s._with(Column.bools([DT.fields(x)[2] == DT.days_in_month(DT.fields(x)[0], DT.fields(x)[1]) for x in c.i]))
    def day_name(self) -> Series:
        c = self._c()
        return self.s._with(Column.strs([DT._DAYS[DT.dayofweek(x)] if x != DT.NAT else NA_STR for x in c.i]))
    def month_name(self) -> Series:
        c = self._c()
        return self.s._with(Column.strs([DT._MONTHS[DT.fields(x)[1] - 1] if x != DT.NAT else NA_STR for x in c.i]))
    def normalize(self) -> Series:
        c = self._c()
        return self.s._with(Column(K_DT, List[float](), [DT.floor_day(x) if x != DT.NAT else x for x in c.i], List[str]()))
    def strftime(self, fmt: str) -> Series:
        c = self._c()
        return self.s._with(Column.strs([DT.strftime_ns(x, fmt) if x != DT.NAT else NA_STR for x in c.i]))
    def floor(self, freq: str) -> Series:
        c = self._c()
        k, n, a = DT.parse_freq(freq)
        return self.s._with(Column(K_DT, List[float](), [(x // n) * n if x != DT.NAT else x for x in c.i], List[str]()))
    def ceil(self, freq: str) -> Series:
        c = self._c()
        k, n, a = DT.parse_freq(freq)
        return self.s._with(Column(K_DT, List[float](), [-((-x) // n) * n if x != DT.NAT else x for x in c.i], List[str]()))
    def round(self, freq: str) -> Series:
        c = self._c()
        k, n, a = DT.parse_freq(freq)
        out = List[int]()
        for x in c.i:
            if x == DT.NAT:
                out.append(x)
            else:
                q = x // n
                r = x - q * n
                if 2 * r > n or (2 * r == n and q % 2 == 1):
                    q += 1
                out.append(q * n)
        return self.s._with(Column(K_DT, List[float](), out, List[str]()))

def _resample_bins(idx: Column, freq: str, closed_right: bool, label_right: bool) -> Tuple[List[int], List[int]]:
    # returns (bin labels, code per row); rows outside bins get -1
    kind, n, arg = DT.parse_freq(freq)
    lo = DT.NAT
    hi = DT.NAT
    for x in idx.i:
        if x == DT.NAT:
            continue
        if lo == DT.NAT or x < lo:
            lo = x
        if hi == DT.NAT or x > hi:
            hi = x
    edges = List[int]()
    if kind == 0:
        # fixed tick: bins aligned to the origin (start_day) like pandas
        origin = DT.floor_day(lo) if n >= DT.NS_DAY or DT.NS_DAY % n == 0 else lo
        if n % DT.NS_DAY == 0:
            origin = DT.floor_day(lo)
        first = origin + ((lo - origin) // n) * n
        cur = first
        while cur <= hi:
            edges.append(cur)
            cur += n
        edges.append(cur)
        labs = edges[:-1]
        codes = List[int]()
        for x in idx.i:
            if x == DT.NAT:
                codes.append(-1)
            else:
                codes.append((x - first) // n)
        return (labs, codes)
    # anchored: bin edges are the anchors; the label is the right edge
    first_anchor = DT.roll_forward(DT.floor_day(lo), kind, arg)
    if kind == 1:
        first_anchor = DT.floor_day(lo)
    anchors = List[int]()
    cur = first_anchor
    if kind == 1:
        # business day: each bin is a day, weekends are folded forward onto Monday
        labs = List[int]()
        codes = List[int]()
        lab_pos = Dict[int, int]()
        for x in idx.i:
            d = DT.floor_day(x)
            wd = DT.dayofweek(d)
            if wd >= 5:
                d = d + (7 - wd) * DT.NS_DAY
            if d not in lab_pos:
                lab_pos[d] = len(labs)
                labs.append(d)
        order = list(range(len(labs)))
        order.sort(key=lambda t: labs[t])
        sorted_labs = [labs[t] for t in order]
        rank = Dict[int, int]()
        for t in range(len(sorted_labs)):
            rank[sorted_labs[t]] = t
        for x in idx.i:
            d = DT.floor_day(x)
            wd = DT.dayofweek(d)
            if wd >= 5:
                d = d + (7 - wd) * DT.NS_DAY
            codes.append(rank[d])
        return (sorted_labs, codes)
    # the anchor on/after lo is the first label; earlier anchors are not needed
    labs = List[int]()
    while True:
        labs.append(cur)
        if cur >= DT.floor_day(hi) + (DT.NS_DAY - 1) and cur >= hi:
            break
        cur = DT.step_anchor(cur, kind, n, arg)
    codes = List[int]()
    for x in idx.i:
        # first label whose bin (prev_label, label] contains x (daily resolution)
        pos = 0
        while pos < len(labs) and DT.floor_day(x) > labs[pos]:
            pos += 1
        codes.append(pos)
    return (labs, codes)

class Resampler:
    s: Series
    freq: str

    def _bins(self):
        return _resample_bins(self.s.index._col, self.freq, False, False)

    def _agg(self, how: str, ddof: int = 1) -> Series:
        labs, codes = self._bins()
        g = _Grouper.__new__()
        g.codes = codes
        g.keys = List[Scalar]()
        g.name = self.s.index.name
        g.levels = List[Column]()
        g.lnames = List[Optional[str]]()
        # only emit the bins in order (including empty ones, as pandas does)
        G = len(labs)
        for t in range(G):
            g.keys.append(Scalar(K_DT, 0.0, labs[t], ""))
        g.key_col = Column(K_DT, List[float](), labs, List[str]())
        col = _agg_col(g, self.s._col, how, ddof)
        if how == "sum":
            pass
        return Series._new(col, g.index(), self.s.name)

    def sum(self) -> Series:
        return self._agg("sum")
    def mean(self) -> Series:
        return self._agg("mean")
    def min(self) -> Series:
        return self._agg("min")
    def max(self) -> Series:
        return self._agg("max")
    def count(self) -> Series:
        return self._agg("count")
    def first(self) -> Series:
        return self._agg("first")
    def last(self) -> Series:
        return self._agg("last")
    def median(self) -> Series:
        return self._agg("median")
    def std(self) -> Series:
        return self._agg("std")
    def var(self) -> Series:
        return self._agg("var")
    def size(self) -> Series:
        return self._agg("size")

    def ohlc(self):
        o = self._agg("first")
        h = self._agg("max")
        l = self._agg("min")
        c = self._agg("last")
        return DataFrame._new([o._col, h._col, l._col, c._col], Index.strs(["open", "high", "low", "close"]), o.index)

def merge_asof(left: DataFrame, right: DataFrame, on: str, by: Optional[str] = None,
               direction: str = "backward", tolerance = None, suffixes: Tuple[str, str] = ("_x", "_y"),
               allow_exact_matches: bool = True):
    lj = left._col_pos(on)
    rj = right._col_pos(on)
    lk = left._cols[lj].to_floats()
    rk = right._cols[rj].to_floats()
    tol = float("inf")
    if tolerance is not None:
        if isinstance(tolerance, Timedelta):
            tol = float(tolerance.value)
        elif isinstance(tolerance, Optional):
            pass
        else:
            tol = float(tolerance)
    pos = List[int](len(lk))
    ropos = list(range(len(rk)))
    ropos.sort(key=lambda t: rk[t])
    bylj = -1
    byrj = -1
    if by is not None:
        bylj = left._col_pos(by.__val__())
        byrj = right._col_pos(by.__val__())
    for i in range(len(lk)):
        best = -1
        bestd = float("inf")
        for t in ropos:
            if by is not None:
                if left._cols[bylj].get(i) != right._cols[byrj].get(t):
                    continue
            d = lk[i] - rk[t]
            ok = False
            if direction == "backward":
                ok = d > 0.0 or (d == 0.0 and allow_exact_matches)
            elif direction == "forward":
                ok = d < 0.0 or (d == 0.0 and allow_exact_matches)
                d = -d
            else:
                ok = True
                d = abs(d)
            if ok and abs(d) <= tol:
                if direction == "backward" or direction == "nearest":
                    if abs(d) < bestd or (direction == "backward" and abs(d) <= bestd):
                        bestd = abs(d)
                        best = t
                else:
                    if abs(d) < bestd:
                        bestd = abs(d)
                        best = t
        pos.append(best)
    cols = List[Column]()
    labels = List[Scalar]()
    for j in range(len(left._cols)):
        cols.append(left._cols[j].copy())
        nm = left.columns[j]
        labels.append(nm if (j == lj or right.columns.get_loc_opt(nm) < 0) else Scalar.of(str(nm) + suffixes[0]))
    for j in range(len(right._cols)):
        if j == rj or (by is not None and j == byrj):
            continue
        cols.append(right._cols[j].take(pos))
        nm = right.columns[j]
        labels.append(nm if left.columns.get_loc_opt(nm) < 0 else Scalar.of(str(nm) + suffixes[1]))
    return DataFrame._new(cols, Index._wrap(Column.from_scalars(labels)), left.index.copy())
