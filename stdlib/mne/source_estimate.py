# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.source_estimate core: SourceEstimate / VolSourceEstimate and their
# vector variants (data, vertices, tmin, tstep, times, crop, resample, mean,
# sum, arithmetic, magnitude, copy).

import numpy as np
from ._info import _time_mask


class _STC[D]:
    """Common source-estimate implementation; D is the data array type
    (2-D for scalar estimates, 3-D (n_vertices, 3, n_times) for vector ones)."""
    _data: D
    vertices: List[np.ndarray[int, 1]]
    _tmin: float
    _tstep: float
    subject: Optional[str]
    kind: str   # 'surface' | 'volume'

    def __init__(self, data: D, vertices, tmin: float, tstep: float, subject = None,
                 kind: str = "volume"):
        vs = List[np.ndarray[int, 1]]()
        if isinstance(vertices, np.ndarray):
            vs.append(vertices.astype(int))
        else:
            for v in vertices:
                vs.append(np.asarray(v).astype(int))
        n = 0
        for v in vs:
            n += v.shape[0]
        if data.shape[0] != n:
            raise ValueError(f"Number of vertices ({n}) and stc.data.shape[0] ({data.shape[0]}) must match")
        if tstep <= 0:
            raise ValueError(".tstep must be greater than 0.")
        self._data = data
        self.vertices = vs
        self._tmin = float(tmin)
        self._tstep = float(tstep)
        self.subject = None if isinstance(subject, NoneType) else str(subject)
        self.kind = kind

    @property
    def data(self) -> D:
        return self._data

    @data.setter
    def data(self, value: D):
        n = 0
        for v in self.vertices:
            n += v.shape[0]
        if value.shape[0] != n:
            raise ValueError(f"The first dimension of the data array must match the number of vertices ({value.shape[0]} != {n}).")
        self._data = value

    @property
    def shape(self):
        return self._data.shape

    @property
    def tmin(self) -> float:
        return self._tmin

    @tmin.setter
    def tmin(self, value: float):
        self._tmin = float(value)

    @property
    def tstep(self) -> float:
        return self._tstep

    @tstep.setter
    def tstep(self, value: float):
        if value <= 0:
            raise ValueError(".tstep must be greater than 0.")
        self._tstep = float(value)

    @property
    def sfreq(self) -> float:
        return 1.0 / self._tstep

    @property
    def times(self):
        return self._tmin + self._tstep * np.arange(self._data.shape[-1]).astype(float)

    @property
    def tmax(self) -> float:
        return self._tmin + self._tstep * (self._data.shape[-1] - 1)

    def _n_vertices(self) -> int:
        n = 0
        for v in self.vertices:
            n += v.shape[0]
        return n

    def copy(self):
        return _STC[D](self._data.copy(), [v.copy() for v in self.vertices], self._tmin,
                       self._tstep, self.subject, self.kind)

    def __repr__(self) -> str:
        nm = self.__class__.__name__
        return f"<{nm} | {self._n_vertices()} vertices, tmin : {1e3 * self._tmin} (ms), tmax : {1e3 * self.tmax} (ms), tstep : {1e3 * self._tstep} (ms), data shape : {self._data.shape}>"

    def crop(self, tmin = None, tmax = None, include_tmax: bool = True):
        tmn: Optional[float] = None if isinstance(tmin, NoneType) else float(tmin)
        tmx: Optional[float] = None if isinstance(tmax, NoneType) else float(tmax)
        times = self.times
        mask = _time_mask(times, tmn, tmx, self.sfreq, True, include_tmax)
        idx = np.array([i for i in range(times.shape[0]) if mask[i]], dtype=int)
        self._tmin = float(times[int(idx[0])])
        self._data = self._data[..., idx].copy()
        return self

    def resample(self, sfreq: float, *, npad = 100, method: str = "fft", window = "auto",
                 pad: str = "auto", n_jobs = None, verbose = None):
        from .filter import resample, _check_resamp_noop
        o_sfreq = 1.0 / self._tstep
        if _check_resamp_noop(sfreq, o_sfreq):
            return self
        d = self._data
        shp = d.shape
        flat = d.reshape(-1, shp[-1])
        out = resample(flat, float(sfreq), o_sfreq, window=window, pad=pad, npad=npad, method=method)
        if d.ndim == 2:
            self._data = out.reshape(shp[0], out.shape[-1])
        else:
            self._data = out.reshape(shp[0], shp[1], out.shape[-1])
        self._tstep = 1.0 / sfreq
        return self

    def sum(self):
        d = self._data
        tmax = self._tmin + self._tstep * d.shape[-1]
        tmin = (self._tmin + tmax) / 2.0
        tstep = tmax - self._tmin
        return _STC[D](d.sum(axis=-1, keepdims=True), [v.copy() for v in self.vertices], tmin,
                       tstep, self.subject, self.kind)

    def mean(self):
        out = self.sum()
        out._data = out._data / float(self._data.shape[-1])
        return out

    def _check_compat(self, a):
        if len(self.vertices) != len(a.vertices):
            raise ValueError("Cannot combine source estimates that do not have the same vertices. Consider using stc.expand().")
        for i in range(len(self.vertices)):
            if not np.array_equal(self.vertices[i], a.vertices[i]):
                raise ValueError("Cannot combine source estimates that do not have the same vertices. Consider using stc.expand().")
        if self.subject != a.subject:
            raise ValueError("source estimates do not have the same subject names")

    def _binop(self, a, op: Literal[str]):
        out = self.copy()
        if isinstance(a, _STC[D]):
            self._check_compat(a)
            b = a._data
            if op == "+":
                out._data = out._data + b
            elif op == "-":
                out._data = out._data - b
            elif op == "*":
                out._data = out._data * b
            else:
                out._data = out._data / b
        else:
            b = float(a)
            if op == "+":
                out._data = out._data + b
            elif op == "-":
                out._data = out._data - b
            elif op == "*":
                out._data = out._data * b
            else:
                out._data = out._data / b
        return out

    def __add__(self, a):
        return self._binop(a, "+")

    def __sub__(self, a):
        return self._binop(a, "-")

    def __mul__(self, a):
        return self._binop(a, "*")

    def __truediv__(self, a):
        return self._binop(a, "/")

    def __radd__(self, a):
        return self._binop(a, "+")

    def __rmul__(self, a):
        return self._binop(a, "*")

    def __neg__(self):
        out = self.copy()
        out._data = -out._data
        return out

    def __abs__(self):
        out = self.copy()
        out._data = np.abs(out._data)
        return out

    def __pow__(self, a: float):
        out = self.copy()
        out._data = out._data ** a
        return out

    def sqrt(self):
        out = self.copy()
        out._data = np.sqrt(out._data)
        return out

    def magnitude(self):
        """Vector estimates: norm over the xyz axis (returns a scalar STC)."""
        if self._data.ndim != 3:
            compile_error("magnitude() is only defined for vector source estimates")
        m = np.sqrt(np.sum(self._data * self._data, axis=1))
        return _STC[np.ndarray[float, 2]](m, [v.copy() for v in self.vertices], self._tmin,
                                          self._tstep, self.subject, self.kind)

    def apply_baseline(self, baseline = (None, 0), *, verbose = None):
        times = self.times
        b0 = float(times[0]) if isinstance(baseline[0], NoneType) else float(baseline[0])
        b1 = float(times[-1]) if isinstance(baseline[1], NoneType) else float(baseline[1])
        mask = (times >= b0) & (times <= b1)
        mean = self._data[..., mask].mean(axis=-1, keepdims=True)
        self._data = self._data - mean
        return self

    def get_peak(self, tmin = None, tmax = None, mode: str = "abs",
                 vert_as_index: bool = False, time_as_index: bool = False):
        d = self._data
        if d.ndim != 2:
            compile_error("get_peak is only available for scalar source estimates")
        times = self.times
        tmn = float(times[0]) if isinstance(tmin, NoneType) else float(tmin)
        tmx = float(times[-1]) if isinstance(tmax, NoneType) else float(tmax)
        best = -np.inf
        bv = 0
        bt = 0
        for t in range(times.shape[0]):
            if times[t] < tmn or times[t] > tmx:
                continue
            for v in range(d.shape[0]):
                x = d[v, t]
                val = abs(x) if mode == "abs" else (x if mode == "pos" else -x)
                if val > best:
                    best = val
                    bv = v
                    bt = t
        allv = np.concatenate(self.vertices)
        vo = bv if vert_as_index else int(allv[bv])
        if time_as_index:
            return vo, float(bt)
        return vo, float(times[bt])


SourceEstimate = _STC[np.ndarray[float, 2]]


def VolSourceEstimate(data, vertices, tmin: float, tstep: float, subject = None, verbose = None):
    return _STC[np.ndarray[float, 2]](np.asarray(data, dtype=float), vertices, tmin, tstep, subject, "volume")


def VolVectorSourceEstimate(data, vertices, tmin: float, tstep: float, subject = None, verbose = None):
    return _STC[np.ndarray[float, 3]](np.asarray(data, dtype=float), vertices, tmin, tstep, subject, "volume")


def _make_stc(data, vertices, tmin: float, tstep: float, subject: Optional[str], kind: str):
    return _STC[type(data)](data, vertices, tmin, tstep, subject, kind)
