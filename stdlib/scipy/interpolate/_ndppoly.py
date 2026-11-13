# Ported from scipy 1.18.1 (BSD-3-Clause): NdPPoly (scipy/interpolate/
# _interpolate.py, evaluate_nd from _ppoly.pyx).  Supported: __call__ (with
# derivative orders nu) and derivative(); antiderivative/integrate are not
# provided.  Real coefficients only.

import numpy as np
import internal.static as static
from _common import NAN, _pp_find

def _poch(a: int, m: int) -> float:
    r = 1.0
    for i in range(m):
        r *= float(a - i)
    return r

def _shape_out(out, lead, c, ntrail: Literal[int]):
    if ntrail == 0:
        return out.reshape(lead)
    else:
        return out.reshape(lead + c.shape[c.ndim - ntrail:])

class NdPPoly:
    def __init__(self, c, x, extrapolate = None):
        xs = tuple(np.ascontiguousarray(np.asarray(v, dtype=float)) for v in x)
        ca = np.ascontiguousarray(np.asarray(c, dtype=float))
        nd = len(xs)
        if ca.ndim < 2 * nd:
            raise ValueError("c must have at least 2*len(x) dimensions")
        for d in range(nd):
            if xs[d].size < 2:
                raise ValueError("x arrays must all contain at least 2 points")
            for i in range(xs[d].size - 1):
                if xs[d][i + 1] - xs[d][i] < 0:
                    raise ValueError("x-coordinates are not in increasing order")
            if ca.shape[nd + d] != xs[d].size - 1:
                raise ValueError("x and c do not agree on the number of intervals")
        self.x = xs
        self.c = ca
        self.extrapolate = True if extrapolate is None else bool(extrapolate)

    def _dims(self):
        nd = len(self.x)
        sh = [self.c.shape[i] for i in range(self.c.ndim)]
        ks = sh[:nd]
        ms = sh[nd:2 * nd]
        rest = 1
        for v in sh[2 * nd:]:
            rest *= v
        return nd, ks, ms, rest

    def __call__(self, x, nu = None, extrapolate = None):
        ext = self.extrapolate if extrapolate is None else bool(extrapolate)
        nd, ks, ms, rest = self._dims()
        xa = np.asarray(x, dtype=float)
        X = np.ascontiguousarray(xa).reshape((-1, nd))
        nus = [0] * nd
        if nu is not None:
            nus = [int(v) for v in nu]
            if len(nus) != nd:
                raise ValueError("invalid number of derivative orders nu")
        C = self.c.reshape((-1,))
        # strides of the flattened c (k1..kn, m1..mn, rest)
        dims = ks + ms + [rest]
        strides = [0] * len(dims)
        s = 1
        for i in range(len(dims) - 1, -1, -1):
            strides[i] = s
            s *= dims[i]
        out = np.empty((X.shape[0], rest))
        for p in range(X.shape[0]):
            ivs = [0] * nd
            bad = False
            for d in range(nd):
                iv = _pp_find(self.x[d], X[p, d], ext)
                if iv < 0:
                    bad = True
                ivs[d] = iv
            if bad:
                for r in range(rest):
                    out[p, r] = NAN
                continue
            # per-dimension power weights w_d[k] for coefficient index k
            ws = List[List[float]]()
            for d in range(nd):
                K = ks[d]
                s0 = X[p, d] - self.x[d][ivs[d]]
                wd = [0.0] * K
                for k in range(K):
                    e = K - 1 - k
                    if e < nus[d]:
                        wd[k] = 0.0
                    else:
                        wd[k] = _poch(e, nus[d]) * s0 ** (e - nus[d])
                ws.append(wd)
            base = 0
            for d in range(nd):
                base += ivs[d] * strides[nd + d]
            for r in range(rest):
                tot = 0.0
                idx = [0] * nd
                while True:
                    w = 1.0
                    off = base + r
                    for d in range(nd):
                        w *= ws[d][idx[d]]
                        off += idx[d] * strides[d]
                    if w != 0.0:
                        tot += w * C[off]
                    d = nd - 1
                    while d >= 0:
                        idx[d] += 1
                        if idx[d] < ks[d]:
                            break
                        idx[d] = 0
                        d -= 1
                    if d < 0:
                        break
                out[p, r] = tot
        return _shape_out(out, xa.shape[:-1], self.c, static.len(self.c.shape) - 2 * static.len(self.x))

    def derivative(self, nu):
        nd, ks, ms, rest = self._dims()
        nus = [int(v) for v in nu]
        c = self.c
        for d in range(nd):
            n = nus[d]
            if n == 0:
                continue
            K = c.shape[d]
            m = max(K - n, 1)
            cm = np.moveaxis(c, d, 0)
            new = np.zeros((m,) + cm.shape[1:])
            if K - n > 0:
                for k in range(K - n):
                    new[k] = cm[k] * _poch(K - 1 - k, n)
            c = np.ascontiguousarray(np.moveaxis(new, 0, d))
        return NdPPoly(c, self.x, self.extrapolate)
