# Ported from scipy 1.18.1 (BSD-3-Clause): scipy/interpolate/_bary_rational.py
# AAA and FloaterHormannInterpolator for real data (1-D values for AAA;
# FloaterHormann values along axis 0).  poles()/residues()/roots() and the
# Froissart clean-up of AAA are not provided (clean_up is accepted and
# ignored).  Weights are defined up to a common sign/scale, evaluations
# match scipy.

import numpy as np
from math import sqrt, fabs

def _prep(x, y):
    z = np.asarray(x, dtype=float)
    f = np.asarray(y, dtype=float)
    if z.ndim != 1:
        raise ValueError("`x` must be 1-D.")
    if z.size != f.shape[0]:
        raise ValueError(f"`x` be of size {f.shape[0]} but got size {z.size}.")
    F = np.ascontiguousarray(f).reshape((z.size, -1))
    keep = List[int]()
    for i in range(z.size):
        ok = True
        for j in range(F.shape[1]):
            if not np.isfinite(F[i, j]):
                ok = False
        if ok:
            keep.append(i)
    # unique (sorted) support points, keeping the first occurrence
    order = sorted(keep, key=lambda i: (z[i], i))
    zs = List[float]()
    rows = List[int]()
    for i in order:
        if len(zs) > 0 and zs[len(zs) - 1] == z[i]:
            continue
        zs.append(z[i])
        rows.append(i)
    Z = np.array(zs, dtype=float)
    FF = np.empty((len(rows), F.shape[1]))
    for a in range(len(rows)):
        FF[a] = F[rows[a]]
    return Z, FF, f.shape[1:]

def _bary_eval(zv, sp, sv, w):
    m = zv.size
    R = sv.shape[1]
    out = np.empty((m, R))
    for i in range(m):
        hit = -1
        for j in range(sp.size):
            if zv[i] == sp[j]:
                hit = j
                break
        if hit >= 0:
            for r in range(R):
                out[i, r] = sv[hit, r]
            continue
        if np.isinf(zv[i]):
            den = 0.0
            for j in range(sp.size):
                den += w[j]
            for r in range(R):
                num = 0.0
                for j in range(sp.size):
                    num += w[j] * sv[j, r]
                out[i, r] = num / den
            continue
        den = 0.0
        for j in range(sp.size):
            den += w[j] / (zv[i] - sp[j])
        for r in range(R):
            num = 0.0
            for j in range(sp.size):
                num += w[j] / (zv[i] - sp[j]) * sv[j, r]
            out[i, r] = num / den
    return out

class FloaterHormannInterpolator:
    def __init__(self, points, values, d: int = 3, axis: int = 0):
        if axis != 0:
            raise NotImplementedError("FloaterHormannInterpolator: only axis=0 is supported")
        z, f, shp = _prep(points, values)
        n = z.size
        if not (0 <= d and d < n):
            raise ValueError("`d` must satisfy 0 <= d < n")
        w = np.zeros(n)
        for k in range(n):
            for i in range(max(k - d, 0), min(k + 1, n - d)):
                p = 1.0
                for j in range(i, i + d + 1):
                    if j != k:
                        p *= abs(z[k] - z[j])
                w[k] += 1 / p
        for k in range(n):
            if (k - d) % 2 != 0:
                w[k] = -w[k]
        self._support_points = z
        self._support_values = f
        self.weights = w
        self._shape = shp

    def __call__(self, z):
        za = np.asarray(z, dtype=float)
        out = _bary_eval(np.ascontiguousarray(za.ravel()), self._support_points, self._support_values, self.weights)
        return out.reshape(za.shape + self._shape)

class AAA:
    def __init__(self, x, y, rtol: Optional[float] = None, max_terms: int = 100,
                 clean_up: bool = True, clean_up_tol: float = 1e-13):
        if max_terms < 1:
            raise ValueError("`max_terms` must be an integer value greater than or equal to one.")
        z, F, shp = _prep(x, y)
        f = F[:, 0].copy()
        M = z.size
        mask = [True] * M
        eps = 2.220446049250313e-16
        rt = eps ** 0.75 if rtol is None else rtol
        fmax = 0.0
        for i in range(M):
            fmax = max(fmax, abs(f[i]))
        atol = rt * fmax
        zj = np.empty(max_terms)
        fj = np.empty(max_terms)
        C = np.zeros((M, max_terms))
        A = np.zeros((M, max_terms))
        errors = np.empty(max_terms)
        mean = float(np.mean(f))
        Rv = np.full(M, mean)
        ill = False
        illtol = 1 / (3 * eps)
        wj = np.zeros(1)
        m = 0
        for mm in range(max_terms):
            m = mm
            best = -1
            bv = -1.0
            for i in range(M):
                if mask[i]:
                    v = abs(f[i] - Rv[i])
                    if v > bv:
                        bv = v
                        best = i
            zj[m] = z[best]
            fj[m] = f[best]
            for i in range(M):
                C[i, m] = 1 / (z[i] - z[best]) if i != best else np.inf
            mask[best] = False
            for i in range(M):
                A[i, m] = (f[i] - fj[m]) * C[i, m] if i != best else np.nan
            rows = List[int]()
            for i in range(M):
                if mask[i]:
                    rows.append(i)
            nr = len(rows)
            if nr >= m + 1:
                Am = np.empty((nr, m + 1))
                for a in range(nr):
                    for b in range(m + 1):
                        Am[a, b] = A[rows[a], b]
                U, s, Vt = np.linalg.svd(Am, full_matrices=False)
                cn = np.ones(m + 1)
                if not ill:
                    if s[0] / s[s.size - 1] > illtol:
                        ill = True
                if ill:
                    for b in range(m + 1):
                        nn = 0.0
                        for a in range(nr):
                            nn += Am[a, b] ** 2
                        cn[b] = sqrt(nn)
                    for a in range(nr):
                        for b in range(m + 1):
                            Am[a, b] /= cn[b]
                    U, s, Vt = np.linalg.svd(Am, full_matrices=False)
                smin = float(np.min(s))
                cnt = 0
                wj = np.zeros(m + 1)
                for k in range(s.size):
                    if s[k] == smin:
                        cnt += 1
                        for b in range(m + 1):
                            wj[b] += Vt[k, b]
                for b in range(m + 1):
                    wj[b] /= sqrt(float(cnt))
                if ill:
                    for b in range(m + 1):
                        wj[b] /= cn[b]
            else:
                Am = np.empty((max(nr, 1), m + 1))
                if nr == 0:
                    Am = np.zeros((1, m + 1))
                else:
                    for a in range(nr):
                        for b in range(m + 1):
                            Am[a, b] = A[rows[a], b]
                U, s, Vt = np.linalg.svd(Am, full_matrices=True)
                rcond = max(Am.shape[0], Am.shape[1]) * eps
                tolv = (float(np.max(s)) if s.size > 0 else 0.0) * rcond
                rank = 0
                for k in range(s.size):
                    if s[k] > tolv:
                        rank += 1
                nm = (m + 1) - rank
                wj = np.zeros(m + 1)
                for k in range(rank, m + 1):
                    for b in range(m + 1):
                        wj[b] += Vt[k, b]
                for b in range(m + 1):
                    wj[b] /= sqrt(float(nm))
            maxerr = 0.0
            for i in range(M):
                N = 0.0
                Dd = 0.0
                for b in range(m + 1):
                    if wj[b] != 0:
                        N += C[i, b] * wj[b] * fj[b]
                        Dd += C[i, b] * wj[b]
                r = 0.0
                if np.isinf(Dd) or np.isnan(Dd):
                    r = f[i]
                else:
                    r = N / Dd
                Rv[i] = r
                maxerr = max(maxerr, abs(f[i] - r))
            errors[m] = maxerr
            if maxerr <= atol:
                break
        nz = List[int]()
        for b in range(m + 1):
            if wj[b] != 0:
                nz.append(b)
        sp = np.empty(len(nz))
        sv = np.empty((len(nz), 1))
        w = np.empty(len(nz))
        for a in range(len(nz)):
            sp[a] = zj[nz[a]]
            sv[a, 0] = fj[nz[a]]
            w[a] = wj[nz[a]]
        self.errors = errors[:m + 1].copy()
        self._points = z
        self._values = f
        self._support_points = sp
        self._support_values = sv
        self.weights = w
        self._shape = shp

    @property
    def support_points(self):
        return self._support_points

    @property
    def support_values(self):
        return self._support_values[:, 0].copy()

    def __call__(self, z):
        za = np.asarray(z, dtype=float)
        out = _bary_eval(np.ascontiguousarray(za.ravel()), self._support_points, self._support_values, self.weights)
        return out.reshape(za.shape)
