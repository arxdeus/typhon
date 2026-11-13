# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# random_table, matrix_t, normal_inverse_gamma, unitary_group.

import numpy as np
import scipy.special._kernels as K
import scipy.linalg as LA
from scipy.special import multigammaln
from numpy.random.logfactorial import logfactorial
from ._distn_infrastructure import check_random_state, NAN, INF
from ._multivariate import invwishart
from ._continuous_distns import invgamma, norm

_LOG_PI = 1.1447298858494002

def _bg(rs):
    if hasattr(rs, '_bitgen'):
        return rs._bitgen
    else:
        return rs.bit_generator

# ------------------------------------------------------------ random_table

def _rcont1(row: np.ndarray[int, 1], col: np.ndarray[int, 1], ntot: int, size: int, rs):
    nr = row.shape[0]
    nc = col.shape[0]
    result = np.zeros((size, nr, nc), dtype=int)
    if nc == 0 or nr == 0 or ntot == 0:
        return result
    bg = _bg(rs)
    work = np.empty(ntot, dtype=int)
    p = 0
    for i in range(nc):
        for _ in range(int(col[i])):
            work[p] = i
            p += 1
    for s in range(size):
        i = ntot - 1
        while i > 0:
            j = int(bg.random_interval(u64(i)))
            tmp = work[j]
            work[j] = work[i]
            work[i] = tmp
            i -= 1
        q = 0
        for ir in range(nr):
            for _ in range(int(row[ir])):
                result[s, ir, int(work[q])] += 1
                q += 1
    return result

def _rcont2_one(table: np.ndarray[int, 2], r: np.ndarray[int, 1], c: np.ndarray[int, 1],
                ntot: int, bg):
    nr = r.shape[0]
    nc = c.shape[0]
    jwork = np.zeros(nc, dtype=int)
    for i in range(nc - 1):
        jwork[i] = c[i]
    jc = ntot
    ib = 0
    for l in range(nr - 1):
        ia = int(r[l])
        if ia == 0:
            for i in range(nc):
                table[l, i] = 0
            continue
        ic = jc
        jc -= int(r[l])
        m = 0
        broke = False
        while m < nc - 1:
            id_ = int(jwork[m])
            ie = ic
            ic -= id_
            ib = ie - ia
            if c[m] == 0:
                for i in range(nr):
                    table[i, m] = 0
                m += 1
                continue
            ii = ib - id_
            if ie == 0:
                for j in range(m, nc - 1):
                    table[l, j] = 0
                ia = 0
                broke = True
                break
            z = bg.next_double()
            nlm = 0
            # state machine of the goto-based original
            state = 131
            x = 0.0
            y = 0.0
            sumprb = 0.0
            nll = 0
            lsp = False
            lsm = False
            while True:
                if state == 131:
                    nlm = int(K.floor(float(ia * id_) / ie + 0.5))
                    x = K.exp(logfactorial(ia) + logfactorial(ib) + logfactorial(ic) + logfactorial(id_)
                              - logfactorial(ie) - logfactorial(nlm) - logfactorial(id_ - nlm)
                              - logfactorial(ia - nlm) - logfactorial(ii + nlm))
                    if x >= z:
                        state = 160
                        continue
                    sumprb = x
                    y = x
                    nll = nlm
                    lsp = False
                    lsm = False
                    state = 140
                elif state == 140:
                    j = (id_ - nlm) * (ia - nlm)
                    if j == 0:
                        state = 156
                        continue
                    nlm += 1
                    x *= float(j) / (nlm * (ii + nlm))
                    sumprb += x
                    if sumprb >= z:
                        state = 160
                        continue
                    state = 150
                elif state == 150:
                    if lsm:
                        state = 155
                        continue
                    j = nll * (ii + nll)
                    if j == 0:
                        state = 154
                        continue
                    nll -= 1
                    y *= float(j) / ((id_ - nll) * (ia - nll))
                    sumprb += y
                    if sumprb >= z:
                        state = 159
                        continue
                    if not lsp:
                        state = 140
                        continue
                    state = 150
                elif state == 154:
                    lsm = True
                    state = 155
                elif state == 155:
                    if not lsp:
                        state = 140
                        continue
                    z = bg.next_double() * sumprb
                    state = 131
                elif state == 156:
                    lsp = True
                    state = 150
                elif state == 159:
                    nlm = nll
                    state = 160
                else:
                    break
            table[l, m] = nlm
            ia -= nlm
            jwork[m] -= nlm
            m += 1
        if not broke:
            table[l, nc - 1] = ia
    # jwork is folded into the last table row in the original
    for i in range(nc - 1):
        table[nr - 1, i] = jwork[i]
    table[nr - 1, nc - 1] = ib - table[nr - 1, nc - 2]

def _rcont2(row: np.ndarray[int, 1], col: np.ndarray[int, 1], ntot: int, size: int, rs):
    nr = row.shape[0]
    nc = col.shape[0]
    result = np.zeros((size, nr, nc), dtype=int)
    if nc == 0 or nr == 0 or ntot == 0:
        return result
    bg = _bg(rs)
    for s in range(size):
        t = np.zeros((nr, nc), dtype=int)
        _rcont2_one(t, row, col, ntot, bg)
        result[s] = t
    return result

class random_table_frozen:
    row: np.ndarray[int, 1]
    col: np.ndarray[int, 1]
    n: int

    def __init__(self, row, col):
        r = np.asarray(row).astype(int).ravel()
        c = np.asarray(col).astype(int).ravel()
        if r.shape[0] < 1 or c.shape[0] < 1:
            raise ValueError("`row` and `col` must be one-dimensional")
        for i in range(r.shape[0]):
            if r[i] < 0:
                raise ValueError("each element of `row` must be non-negative")
        for i in range(c.shape[0]):
            if c[i] < 0:
                raise ValueError("each element of `col` must be non-negative")
        n = int(np.sum(r))
        if n != int(np.sum(c)):
            raise ValueError("sums over `row` and `col` must be equal")
        self.row = r
        self.col = c
        self.n = n

    def mean(self):
        return np.outer(self.row, self.col).astype(float) / self.n

    def logpmf(self, x):
        xa = np.asarray(x).astype(int)
        if xa.ndim != 2:
            raise ValueError("`x` must be two-dimensional in this port")
        nr = self.row.shape[0]
        nc = self.col.shape[0]
        if xa.shape[0] != nr:
            raise ValueError("shape of `x` must agree with `row`")
        if xa.shape[1] != nc:
            raise ValueError("shape of `x` must agree with `col`")
        for i in range(nr):
            for j in range(nc):
                if xa[i, j] < 0:
                    raise ValueError("`x` must contain only non-negative values")
        for i in range(nr):
            s = 0
            for j in range(nc):
                s += int(xa[i, j])
            if s != int(self.row[i]):
                return -INF
        for j in range(nc):
            s = 0
            for i in range(nr):
                s += int(xa[i, j])
            if s != int(self.col[j]):
                return -INF
        tot = 0.0
        for i in range(nr):
            tot += K.gammaln(self.row[i] + 1.0)
        for j in range(nc):
            tot += K.gammaln(self.col[j] + 1.0)
        tot -= K.gammaln(self.n + 1.0)
        for i in range(nr):
            for j in range(nc):
                tot -= K.gammaln(xa[i, j] + 1.0)
        return tot

    def pmf(self, x):
        return K.exp(self.logpmf(x))

    def rvs(self, size: int = 1, method: str = '', random_state = None):
        rs = check_random_state(random_state)
        k = self.row.shape[0] * self.col.shape[0]
        use_pat = self.n > 1.0 * K.log(self.n + 1.0) * k
        if method == 'patefield':
            use_pat = True
        elif method == 'boyett':
            use_pat = False
        elif method != '':
            raise ValueError("'method' not recognized, must be 'boyett' or 'patefield'")
        r = _rcont2(self.row, self.col, self.n, size, rs) if use_pat else \
            _rcont1(self.row, self.col, self.n, size, rs)
        if size == 1:
            return r[0].copy()
        return r

class random_table_gen:
    def __call__(self, row, col):
        return random_table_frozen(row, col)
    def mean(self, row, col): return random_table_frozen(row, col).mean()
    def logpmf(self, x, row, col): return random_table_frozen(row, col).logpmf(x)
    def pmf(self, x, row, col): return random_table_frozen(row, col).pmf(x)
    def rvs(self, row, col, *, size: int = 1, method: str = '', random_state = None):
        return random_table_frozen(row, col).rvs(size, method, random_state)

random_table = random_table_gen()

# ----------------------------------------------------- normal_inverse_gamma

class normal_inverse_gamma_frozen:
    mu: float
    lmbda: float
    a: float
    b: float
    def __init__(self, mu: float = 0.0, lmbda: float = 1.0, a: float = 1.0, b: float = 1.0):
        self.mu = float(mu)
        self.lmbda = float(lmbda)
        self.a = float(a)
        self.b = float(b)

    def _valid(self) -> bool:
        return self.lmbda > 0 and self.a > 0 and self.b > 0

    def logpdf(self, x: float, s2: float) -> float:
        if not self._valid():
            return NAN
        if s2 <= 0:
            return -INF
        t1 = 0.5 * (K.log(self.lmbda) - K.log(2 * 3.141592653589793 * s2))
        t2 = self.a * K.log(self.b) - K.gammaln(self.a)
        t3 = -(self.a + 1) * K.log(s2)
        t4 = -(2 * self.b + self.lmbda * (x - self.mu) ** 2) / (2 * s2)
        return t1 + t2 + t3 + t4

    def pdf(self, x: float, s2: float) -> float:
        if not self._valid():
            return NAN
        if s2 <= 0:
            return 0.0
        t1 = K.sqrt(self.lmbda / (2 * 3.141592653589793 * s2))
        t2 = self.b ** self.a / K.gamma(self.a)
        t3 = (1 / s2) ** (self.a + 1)
        t4 = K.exp(-(2 * self.b + self.lmbda * (x - self.mu) ** 2) / (2 * s2))
        return t1 * t2 * t3 * t4

    def mean(self):
        if not self._valid() or not (self.a > 1):
            return (NAN, NAN)
        return (self.mu, self.b / (self.a - 1))

    def var(self):
        if not self._valid():
            return (NAN, NAN)
        vx = self.b / ((self.a - 1) * self.lmbda) if self.a > 1 else NAN
        vs = self.b ** 2 / ((self.a - 1) ** 2 * (self.a - 2)) if self.a > 2 else NAN
        return (vx, vs)

    def rvs(self, size = None, random_state = None):
        rs = check_random_state(random_state)
        s2 = invgamma.rvs(self.a, scale=self.b, size=size, random_state=rs)
        scale = (s2 / self.lmbda) ** 0.5
        x = norm.rvs(loc=self.mu, scale=scale, size=size, random_state=rs)
        return (x, s2)

class normal_inverse_gamma_gen:
    def __call__(self, mu = 0.0, lmbda = 1.0, a = 1.0, b = 1.0):
        return normal_inverse_gamma_frozen(mu, lmbda, a, b)
    def logpdf(self, x, s2, mu = 0.0, lmbda = 1.0, a = 1.0, b = 1.0):
        return self(mu, lmbda, a, b).logpdf(float(x), float(s2))
    def pdf(self, x, s2, mu = 0.0, lmbda = 1.0, a = 1.0, b = 1.0):
        return self(mu, lmbda, a, b).pdf(float(x), float(s2))
    def mean(self, mu = 0.0, lmbda = 1.0, a = 1.0, b = 1.0): return self(mu, lmbda, a, b).mean()
    def var(self, mu = 0.0, lmbda = 1.0, a = 1.0, b = 1.0): return self(mu, lmbda, a, b).var()
    def rvs(self, mu = 0.0, lmbda = 1.0, a = 1.0, b = 1.0, size = None, random_state = None):
        return self(mu, lmbda, a, b).rvs(size, random_state)

normal_inverse_gamma = normal_inverse_gamma_gen()

# ----------------------------------------------------------------- matrix_t

def _mt_spread(s, n: int) -> np.ndarray[float, 2]:
    a = np.asarray(s).astype(float)
    if a.ndim == 0:
        return float(a.item()) * np.eye(n)
    elif a.ndim == 1:
        return np.diag(a)
    return a

class matrix_t_frozen:
    mean: np.ndarray[float, 2]
    row_spread: np.ndarray[float, 2]
    col_spread: np.ndarray[float, 2]
    df: float
    invrow: np.ndarray[float, 2]
    invcol: np.ndarray[float, 2]
    logdetrow: float
    logdetcol: float

    def __init__(self, mean = None, row_spread = 1.0, col_spread = 1.0, df = 1.0):
        m = 1
        n = 1
        if mean is not None:
            ma = np.asarray(mean).astype(float)
            if ma.ndim != 2 or ma.shape[0] == 0 or ma.shape[1] == 0:
                raise ValueError("Array `mean` must be 2D.")
            m = ma.shape[0]
            n = ma.shape[1]
            self.mean = ma
        else:
            rs0 = np.asarray(row_spread).astype(float)
            cs0 = np.asarray(col_spread).astype(float)
            m = rs0.shape[0] if rs0.ndim > 0 else 1
            n = cs0.shape[0] if cs0.ndim > 0 else 1
            self.mean = np.zeros((m, n))
        self.row_spread = _mt_spread(row_spread, m)
        self.col_spread = _mt_spread(col_spread, n)
        if self.row_spread.shape[0] != m or self.row_spread.shape[1] != m:
            raise ValueError("Arrays `mean` and `row_spread` must have the same number of rows.")
        if self.col_spread.shape[0] != n or self.col_spread.shape[1] != n:
            raise ValueError("Arrays `mean` and `col_spread` must have the same number of columns.")
        d = float(df)
        if d <= 0:
            raise ValueError("Degrees of freedom must be positive.")
        self.df = d
        self.invrow = np.linalg.inv(self.row_spread)
        self.invcol = np.linalg.inv(self.col_spread)
        self.logdetrow = float(np.linalg.slogdet(self.row_spread)[1])
        self.logdetcol = float(np.linalg.slogdet(self.col_spread)[1])

    def logpdf(self, X):
        Xa = np.asarray(X).astype(float)
        m = self.mean.shape[0]
        n = self.mean.shape[1]
        if Xa.ndim == 2:
            return self._lp(Xa)
        lead = Xa.shape[:-2]
        flat = Xa.reshape((-1, m, n))
        out = np.empty(flat.shape[0], dtype=float)
        for i in range(flat.shape[0]):
            out[i] = self._lp(np.ascontiguousarray(flat[i]))
        return out.reshape(lead)

    def _lp(self, X: np.ndarray[float, 2]) -> float:
        m = self.mean.shape[0]
        n = self.mean.shape[1]
        Xc = X - self.mean
        det_arg = np.eye(n) + Xc.T @ self.invrow @ Xc @ self.invcol
        logdet = float(np.linalg.slogdet(det_arg)[1])
        df = self.df
        log_d_mn = -((df + m + n - 1) / 2) * logdet
        log_f_mn = (multigammaln((df + m + n - 1) / 2, n) - multigammaln((df + n - 1) / 2, n)
                    - (m * n / 2) * _LOG_PI - (n / 2) * self.logdetrow - (m / 2) * self.logdetcol)
        return log_d_mn + log_f_mn

    def pdf(self, X):
        return np.exp(self.logpdf(X))

    def rvs(self, size: int = 1, random_state = None):
        rs = check_random_state(random_state)
        m = self.mean.shape[0]
        n = self.mean.shape[1]
        std_norm = np.asarray(rs.standard_normal(size=(n, size, m))).transpose(1, 2, 0)
        out = np.empty((size, m, n), dtype=float)
        if m <= n:
            iw = self._iw_chol(self.row_spread, size, rs)
            colchol = LA.cholesky(self.col_spread, lower=True)
            for i in range(size):
                out[i] = self.mean + iw[i] @ std_norm[i] @ colchol.T
        else:
            rowchol = LA.cholesky(self.row_spread, lower=True)
            iw = self._iw_chol(self.col_spread, size, rs)
            for i in range(size):
                out[i] = self.mean + rowchol @ std_norm[i] @ iw[i].T
        if size == 1:
            return out[0].copy()
        return out

    def _iw_chol(self, scale: np.ndarray[float, 2], size: int, rs):
        d = scale.shape[0]
        df_iw = self.df + d - 1
        smp = invwishart.rvs(df_iw, scale, size, rs)
        if size == 1:
            smp = smp.reshape((1, d, d))
        out = np.empty((size, d, d), dtype=float)
        for i in range(size):
            out[i] = LA.cholesky(np.ascontiguousarray(smp[i]), lower=True)
        return out

class matrix_t_gen:
    def __call__(self, mean = None, row_spread = 1.0, col_spread = 1.0, df = 1.0):
        return matrix_t_frozen(mean, row_spread, col_spread, df)
    def logpdf(self, X, mean = None, row_spread = 1.0, col_spread = 1.0, df = 1.0):
        return self(mean, row_spread, col_spread, df).logpdf(X)
    def pdf(self, X, mean = None, row_spread = 1.0, col_spread = 1.0, df = 1.0):
        return self(mean, row_spread, col_spread, df).pdf(X)
    def rvs(self, mean = None, row_spread = 1.0, col_spread = 1.0, df = 1.0, size: int = 1,
            random_state = None):
        return self(mean, row_spread, col_spread, df).rvs(size, random_state)

matrix_t = matrix_t_gen()

# ----------------------------------------------------------- unitary_group

def _unitary_rvs(dim: int, rs):
    # complex Gram-Schmidt QR (column phases normalised so that diag(R) > 0,
    # the same Q as LAPACK's QR followed by scipy's d/|d| correction)
    s = 1 / K.sqrt(2.0)
    re = np.asarray(rs.normal(size=(dim, dim)))
    im = np.asarray(rs.normal(size=(dim, dim)))
    qr = np.empty((dim, dim), dtype=float)
    qi = np.empty((dim, dim), dtype=float)
    for j in range(dim):
        vr = s * re[:, j]
        vi = s * im[:, j]
        for k in range(j):
            # coefficient <q_k, v> = sum conj(q_k) v
            cr = 0.0
            ci = 0.0
            for i in range(dim):
                cr += qr[i, k] * vr[i] + qi[i, k] * vi[i]
                ci += qr[i, k] * vi[i] - qi[i, k] * vr[i]
            for i in range(dim):
                vr[i] -= cr * qr[i, k] - ci * qi[i, k]
                vi[i] -= cr * qi[i, k] + ci * qr[i, k]
        nrm = 0.0
        for i in range(dim):
            nrm += vr[i] * vr[i] + vi[i] * vi[i]
        nrm = K.sqrt(nrm)
        for i in range(dim):
            qr[i, j] = vr[i] / nrm
            qi[i, j] = vi[i] / nrm
    out = np.empty((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            out[i, j] = complex(qr[i, j], qi[i, j])
    return out

class unitary_group_frozen:
    dim: int
    def __init__(self, dim: int):
        if dim < 0:
            raise ValueError("Dimension of rotation must be specified,and must be a scalar nonnegative integer.")
        self.dim = dim
    def rvs(self, size: int = 1, random_state = None):
        rs = check_random_state(random_state)
        if size > 1:
            out = np.empty((size, self.dim, self.dim), dtype=complex)
            for k in range(size):
                out[k] = _unitary_rvs(self.dim, rs)
            return out
        return _unitary_rvs(self.dim, rs)

class unitary_group_gen:
    def __call__(self, dim: int):
        return unitary_group_frozen(dim)
    def rvs(self, dim: int, size: int = 1, random_state = None):
        return unitary_group_frozen(dim).rvs(size, random_state)

unitary_group = unitary_group_gen()
