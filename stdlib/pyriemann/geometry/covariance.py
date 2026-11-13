# Ported from pyriemann 0.12 (BSD-3-Clause)
# Shrinkage estimators follow scikit-learn's ledoit_wolf / oas (BSD-3-Clause).
#
# Real-valued input only.

import numpy as np
from .base import ctranspose, invsqrtm
from sklearn.covariance import fast_mcd

def _per_matrix(X, f):
    # Apply f: (n_channels, n_times) -> (n_channels, n_channels) over all
    # leading batch dimensions.
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        return f(X)
    else:
        c = X.shape[X.ndim - 2]
        t = X.shape[X.ndim - 1]
        Xf = X.reshape(-1, c, t)
        nb = Xf.shape[0]
        out = np.empty((nb, c, c))
        for b in range(nb):
            out[b] = f(Xf[b])
        return out.reshape(X.shape[:-2] + (c, c))

def _empirical_covariance(Xs, assume_centered: bool):
    # Xs: (n_samples, n_features), as in sklearn.
    if not assume_centered:
        Xs = Xs - np.mean(Xs, axis=0)
    return (Xs.T @ Xs) / Xs.shape[0]

def _ledoit_wolf_shrinkage(Xs):
    # Xs centered, (n_samples, n_features). Single-block version of sklearn's
    # blocked computation (identical for n_features <= block_size=1000).
    n_samples, n_features = Xs.shape[0], Xs.shape[1]
    X2 = Xs * Xs
    emp_cov_trace = np.sum(X2, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features
    beta_ = float(np.sum(X2.T @ X2))
    XtX = Xs.T @ Xs
    delta_ = float(np.sum(XtX * XtX)) / n_samples ** 2
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    delta = delta_ - 2.0 * mu * np.sum(emp_cov_trace) + n_features * mu ** 2
    delta /= n_features
    beta = min(beta, delta)
    return 0.0 if beta == 0 else beta / delta

def ledoit_wolf(X, assume_centered: bool = False):
    # sklearn.covariance.ledoit_wolf: X (n_samples, n_features) -> (cov, shrinkage)
    Xs = np.asarray(X, dtype=float)
    if not assume_centered:
        Xs = Xs - np.mean(Xs, axis=0)
    n_features = Xs.shape[1]
    if n_features == 1:
        return np.full((1, 1), float(np.mean(Xs * Xs))), 0.0
    shrinkage = _ledoit_wolf_shrinkage(Xs)
    emp_cov = (Xs.T @ Xs) / Xs.shape[0]
    mu = np.trace(emp_cov) / n_features
    shrunk = (1.0 - shrinkage) * emp_cov
    for i in range(n_features):
        shrunk[i, i] += shrinkage * mu
    return shrunk, shrinkage

def oas(X, assume_centered: bool = False):
    # sklearn.covariance.oas: X (n_samples, n_features) -> (cov, shrinkage)
    Xs = np.asarray(X, dtype=float)
    if not assume_centered:
        Xs = Xs - np.mean(Xs, axis=0)
    n_samples, n_features = Xs.shape[0], Xs.shape[1]
    if n_features == 1:
        return np.full((1, 1), float(np.mean(Xs * Xs))), 0.0
    emp_cov = (Xs.T @ Xs) / n_samples
    alpha = np.mean(emp_cov * emp_cov)
    mu = np.trace(emp_cov) / n_features
    mu_squared = mu ** 2
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
    shrunk = (1.0 - shrinkage) * emp_cov
    for i in range(n_features):
        shrunk[i, i] += shrinkage * mu
    return shrunk, shrinkage

def _cov(x):
    return np.atleast_2d(np.cov(x))

def _corr(x):
    return np.atleast_2d(np.corrcoef(x))

def _lwf(x):
    return ledoit_wolf(x.T)[0]

def _oas(x):
    return oas(x.T)[0]

def _scm(x):
    xc = x - np.mean(x, axis=-1, keepdims=True)
    return (xc @ xc.T) / x.shape[1]

def _sch(x):
    n_times = x.shape[1]
    n = x.shape[0]
    X_c = x - np.mean(x, axis=-1, keepdims=True)
    C_scm = X_c @ X_c.T / n_times
    std = np.std(x, axis=-1)
    R = n_times / ((n_times - 1.0) * (std[:, None] * std[None, :])) * C_scm
    X_c2 = X_c * X_c
    var_R = X_c2 @ X_c2.T - n_times * C_scm * C_scm
    var = np.var(x, axis=-1)
    var_R *= n_times / ((n_times - 1) ** 3 * (var[:, None] * var[None, :]))
    for i in range(n):
        R[i, i] = 0.0
        var_R[i, i] = 0.0
    R2_sum = float(np.sum(R * R))
    g = 0.0 if R2_sum == 0 else float(np.sum(var_R)) / R2_sum
    gamma = min(max(g, 0.0), 1.0)
    out = (1.0 - gamma) * (n_times / (n_times - 1.0)) * C_scm
    for i in range(n):
        out[i, i] += gamma * (n_times / (n_times - 1.0)) * C_scm[i, i]
    return out

def covariance_scm(X, assume_centered: bool = False, weights = None):
    X = np.asarray(X, dtype=float)
    n_times = X.shape[X.ndim - 1]
    if weights is None:
        w = np.ones(n_times) / n_times
    else:
        w0 = np.asarray(weights, dtype=float)
        w = w0 / np.sum(w0)
    if not assume_centered:
        X = X - np.sum(X * w, axis=-1, keepdims=True)
    return (X * w) @ ctranspose(X)

def covariance_sch(X):
    return _per_matrix(X, _sch)

class _McdFn:
    random_state: Optional[int]
    support_fraction: Optional[float]
    def __init__(self, random_state: Optional[int], support_fraction: Optional[float]):
        self.random_state = random_state
        self.support_fraction = support_fraction
    def __call__(self, x):
        r = fast_mcd(np.ascontiguousarray(x.T), self.support_fraction, self.random_state)
        return r[1]

class _MestFn:
    m: str
    def __init__(self, m: str):
        self.m = m
    def __call__(self, x):
        return covariance_mest(x, self.m)

def covariances(X, estimator: str = "cov", random_state: Optional[int] = None,
                support_fraction: Optional[float] = None):
    if estimator == "cov":
        return _per_matrix(X, _cov)
    elif estimator == "scm":
        return covariance_scm(X)
    elif estimator == "lwf":
        return _per_matrix(X, _lwf)
    elif estimator == "oas":
        return _per_matrix(X, _oas)
    elif estimator == "sch":
        return _per_matrix(X, _sch)
    elif estimator == "corr":
        return _per_matrix(X, _corr)
    elif estimator == "mcd":
        return _per_matrix(X, _McdFn(random_state, support_fraction))
    elif estimator == "hub" or estimator == "stu" or estimator == "tyl":
        return _per_matrix(X, _MestFn(estimator))
    else:
        raise ValueError(
            f"Unknown function name '{estimator}'. Must be one of corr cov hub lwf "
            "mcd oas sch scm stu tyl")

def normalize(X, norm: str):
    X = np.asarray(X)
    if X.shape[X.ndim - 1] != X.shape[X.ndim - 2]:
        raise ValueError("Matrices must be square")
    n = X.shape[X.ndim - 1]
    if norm == "corr":
        stddev = np.sqrt(np.abs(np.diagonal(X, axis1=-2, axis2=-1)))
        Xn = X / (stddev[..., None, :] * stddev[..., None])
        return np.clip(Xn, -1.0, 1.0)
    elif norm == "trace":
        denom = np.trace(X, axis1=-2, axis2=-1)
        return X / np.asarray(denom)[..., None, None]
    elif norm == "determinant":
        denom = np.abs(np.linalg.det(X)) ** (1.0 / n)
        return X / np.asarray(denom)[..., None, None]
    else:
        raise ValueError(f"{norm} is not a supported normalization")

def get_nondiag_weight(X):
    X = np.asarray(X)
    if X.shape[X.ndim - 1] != X.shape[X.ndim - 2]:
        raise ValueError("Matrices must be square")
    X2 = X * X
    denom = np.trace(X2, axis1=-2, axis2=-1)
    num = np.sum(X2, axis=(-2, -1)) - denom
    return 1.0 / (X.shape[X.ndim - 1] - 1) * (num / denom)

def _mest_weight(x, m_estimator: str, n_channels: int, q: float, nu: float):
    if m_estimator == "hub":
        from scipy.special._kernels import gammainc, gammaincinv
        c2 = gammaincinv(n_channels / 2.0, q)
        b = gammainc((n_channels + 1) / 2.0, c2) + c2 * (1.0 - q) / n_channels
        return np.minimum(np.ones_like(x), c2 / x) / b
    elif m_estimator == "stu":
        return (2.0 * n_channels + nu) / (nu + 2.0 * x)
    return n_channels / x

def covariance_mest(X, m_estimator: str, init = None, tol: float = 10e-3,
                    n_iter_max: int = 50, assume_centered: bool = False,
                    q: float = 0.9, nu: float = 5.0, norm: str = "trace"):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        n_c0 = X.shape[X.ndim - 2]
        X3 = X.reshape(-1, X.shape[X.ndim - 2], X.shape[X.ndim - 1])
        out = np.empty((X3.shape[0], n_c0, n_c0))
        for b in range(X3.shape[0]):
            out[b] = covariance_mest(X3[b], m_estimator, init, tol, n_iter_max,
                                     assume_centered, q, nu, norm)
        return out.reshape(X.shape[:-2] + (n_c0, n_c0))
    n_channels = X.shape[0]
    n_times = X.shape[1]
    if m_estimator == "hub":
        if not (0 < q and q <= 1):
            raise ValueError(f"Value q must be included in (0, 1] (Got {q})")
    elif m_estimator == "stu":
        if nu <= 0:
            raise ValueError(f"Value nu must be strictly positive (Got {nu})")
    elif m_estimator != "tyl":
        raise ValueError(f"Unsupported m_estimator: {m_estimator}")
    if not assume_centered:
        X = X - np.mean(X, axis=-1, keepdims=True)
    if init is None:
        cov = X @ X.T / n_times
    else:
        cov = np.array(np.asarray(init, dtype=float))
    for _ in range(n_iter_max):
        w = invsqrtm(cov) @ X
        dist2 = np.sum(w * w, axis=0)
        Xw = np.sqrt(_mest_weight(dist2, m_estimator, n_channels, q, nu))[None, :] * X
        cov_new = Xw @ Xw.T / n_times
        norm_delta = np.linalg.norm(cov_new - cov)
        norm_cov = np.linalg.norm(cov)
        cov = cov_new
        if norm_delta / norm_cov <= tol:
            break
    if m_estimator == "tyl":
        cov = normalize(cov, norm)
        if norm == "trace":
            cov = cov * n_channels
    return cov

def covariances_EP(X, P, estimator: str = "cov"):
    X = np.asarray(X, dtype=float)
    P = np.asarray(P, dtype=float)
    nt = X.shape[X.ndim - 1]
    if P.shape[1] != nt:
        raise ValueError(
            f"X and P do not have the same n_times: {nt} and {P.shape[1]}")
    X3 = X.reshape(-1, X.shape[X.ndim - 2], nt)
    npr = P.shape[0]
    PX = np.empty((X3.shape[0], npr + X3.shape[1], nt))
    for b in range(X3.shape[0]):
        PX[b, :npr, :] = P
        PX[b, npr:, :] = X3[b]
    out = covariances(PX, estimator)
    if X.ndim == 2:
        return out[0]
    return out.reshape(X.shape[:-2] + (out.shape[1], out.shape[2]))

def covariances_X(X, estimator: str = "cov", alpha: float = 0.2):
    X = np.asarray(X, dtype=float)
    X3 = X.reshape(-1, X.shape[X.ndim - 2], X.shape[X.ndim - 1])
    nc = X3.shape[1]
    nt = X3.shape[2]
    Hc = np.eye(nc) - np.ones((nc, nc)) / nc
    Ht = np.eye(nt) - np.ones((nt, nt)) / nt
    Z = np.zeros((X3.shape[0], nc + nt, nc + nt))
    for b in range(X3.shape[0]):
        Xc = Hc @ X3[b] @ Ht
        top = np.concatenate((Xc, alpha * np.eye(nc)), axis=1)
        bot = np.concatenate((alpha * np.eye(nt), Xc.T), axis=1)
        Z[b] = np.concatenate((top, bot), axis=0)
    out = covariances(Z, estimator) / (2.0 * alpha)
    return out

def block_covariances(X, blocks: List[int], estimator: str = "cov"):
    X = np.asarray(X, dtype=float)
    nc = X.shape[X.ndim - 2]
    tot = 0
    for j in blocks:
        tot += j
    if tot != nc:
        raise ValueError("Sum of individual block sizes must match number of channels of X.")
    X3 = X.reshape(-1, nc, X.shape[X.ndim - 1])
    out = np.zeros((X3.shape[0], nc, nc))
    i0 = 0
    for j in blocks:
        bc = covariances(X3[:, i0:i0 + j, :], estimator)
        if bc.ndim < 3:
            bc = bc.reshape(X3.shape[0], 1, 1)
        out[:, i0:i0 + j, i0:i0 + j] = bc
        i0 += j
    return out.reshape(X.shape[:-2] + (nc, nc))

def cross_spectrum(X, window: int = 128, overlap: float = 0.75, fmin = None,
                   fmax = None, fs = None):
    # Returns (S_re, S_im, freqs): real and imaginary parts of the
    # cross-spectral matrices, shape (n_matrices, n_ch, n_ch, n_freqs).
    X = np.asarray(X, dtype=float)
    if window < 1:
        raise ValueError("Value window must be a positive integer")
    if not (0 < overlap and overlap < 1):
        raise ValueError(f"Value overlap must be included in (0, 1) (Got {overlap})")
    X3 = X.reshape(-1, X.shape[X.ndim - 2], X.shape[X.ndim - 1])
    nm = X3.shape[0]
    nch = X3.shape[1]
    n_times = X3.shape[2]
    n_freqs = window // 2 + 1
    step = int((1.0 - overlap) * window)
    n_windows = int((n_times - window) / step + 1)
    win = np.hanning(window)
    fdata_re = np.empty((nch, n_windows, n_freqs))
    fdata_im = np.empty((nch, n_windows, n_freqs))
    wnorm = float(np.sum(win * win))
    # frequency selection
    f_all = np.arange(n_freqs) * (float(fs) / window if fs is not None else 1.0)
    fix = np.ones(n_freqs, dtype=bool)
    freqs = f_all
    if fs is not None:
        f0 = 0.0 if fmin is None else float(fmin)
        f1 = float(fs) / 2.0 if fmax is None else float(fmax)
        if f1 <= f0:
            raise ValueError("Parameter fmax must be superior to fmin")
        if 2.0 * f1 > float(fs):
            raise ValueError("Parameter fmax must be inferior to fs/2")
        fix = (f_all >= f0) & (f_all <= f1)
        freqs = f_all[fix]
    nsel = int(np.sum(fix))
    S_re = np.zeros((nm, nch, nch, nsel))
    S_im = np.zeros((nm, nch, nch, nsel))
    for m in range(nm):
        for c in range(nch):
            for w in range(n_windows):
                seg = X3[m, c, w * step:w * step + window] * win
                F = np.fft.rfft(seg, n=window)
                fdata_re[c, w, :] = np.real(F)
                fdata_im[c, w, :] = np.imag(F)
        fr = fdata_re[:, :, fix]
        fi = fdata_im[:, :, fix]
        for c in range(nch):
            for d in range(nch):
                # conj(F_c) * F_d summed over windows
                S_re[m, c, d, :] = np.sum(fr[c] * fr[d] + fi[c] * fi[d], axis=0)
                S_im[m, c, d, :] = np.sum(fr[c] * fi[d] - fi[c] * fr[d], axis=0)
    S_re = S_re / (n_windows * wnorm)
    S_im = S_im / (n_windows * wnorm)
    # Parseval correction (applied on the selected bins, as pyriemann does)
    for k in range(nsel):
        dbl = k >= 1 and not (window % 2 == 0 and k == nsel - 1)
        if dbl:
            S_re[:, :, :, k] *= 2.0
            S_im[:, :, :, k] *= 2.0
    shp = X.shape[:-2] + (nch, nch, nsel)
    return S_re.reshape(shp), S_im.reshape(shp), freqs

def coherence(X, window: int = 128, overlap: float = 0.75, fmin = None,
              fmax = None, fs = None, coh: str = "ordinary"):
    Sre, Sim, freqs = cross_spectrum(X, window, overlap, fmin, fmax, fs)
    nf = Sre.shape[Sre.ndim - 1]
    nch = Sre.shape[Sre.ndim - 2]
    S3r = Sre.reshape(-1, nch, nch, nf)
    S3i = Sim.reshape(-1, nch, nch, nf)
    C = np.zeros(S3r.shape)
    nm = S3r.shape[0]
    f_ok = np.ones(nf, dtype=bool)
    if coh == "lagged":
        if fs is None:
            f_ok = np.zeros(nf, dtype=bool)
            for k in range(1, nf - 1):
                f_ok[k] = True
        else:
            f_ok = (freqs > 0) & (freqs < float(fs) / 2.0)
    for m in range(nm):
        psd = np.empty((nch, nf))
        for c in range(nch):
            psd[c] = np.sqrt(S3r[m, c, c] ** 2 + S3i[m, c, c] ** 2)
        for c in range(nch):
            for d in range(nch):
                pp = psd[c] * psd[d]
                a2 = S3r[m, c, d] ** 2 + S3i[m, c, d] ** 2
                if coh == "ordinary":
                    v = a2 / pp
                elif coh == "instantaneous":
                    v = S3r[m, c, d] ** 2 / pp
                elif coh == "lagged":
                    sr = S3r[m, c, d] if c != d else np.zeros(nf)
                    den = pp - sr ** 2
                    den = np.where(np.abs(den) < 1e-10, 1e-10, den)
                    v = S3i[m, c, d] ** 2 / den
                elif coh == "imaginary":
                    v = S3i[m, c, d] ** 2 / pp
                else:
                    raise ValueError(f"{coh} is not a supported coherence")
                C[m, c, d] = np.where(f_ok, v, 0.0)
    return C.reshape(Sre.shape), freqs
