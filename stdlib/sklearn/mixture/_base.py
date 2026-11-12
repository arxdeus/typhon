# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# Shared Gaussian mixture machinery. Covariance-like arrays are stored in a
# canonical 3-d layout so one code path serves every covariance_type:
#   full: (k, d, d)   tied: (1, d, d)   diag: (1, k, d)   spherical: (1, 1, k)

import numpy as np
import scipy.linalg as sl
from ._kmeans_init import kmeans_labels, kmeans_plusplus

_EPS = 2.220446049250313e-16

def _err_msg() -> str:
    return ("Fitting the mixture model failed because some components have ill-defined "
            "empirical covariance (for instance caused by singleton or collapsed samples). "
            "Try to decrease the number of components, increase reg_covar, or scale the input data.")

def _init_resp(X, k: int, init_params: str, rs):
    n = X.shape[0]
    resp = np.zeros((n, k))
    if init_params == "kmeans":
        lab = kmeans_labels(X, k, rs)
        for i in range(n):
            resp[i, lab[i]] = 1.0
    elif init_params == "random":
        resp = rs.uniform(size=(n, k))
        resp /= np.sum(resp, axis=1)[:, None]
    elif init_params == "random_from_data":
        idx = rs.choice(n, size=k, replace=False)
        for c in range(k):
            resp[idx[c], c] = 1.0
    elif init_params == "k-means++":
        _, idx = kmeans_plusplus(X, k, np.sum(X * X, axis=1), rs)
        for c in range(k):
            resp[idx[c], c] = 1.0
    else:
        raise ValueError(f"Unimplemented initialization method '{init_params}'")
    return resp

def _estimate_gaussian_parameters(X, resp, reg_covar: float, ct: str):
    n, d = X.shape
    k = resp.shape[1]
    nk = np.sum(resp, axis=0) + 10 * _EPS
    means = (resp.T @ X) / nk[:, None]
    cov = np.empty((0, 0, 0))
    if ct == "full":
        cov = np.empty((k, d, d))
        for c in range(k):
            diff = X - means[c, :]
            C = (resp[:, c] * diff.T) @ diff / nk[c]
            for a in range(d):
                C[a, a] += reg_covar
            cov[c, :, :] = C
    elif ct == "tied":
        C = X.T @ X - (nk * means.T) @ means
        C /= float(np.sum(nk))
        for a in range(d):
            C[a, a] += reg_covar
        cov = C.reshape((1, d, d))
    else:
        D = (resp.T @ (X * X)) / nk[:, None] - means ** 2 + reg_covar
        if ct == "diag":
            cov = D.reshape((1, k, d))
        elif ct == "spherical":
            cov = np.mean(D, axis=1).reshape((1, 1, k))
        else:
            raise ValueError(f"invalid covariance_type {ct}")
    return nk, means, cov

def _chol_lower(C):
    try:
        return sl.cholesky(C, lower=True)
    except:
        raise ValueError(_err_msg())

def _compute_precision_cholesky(cov, ct: str):
    if ct == "full" or ct == "tied":
        m, d, _ = cov.shape
        out = np.empty((m, d, d))
        for c in range(m):
            L = _chol_lower(cov[c])
            out[c, :, :] = sl.solve_triangular(L, np.eye(d), lower=True).T
        return out
    for v in cov.ravel():
        if v <= 0.0:
            raise ValueError(_err_msg())
    return 1.0 / np.sqrt(cov)

def _flipudlr(A):
    return np.ascontiguousarray(A[::-1, ::-1])

def _precision_cholesky_from_precisions(P, ct: str):
    if ct == "full" or ct == "tied":
        m = P.shape[0]
        out = np.empty(P.shape)
        for c in range(m):
            out[c, :, :] = _flipudlr(_chol_lower(_flipudlr(P[c])))
        return out
    return np.sqrt(P)

def _to_canonical(A, ct: str, k: int, d: int):
    """user-facing covariance/precision array -> canonical 3-d layout"""
    B = np.asarray(A, dtype=float)
    if ct == "full":
        return np.ascontiguousarray(B.reshape((k, d, d)))
    elif ct == "tied":
        return np.ascontiguousarray(B.reshape((1, d, d)))
    elif ct == "diag":
        return np.ascontiguousarray(B.reshape((1, k, d)))
    return np.ascontiguousarray(B.reshape((1, 1, k)))

def _log_det_cholesky(pc, ct: str, k: int, d: int):
    """per-component log det of the precision cholesky (tied: repeated)"""
    out = np.zeros(k)
    if ct == "full":
        for c in range(k):
            s = 0.0
            for a in range(d):
                s += np.log(pc[c, a, a])
            out[c] = s
    elif ct == "tied":
        s = 0.0
        for a in range(d):
            s += np.log(pc[0, a, a])
        out.fill(s)
    elif ct == "diag":
        out = np.sum(np.log(pc[0]), axis=1)
    else:
        out = d * np.log(pc[0, 0])
    return out

def _estimate_log_gaussian_prob(X, means, pc, ct: str):
    n, d = X.shape
    k = means.shape[0]
    log_det = _log_det_cholesky(pc, ct, k, d)
    lp = np.empty((n, k))
    if ct == "full" or ct == "tied":
        for c in range(k):
            P = pc[c] if ct == "full" else pc[0]
            y = X @ P - means[c, :] @ P
            lp[:, c] = np.sum(y * y, axis=1)
    elif ct == "diag":
        prec = pc[0] ** 2
        lp = np.sum(means ** 2 * prec, axis=1) - 2.0 * (X @ (means * prec).T) + (X ** 2) @ prec.T
    else:
        prec = pc[0, 0] ** 2
        xs = np.sum(X * X, axis=1)
        lp = np.sum(means ** 2, axis=1) * prec - 2 * ((X @ means.T) * prec) + xs[:, None] * prec[None, :]
    return -0.5 * (d * np.log(2 * np.pi) + lp) + log_det

def _precisions_from_chol(pc, ct: str):
    if ct == "full" or ct == "tied":
        out = np.empty(pc.shape)
        for c in range(pc.shape[0]):
            out[c, :, :] = pc[c] @ pc[c].T
        return out
    return pc ** 2

def _from_canonical(A, CT: Literal[str]):
    if CT == "full":
        return A
    elif CT == "tied":
        return A[0]
    elif CT == "diag":
        return A[0]
    else:
        return A[0, 0]

def _logsumexp_rows(a):
    m = np.max(a, axis=1)
    m = np.where(np.isfinite(m), m, 0.0)
    s = np.sum(np.exp(a - m[:, None]), axis=1)
    return np.log(s) + m

def _fit_loop(model, X, n_components: int, n_init: int, max_iter: int, tol: float, rs, do_init: bool):
    """BaseMixture.fit_predict EM driver; returns labels."""
    if X.shape[0] < n_components:
        raise ValueError(f"Expected n_samples >= n_components but got n_components = {n_components}, n_samples = {X.shape[0]}")
    ni = n_init if do_init else 1
    max_lb = -np.inf
    best_lbs: List[float] = []
    best_params = model._get_parameters()
    best_n_iter = 0
    model.converged_ = False
    for _ in range(ni):
        if do_init:
            model._initialize_parameters(X, rs)
        lb = -np.inf if do_init else model.lower_bound_
        cur: List[float] = []
        if max_iter == 0:
            best_params = model._get_parameters()
            best_n_iter = 0
        else:
            converged = False
            n_iter = 0
            for it in range(1, max_iter + 1):
                n_iter = it
                prev = lb
                lpn, log_resp = model._e_step(X)
                model._m_step(X, log_resp)
                lb = model._compute_lower_bound(log_resp, lpn)
                cur.append(lb)
                if abs(lb - prev) < tol:
                    converged = True
                    break
            if lb > max_lb or max_lb == -np.inf:
                max_lb = lb
                best_params = model._get_parameters()
                best_n_iter = n_iter
                best_lbs = cur
                model.converged_ = converged
    model._set_parameters(best_params)
    model.n_iter_ = best_n_iter
    model.lower_bound_ = max_lb
    model.lower_bounds_ = best_lbs
    _, log_resp = model._e_step(X)
    return np.argmax(log_resp, axis=1)
