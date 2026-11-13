# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Clustering metrics (sklearn.metrics.cluster): supervised label-agreement
# scores and unsupervised (silhouette, Calinski-Harabasz, Davies-Bouldin).

import numpy as np
from math import lgamma, log as _log
from scipy.special import gammaln

_EPS = 2.220446049250313e-16

def _encode(labels):
    a = np.asarray(labels)
    if static.len(a.shape) != 1:
        compile_error("labels must be 1D")
    u, inv = np.unique(a, return_inverse=True)
    return u.shape[0], np.asarray(inv, dtype=int).reshape(-1)

def contingency_matrix(labels_true, labels_pred, *, eps = None,
                       sparse: bool = False, dtype: type = int):
    if sparse:
        raise NotImplementedError("contingency_matrix: sparse=True is not supported")
    nc, ci = _encode(labels_true)
    nk, ki = _encode(labels_pred)
    if ci.shape[0] != ki.shape[0]:
        raise ValueError("Found input variables with inconsistent numbers of samples")
    C = np.zeros((nc, nk), dtype=int)
    for i in range(ci.shape[0]):
        C[ci[i], ki[i]] += 1
    if isinstance(eps, NoneType):
        return C.astype(dtype)
    else:
        return C.astype(float) + float(eps)

def pair_confusion_matrix(labels_true, labels_pred):
    C = contingency_matrix(labels_true, labels_pred)
    n = 0
    nc, nk = C.shape
    ss = 0
    a = 0
    b = 0
    rows = np.sum(C, axis=1)
    cols = np.sum(C, axis=0)
    for i in range(nc):
        for j in range(nk):
            v = int(C[i, j])
            n += v
            ss += v * v
            a += v * int(cols[j])
            b += v * int(rows[i])
    M = np.empty((2, 2), dtype=int)
    M[1, 1] = ss - n
    M[0, 1] = a - ss
    M[1, 0] = b - ss
    M[0, 0] = n * n - M[0, 1] - M[1, 0] - ss
    return M

def rand_score(labels_true, labels_pred) -> float:
    C = pair_confusion_matrix(labels_true, labels_pred)
    num = int(C[0, 0] + C[1, 1])
    den = int(C[0, 0] + C[0, 1] + C[1, 0] + C[1, 1])
    if num == den or den == 0:
        return 1.0
    return float(num) / float(den)

def adjusted_rand_score(labels_true, labels_pred) -> float:
    C = pair_confusion_matrix(labels_true, labels_pred)
    tn = int(C[0, 0]); fp = int(C[0, 1]); fn = int(C[1, 0]); tp = int(C[1, 1])
    if fn == 0 and fp == 0:
        return 1.0
    return 2.0 * float(tp * tn - fn * fp) / float((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))

def _entropy_codes(k: int, codes: np.ndarray[int, 1]) -> float:
    n = codes.shape[0]
    if n == 0:
        return 1.0
    pi = np.zeros(k)
    for i in range(n):
        pi[codes[i]] += 1.0
    if k == 1:
        return 0.0
    s = np.sum(pi)
    return float(-np.sum((pi / s) * (np.log(pi) - _log(s))))

def _mi_from_contingency(C) -> float:
    Cf = np.asarray(C, dtype=float)
    nc, nk = Cf.shape
    if nc == 1 or nk == 1:
        return 0.0
    pi = np.sum(Cf, axis=1)
    pj = np.sum(Cf, axis=0)
    tot = float(np.sum(Cf))
    nnz = 0
    for i in range(nc):
        for j in range(nk):
            if Cf[i, j] != 0.0:
                nnz += 1
    mi = np.empty(nnz)
    t = 0
    lpi = _log(float(np.sum(pi)))
    lpj = _log(float(np.sum(pj)))
    lt = _log(tot)
    for i in range(nc):
        for j in range(nk):
            v = Cf[i, j]
            if v != 0.0:
                cnm = v / tot
                outer = float(int(pi[i]) * int(pj[j]))
                lo = -_log(outer) + lpi + lpj
                r = cnm * (_log(v) - lt) + cnm * lo
                mi[t] = 0.0 if abs(r) < _EPS else r
                t += 1
    s = float(np.sum(mi))
    return s if s > 0.0 else 0.0

def mutual_info_score(labels_true, labels_pred, *, contingency = None) -> float:
    if contingency is None:
        return _mi_from_contingency(contingency_matrix(labels_true, labels_pred))
    else:
        return _mi_from_contingency(contingency)

def _generalized_average(U: float, V: float, method: str) -> float:
    if method == "min":
        return min(U, V)
    elif method == "geometric":
        return np.sqrt(U * V)
    elif method == "arithmetic":
        return float(np.mean(np.array([U, V])))
    elif method == "max":
        return max(U, V)
    raise ValueError("'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'")

def _expected_mutual_information(C, n_samples: int) -> float:
    nr, ncl = C.shape
    a = np.sum(C, axis=1).astype(int)
    b = np.sum(C, axis=0).astype(int)
    if nr == 1 or ncl == 1:
        return 0.0
    mx = max(int(np.max(a)), int(np.max(b)))
    nijs = np.arange(0, mx + 1).astype(float)
    nijs[0] = 1.0
    N = float(n_samples)
    term1 = nijs / N
    log_a = np.log(a.astype(float))
    log_b = np.log(b.astype(float))
    log_Nnij = np.log(N) + np.log(nijs)
    gln_a = gammaln(a.astype(float) + 1.0)
    gln_b = gammaln(b.astype(float) + 1.0)
    gln_Na = gammaln(N - a.astype(float) + 1.0)
    gln_Nb = gammaln(N - b.astype(float) + 1.0)
    gln_Nnij = gammaln(nijs + 1.0) + float(gammaln(N + 1.0))
    emi = 0.0
    for i in range(nr):
        for j in range(ncl):
            ai = int(a[i]); bj = int(b[j])
            start = max(1, ai - n_samples + bj)
            end = min(ai, bj) + 1
            for nij in range(start, end):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                       - gln_Nnij[nij] - lgamma(float(ai - nij + 1))
                       - lgamma(float(bj - nij + 1))
                       - lgamma(float(n_samples - ai - bj + nij + 1)))
                emi += term1[nij] * term2 * np.exp(gln)
    return emi

def adjusted_mutual_info_score(labels_true, labels_pred, *, average_method: str = "arithmetic") -> float:
    kc, ci = _encode(labels_true)
    kk, ki = _encode(labels_pred)
    n = ci.shape[0]
    if (kc == 1 and kk == 1) or (kc == 0 and kk == 0):
        return 1.0
    elif kc == 1 or kk == 1:
        return 0.0
    C = contingency_matrix(labels_true, labels_pred)
    mi = _mi_from_contingency(C)
    emi = _expected_mutual_information(C, n)
    norm = _generalized_average(_entropy_codes(kc, ci), _entropy_codes(kk, ki), average_method)
    den = norm - emi
    den = min(den, -_EPS) if den < 0 else max(den, _EPS)
    num = mi - emi
    num = min(num, -_EPS) if num < 0 else max(num, _EPS)
    return num / den

def normalized_mutual_info_score(labels_true, labels_pred, *, average_method: str = "arithmetic") -> float:
    kc, ci = _encode(labels_true)
    kk, ki = _encode(labels_pred)
    if (kc == 1 and kk == 1) or (kc == 0 and kk == 0):
        return 1.0
    mi = _mi_from_contingency(contingency_matrix(labels_true, labels_pred))
    if mi == 0:
        return 0.0
    return mi / _generalized_average(_entropy_codes(kc, ci), _entropy_codes(kk, ki), average_method)

def homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta: float = 1.0):
    kc, ci = _encode(labels_true)
    kk, ki = _encode(labels_pred)
    if ci.shape[0] == 0:
        return 1.0, 1.0, 1.0
    hC = _entropy_codes(kc, ci)
    hK = _entropy_codes(kk, ki)
    mi = _mi_from_contingency(contingency_matrix(labels_true, labels_pred))
    h = mi / hC if hC != 0.0 else 1.0
    c = mi / hK if hK != 0.0 else 1.0
    v = 0.0 if h + c == 0.0 else (1 + beta) * h * c / (beta * h + c)
    return h, c, v

def homogeneity_score(labels_true, labels_pred) -> float:
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[0]

def completeness_score(labels_true, labels_pred) -> float:
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[1]

def v_measure_score(labels_true, labels_pred, *, beta: float = 1.0) -> float:
    return homogeneity_completeness_v_measure(labels_true, labels_pred, beta=beta)[2]

def fowlkes_mallows_score(labels_true, labels_pred) -> float:
    C = contingency_matrix(labels_true, labels_pred)
    n = int(np.sum(C))
    tk = int(np.sum(C * C)) - n
    pk = int(np.sum(np.sum(C, axis=0) ** 2)) - n
    qk = int(np.sum(np.sum(C, axis=1) ** 2)) - n
    return float(np.sqrt(tk / pk) * np.sqrt(tk / qk)) if tk != 0 else 0.0

def _check_n_labels(k: int, n: int):
    if not 1 < k < n:
        raise ValueError(f"Number of labels is {k}. Valid values are 2 to n_samples - 1 (inclusive)")

def _pairwise(X: np.ndarray[float, 2], metric: str) -> np.ndarray[float, 2]:
    n, d = X.shape
    if metric == "precomputed":
        return X
    D = np.zeros((n, n))
    if metric == "euclidean" or metric == "l2":
        sq = np.empty(n)
        for i in range(n):
            s = 0.0
            for k in range(d):
                s += X[i, k] * X[i, k]
            sq[i] = s
        G = X @ X.T
        for i in range(n):
            for j in range(n):
                v = -2.0 * G[i, j] + sq[i] + sq[j]
                D[i, j] = np.sqrt(v) if v > 0.0 else 0.0
            D[i, i] = 0.0
        return D
    for i in range(n):
        for j in range(n):
            r = 0.0
            if metric == "manhattan" or metric == "l1" or metric == "cityblock":
                for k in range(d):
                    r += abs(X[i, k] - X[j, k])
            elif metric == "chebyshev":
                for k in range(d):
                    r = max(r, abs(X[i, k] - X[j, k]))
            else:
                raise ValueError(f"Unknown metric {metric}")
            D[i, j] = r
    return D

def silhouette_samples(X, labels, *, metric: str = "euclidean"):
    Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
    k, lab = _encode(labels)
    n = lab.shape[0]
    _check_n_labels(k, n)
    D = _pairwise(Xa, metric)
    freqs = np.zeros(k)
    for i in range(n):
        freqs[lab[i]] += 1.0
    out = np.empty(n)
    for i in range(n):
        cd = np.zeros(k)
        for j in range(n):
            cd[lab[j]] += D[i, j]
        intra = cd[lab[i]]
        cd[lab[i]] = np.inf
        inter = np.inf
        for c in range(k):
            v = cd[c] / freqs[c]
            if v < inter:
                inter = v
        den = freqs[lab[i]] - 1.0
        a = intra / den if den != 0.0 else (np.nan if intra == 0.0 else np.inf)
        s = (inter - a) / max(a, inter)
        out[i] = 0.0 if np.isnan(s) else s
    return out

def silhouette_score(X, labels, *, metric: str = "euclidean", sample_size = None,
                     random_state = None) -> float:
    if sample_size is not None:
        raise NotImplementedError("silhouette_score: sample_size is not supported")
    return float(np.mean(silhouette_samples(X, labels, metric=metric)))

def calinski_harabasz_score(X, labels) -> float:
    Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
    k, lab = _encode(labels)
    n = Xa.shape[0]
    _check_n_labels(k, n)
    extra = 0.0
    intra = 0.0
    mean = np.mean(Xa, axis=0)
    for c in range(k):
        Xk = Xa[lab == c]
        mk = np.mean(Xk, axis=0)
        extra += Xk.shape[0] * float(np.sum((mk - mean) ** 2))
        intra += float(np.sum((Xk - mk) ** 2))
    return 1.0 if intra == 0.0 else extra * (n - k) / (intra * (k - 1.0))

def davies_bouldin_score(X, labels) -> float:
    Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
    k, lab = _encode(labels)
    n, p = Xa.shape
    _check_n_labels(k, n)
    intra = np.zeros(k)
    cent = np.zeros((k, p))
    for c in range(k):
        Xk = np.ascontiguousarray(Xa[lab == c])
        ck = np.mean(Xk, axis=0)
        cent[c] = ck
        s = 0.0
        for i in range(Xk.shape[0]):
            r = 0.0
            for f in range(p):
                r += (Xk[i, f] - ck[f]) ** 2
            s += np.sqrt(r)
        intra[c] = s / Xk.shape[0]
    cd = _pairwise(cent, "euclidean")
    all0 = True
    for c in range(k):
        if abs(intra[c]) > 1e-8:
            all0 = False
    allc0 = True
    for i in range(k):
        for j in range(k):
            if abs(cd[i, j]) > 1e-8:
                allc0 = False
    if all0 or allc0:
        return 0.0
    tot = 0.0
    for i in range(k):
        m = -np.inf
        for j in range(k):
            d = cd[i, j] if cd[i, j] != 0.0 else np.inf
            v = (intra[i] + intra[j]) / d
            if v > m:
                m = v
        tot += m
    return tot / k

# ---------------------------------------------------------- consensus_score

def consensus_score(a, b, *, similarity: Literal[str] = "jaccard") -> float:
    """Similarity of two sets of biclusters ((rows, columns) indicator
    matrices), Jaccard coefficient + linear sum assignment."""
    from scipy.optimize import linear_sum_assignment
    ar = np.asarray(a[0], dtype=float)
    ac = np.asarray(a[1], dtype=float)
    br = np.asarray(b[0], dtype=float)
    bc = np.asarray(b[1], dtype=float)
    if ar.shape[0] != ac.shape[0] or br.shape[0] != bc.shape[0]:
        raise ValueError("Found input variables with inconsistent numbers of samples")
    n_a = ar.shape[0]
    n_b = br.shape[0]
    M = np.empty((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            inter = float(np.sum(ar[i] * br[j])) * float(np.sum(ac[i] * bc[j]))
            asz = float(np.sum(ar[i])) * float(np.sum(ac[i]))
            bsz = float(np.sum(br[j])) * float(np.sum(bc[j]))
            M[i, j] = inter / (asz + bsz - inter)
    ri, ci = linear_sum_assignment(1.0 - M)
    tot = 0.0
    for k in range(ri.shape[0]):
        tot += M[ri[k], ci[k]]
    return tot / float(max(n_a, n_b))
