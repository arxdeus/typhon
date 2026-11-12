# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# Minimal FastICA core as used by mne's ICA (whiten=False, parallel
# algorithm, logcosh, alpha=1). Kept local so mne.preprocessing does not
# have to compile the whole sklearn.decomposition package.
# w_init = RandomState(seed).normal(size=(k, k)), exactly as sklearn.

import numpy as np
import scipy.linalg as sl

def _sym_decorrelation(W):
    s, u = sl.eigh(W @ W.T)
    tiny = 2.2250738585072014e-308
    for i in range(s.shape[0]):
        if s[i] < tiny:
            s[i] = tiny
    return ((u * (1.0 / np.sqrt(s))) @ u.T) @ W

def fastica_unmixing(X: np.ndarray[float, 2], random_state: int, max_iter: int = 200,
                     tol: float = 1e-4):
    """Return (components_, n_iter_) of FastICA(whiten=False).fit(X)."""
    XT = np.ascontiguousarray(X.T)
    k = XT.shape[0]
    p_ = float(XT.shape[1])
    w_init = np.random.RandomState(random_state).normal(size=(k, k))
    W = _sym_decorrelation(w_init)
    it = 0
    while it < max_iter:
        gwtx = np.tanh((W @ XT) * 1.0)
        g_wtx = np.empty(k)
        for i in range(k):
            g_wtx[i] = np.mean(1.0 * (1 - gwtx[i] ** 2))
        W1 = _sym_decorrelation((gwtx @ XT.T) / p_ - g_wtx[:, None] * W)
        lim = float(np.max(np.abs(np.abs(np.sum(W1 * W, axis=1)) - 1.0)))
        W = W1
        it += 1
        if lim < tol:
            break
    return W, it
