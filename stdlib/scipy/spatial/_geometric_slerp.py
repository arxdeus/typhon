# Ported from scipy 1.18.1 (BSD-3-Clause)

import numpy as np

def _geometric_slerp(start, end, t):
    n = start.shape[0]
    basis = np.empty((2, n), dtype=float)
    basis[0] = start
    basis[1] = end
    Q, R = np.linalg.qr(basis.T)
    Qt = Q.T.copy()
    Rt = R.T.copy()
    for i in range(2):
        if R[i, i] < 0:
            Qt[i] = -Qt[i]
            Rt[i] = -Rt[i]
    c = float(np.dot(start, end))
    s = Rt[0, 0] * Rt[1, 1] - Rt[0, 1] * Rt[1, 0]
    omega = np.arctan2(s, c)
    m = t.shape[0]
    out = np.empty((m, n), dtype=float)
    for i in range(m):
        si = np.sin(t[i] * omega)
        ci = np.cos(t[i] * omega)
        for k in range(n):
            out[i, k] = Qt[0, k] * ci + Qt[1, k] * si
    return out

def _slerp2d(start, end, t):
    st = np.asarray(start, dtype=float)
    en = np.asarray(end, dtype=float)
    if st.ndim != 1 or en.ndim != 1:
        raise ValueError("Start and end coordinates must be one-dimensional")
    if st.size != en.size:
        raise ValueError("The dimensions of start and end must match (have same size)")
    if st.size < 2:
        raise ValueError("The start and end coordinates must both be in at least two-dimensional space")
    ta = np.atleast_1d(np.asarray(t, dtype=float))
    if ta.ndim > 1:
        raise ValueError("The interpolation parameter value must be one dimensional.")
    tt = ta.ravel()
    if np.array_equal(st, en):
        out = np.empty((tt.shape[0], st.shape[0]), dtype=float)
        for i in range(tt.shape[0]):
            out[i] = st
        return out
    for coord in (st, en):
        if abs(float(np.linalg.norm(coord)) - 1.0) > 1e-9:
            raise ValueError("start and end are not on a unit n-sphere")
    if tt.shape[0] == 0:
        return np.empty((0, st.shape[0]), dtype=float)
    return _geometric_slerp(st, en, tt)

def geometric_slerp(start, end, t, tol: float = 1e-7):
    if isinstance(t, float) or isinstance(t, int):
        return _slerp2d(start, end, t).ravel()
    else:
        return _slerp2d(start, end, t)
