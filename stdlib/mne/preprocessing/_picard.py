# Ported from python-picard 0.8.2 (BSD-3-Clause)
#
# Picard / Picard-O (preconditioned L-BFGS ICA, Ablin, Cardoso, Gramfort 2018),
# as called by mne.preprocessing.ICA(method='picard') with whiten=False.
# Densities: 0 = tanh(alpha=1), 1 = exp(alpha=0.1), 2 = cube.

import numpy as np
import scipy.linalg as sl

def _score_and_der(Y: np.ndarray[float, 2], fun: int):
    if fun == 0:
        s = np.tanh(Y)
        return s, 1.0 - s ** 2
    elif fun == 1:
        a = 0.1
        Ysq = Y ** 2
        K = np.exp(-a / 2.0 * Ysq)
        return Y * K, (1.0 - a * Ysq) * K
    else:
        return Y ** 3, 3.0 * Y ** 2

def _log_lik_row(y: np.ndarray[float, 1], fun: int) -> float:
    n = y.shape[0]
    s = 0.0
    if fun == 0:
        for i in range(n):
            a = abs(y[i])
            s += a + np.log1p(np.exp(-2.0 * a))
    elif fun == 1:
        a = 0.1
        for i in range(n):
            s += -np.exp(-a * y[i] * y[i] / 2.0) / a
    else:
        for i in range(n):
            s += y[i] ** 4 / 4.0
    return s / n

def _loss(Y: np.ndarray[float, 2], W: np.ndarray[float, 2], fun: int, signs: np.ndarray[float, 1],
          ortho: bool, extended: bool) -> float:
    loss = 0.0
    if not ortho:
        loss = -np.log(abs(np.linalg.det(W)))
    for i in range(Y.shape[0]):
        y = np.ascontiguousarray(Y[i])
        loss += signs[i] * _log_lik_row(y, fun)
        if extended and not ortho:
            loss += 0.5 * float(np.mean(y ** 2))
    return loss

def _line_search(Y: np.ndarray[float, 2], W: np.ndarray[float, 2], fun: int,
                 direction: np.ndarray[float, 2], signs: np.ndarray[float, 1],
                 current_loss0: float, have_loss: bool, ls_tries: int, ortho: bool, extended: bool):
    N = W.shape[0]
    alpha = 1.0
    current_loss = current_loss0
    if not have_loss:
        current_loss = _loss(Y, W, fun, signs, ortho, extended)
    Y_new = Y
    W_new = W
    new_loss = current_loss
    for _ in range(ls_tries):
        transform = np.eye(N)
        if ortho:
            transform = sl.expm(alpha * direction)
        else:
            transform = np.eye(N) + alpha * direction
        Y_new = transform @ Y
        W_new = transform @ W
        new_loss = _loss(Y_new, W_new, fun, signs, ortho, extended)
        if np.isfinite(new_loss) and new_loss < current_loss:
            return True, Y_new, W_new, new_loss, alpha * direction
        alpha /= 2.0
    return False, Y_new, W_new, new_loss, alpha * direction

def _regularize_hessian(h: np.ndarray[float, 2], h_off: np.ndarray[float, 1], lambda_min: float):
    N = h.shape[0]
    h0 = h.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                d = h0[i, j] - h0[j, i]
                discr = np.sqrt(d * d + 4.0 * h_off[i] * h_off[j])
                ev = 0.5 * (h0[i, j] + h0[j, i] - discr)
                if ev < lambda_min:
                    h[i, j] += lambda_min - ev
    return h

def _l_bfgs_direction(G: np.ndarray[float, 2], h: np.ndarray[float, 2], h_off: np.ndarray[float, 1],
                      s_list: List[np.ndarray[float, 2]], y_list: List[np.ndarray[float, 2]],
                      r_list: List[float], ortho: bool):
    q = G.copy()
    a_list = List[float]()
    m = len(s_list)
    for t in range(m - 1, -1, -1):
        alpha = r_list[t] * float(np.sum(s_list[t] * q))
        a_list.append(alpha)
        q -= alpha * y_list[t]
    N = G.shape[0]
    z = np.empty((N, N))
    if ortho:
        z = q / h
        z = (z - z.T) / 2.0
    else:
        for i in range(N):
            for j in range(N):
                det = h[i, j] * h[j, i] - h_off[i] * h_off[j]
                z[i, j] = (h[j, i] * q[i, j] - h_off[i] * q[j, i]) / det
    for t in range(m):
        alpha = a_list[m - 1 - t]
        beta = r_list[t] * float(np.sum(y_list[t] * z))
        z += (alpha - beta) * s_list[t]
    return -z

def core_picard(X: np.ndarray[float, 2], fun: int, ortho: bool, extended: bool, m: int,
                max_iter: int, tol: float, lambda_min: float, ls_tries: int,
                covariance: np.ndarray[float, 2], have_cov: bool):
    N, T = X.shape
    W = np.eye(N)
    Y = X
    s_list = List[np.ndarray[float, 2]]()
    y_list = List[np.ndarray[float, 2]]()
    r_list = List[float]()
    signs = np.ones(N)
    old_signs = np.ones(N)
    current_loss = _loss(Y, W, fun, signs, ortho, extended)
    have_loss = True
    sign_change = False
    cov = covariance
    if extended and not have_cov:
        cov = (X @ X.T) / T
    C = cov.copy()
    direction = np.zeros((N, N))
    G_old = np.zeros((N, N))
    n_done = 0
    converged = False
    for n in range(max_iter):
        n_done = n
        psiY, psidY = _score_and_der(Y, fun)
        G = (psiY @ Y.T) / T
        Y_square = Y ** 2
        if extended:
            K = np.mean(psidY, axis=1)
            for i in range(N):
                K[i] = K[i] * C[i, i] - G[i, i]
            signs = np.sign(K)
            if n > 0:
                sign_change = False
                for i in range(N):
                    if signs[i] != old_signs[i]:
                        sign_change = True
            old_signs = signs
            for i in range(N):
                for j in range(N):
                    G[i, j] = G[i, j] * signs[i]
                for t in range(T):
                    psidY[i, t] = psidY[i, t] * signs[i]
            if not ortho:
                G += C
                psidY += 1.0
        h_off = np.ones(N)
        if ortho:
            for i in range(N):
                h_off[i] = G[i, i]
        h = np.zeros((N, N))
        if ortho:
            pm = np.mean(psidY, axis=1)
            for i in range(N):
                for j in range(N):
                    v = 0.5 * (pm[i] + pm[j] - h_off[i] - h_off[j])
                    if v < lambda_min:
                        v = lambda_min
                    h[i, j] = v
        else:
            h = (psidY @ Y_square.T) / T
            h = _regularize_hessian(h, h_off, lambda_min)
        if ortho:
            G = (G - G.T) / 2.0
        else:
            G -= np.eye(N)
        gnorm = float(np.max(np.abs(G)))
        if gnorm < tol:
            converged = True
            break
        if n > 0:
            s_list.append(direction)
            y = G - G_old
            y_list.append(y)
            r_list.append(1.0 / float(np.sum(direction * y)))
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                r_list.pop(0)
        G_old = G
        if extended and sign_change:
            have_loss = False
            s_list = List[np.ndarray[float, 2]]()
            y_list = List[np.ndarray[float, 2]]()
            r_list = List[float]()
        direction = _l_bfgs_direction(G, h, h_off, s_list, y_list, r_list, ortho)
        ok, newY, newW, new_loss, direction = _line_search(Y, W, fun, direction, signs, current_loss,
                                                           have_loss, ls_tries, ortho, extended)
        if not ok:
            direction = -G
            s_list = List[np.ndarray[float, 2]]()
            y_list = List[np.ndarray[float, 2]]()
            r_list = List[float]()
            ok2, newY, newW, new_loss, direction = _line_search(Y, W, fun, direction, signs, current_loss,
                                                                have_loss, 10, ortho, extended)
        Y = newY
        W = newW
        if extended or have_cov:
            C = (W @ cov) @ W.T
        current_loss = new_loss
        have_loss = True
    return Y, W, n_done, converged

def _sym_decorrelation(W: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    s, u = sl.eigh(W @ W.T)
    return ((u * (1.0 / np.sqrt(s))) @ u.T) @ W

def picard_unmixing(X: np.ndarray[float, 2], random_state: int, fun: int = 0, ortho: bool = True,
                    extended: bool = True, max_iter: int = 500, tol: float = 1e-7, m: int = 7,
                    ls_tries: int = 10, lambda_min: float = 0.01, fastica_it: int = 0):
    """X is (n_components, n_samples), already whitened (mne passes whiten=False).
    Returns (W, n_iter) with n_iter as picard() reports it (starts at 0)."""
    n, p = X.shape
    X1 = X.copy()
    mean = np.mean(X1, axis=1)
    for i in range(n):
        for j in range(p):
            X1[i, j] -= mean[i]
    w_init = _sym_decorrelation(np.random.RandomState(random_state).normal(size=(n, n)))
    if fastica_it > 0:
        W0 = _sym_decorrelation(w_init)
        for _ in range(fastica_it):
            gw, gd = _score_and_der(W0 @ X1, fun)
            gm = np.mean(gd, axis=1)
            Cc = (gw @ X1.T) / float(p) - gm[:, None] * W0
            W0 = _sym_decorrelation(Cc)
        w_init = W0
    X2 = w_init @ X1
    Y, W, nit, conv = core_picard(X2, fun, ortho, extended, m, max_iter, tol, lambda_min, ls_tries,
                                  np.eye(n), False)
    return W @ w_init, nit
