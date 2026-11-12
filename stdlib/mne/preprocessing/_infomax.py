# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Infomax / extended Infomax ICA (Bell & Sejnowski; Lee et al.), as in
# mne.preprocessing.infomax.  An integer random_state seeds a legacy
# RandomState like mne's check_random_state.

import numpy as np
import math

def _kurt_rows(x: np.ndarray[float, 2]):
    n = x.shape[1]
    out = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        mu = 0.0
        for j in range(n):
            mu += x[i, j]
        mu /= n
        m2 = 0.0
        m4 = 0.0
        for j in range(n):
            d = x[i, j] - mu
            d2 = d * d
            m2 += d2
            m4 += d2 * d2
        m2 /= n
        m4 /= n
        out[i] = m4 / (m2 * m2) - 3.0
    return out

def _expit(u):
    return 1.0 / (1.0 + np.exp(-u))

def infomax(data, weights = None, l_rate: Optional[float] = None, block: Optional[int] = None,
            w_change: float = 1e-12, anneal_deg: float = 60.0, anneal_step: float = 0.9,
            extended: bool = True, n_subgauss: int = 1, kurt_size: int = 6000,
            ext_blocks: int = 1, max_iter: int = 200, blowup: float = 1e4,
            blowup_fac: float = 0.5, n_small_angle: Optional[int] = 20, use_bias: bool = True,
            random_state: int = 0):
    X = np.ascontiguousarray(np.asarray(data, dtype=float))
    rng = np.random.RandomState(random_state)
    max_weight = 1e8
    restart_fac = 0.9
    min_l_rate = 1e-10
    degconst = 180.0 / np.pi
    extmomentum = 0.5
    signsbias = 0.02
    signcount_threshold = 25
    signcount_step = 2
    n_samples = X.shape[0]
    n_features = X.shape[1]
    nsq = n_features * n_features
    lr = 0.01 / math.log(float(n_features) ** 2.0) if l_rate is None else l_rate.__val__()
    blk = int(math.floor(math.sqrt(n_samples / 3.0))) if block is None else block.__val__()
    nblock = n_samples // blk
    lastt = (nblock - 1) * blk + 1
    W = np.identity(n_features)
    if weights is not None:
        W = np.asarray(weights, dtype=float).T.copy()
    BI = float(blk) * np.identity(n_features)
    bias = np.zeros((n_features, 1))
    startweights = W.copy()
    oldweights = startweights.copy()
    step = 0
    count_small_angle = 0
    wts_blowup = False
    blockno = 0
    signcount = 0
    eb = ext_blocks
    initial_ext_blocks = ext_blocks
    signs = np.ones(n_features)
    for k in range(n_subgauss):
        signs[k] = -1.0
    ks = min(kurt_size, n_samples)
    old_kurt = np.zeros(n_features)
    oldsigns = np.zeros(n_features)
    olddelta = np.zeros((1, nsq))
    oldchange = 0.0
    first = True
    mi = max_iter
    while step < mi:
        permute = np.argsort(rng.uniform(size=n_samples))
        t = 0
        while t < lastt:
            u = X[permute[t:t + blk], :] @ W
            u += bias.T
            if extended:
                y = np.tanh(u)
                W += lr * (W @ (BI - signs[None, :] * (u.T @ y) - u.T @ u))
                if use_bias:
                    bias += lr * (y.sum(axis=0) * -2.0).reshape((n_features, 1))
            else:
                y = _expit(u)
                W += lr * (W @ (BI + u.T @ (1.0 - 2.0 * y)))
                if use_bias:
                    bias += lr * (1.0 - 2.0 * y).sum(axis=0).reshape((n_features, 1))
            if np.max(np.abs(W)) > max_weight:
                wts_blowup = True
            blockno += 1
            if wts_blowup:
                break
            if extended:
                if eb > 0 and blockno % eb == 0:
                    tp = np.empty((0, 0))
                    if ks < n_samples:
                        rp = np.floor(rng.uniform(0.0, 1.0, ks) * float(n_samples - 1)).astype(int)
                        tp = np.ascontiguousarray((X[rp, :] @ W).T)
                    else:
                        tp = np.ascontiguousarray((X @ W).T)
                    kurt = _kurt_rows(tp)
                    kurt = extmomentum * old_kurt + (1.0 - extmomentum) * kurt
                    old_kurt = kurt
                    signs = np.sign(kurt + signsbias)
                    ndiff = int((signs - oldsigns != 0.0).sum())
                    if ndiff == 0:
                        signcount += 1
                    else:
                        signcount = 0
                    oldsigns = signs
                    if signcount >= signcount_threshold:
                        eb = int(eb * signcount_step)
                        signcount = 0
            t += blk
        if not wts_blowup:
            oldwtchange = W - oldweights
            step += 1
            angledelta = 0.0
            delta = oldwtchange.reshape((1, nsq))
            change = float((delta * delta).sum())
            if step > 2:
                c = float((delta * olddelta).sum()) / math.sqrt(change * oldchange)
                angledelta = math.acos(min(max(c, -1.0), 1.0)) * degconst
            oldweights = W.copy()
            if angledelta > anneal_deg:
                lr *= anneal_step
                olddelta = delta
                oldchange = change
                count_small_angle = 0
            else:
                if step == 1:
                    olddelta = delta
                    oldchange = change
                if n_small_angle is not None:
                    count_small_angle += 1
                    if count_small_angle > n_small_angle.__val__():
                        mi = step
            if step > 2 and change < w_change:
                step = mi
            elif change > blowup:
                lr *= blowup_fac
        else:
            step = 0
            wts_blowup = False
            blockno = 1
            lr *= restart_fac
            W = startweights.copy()
            oldweights = startweights.copy()
            olddelta = np.zeros((1, nsq))
            bias = np.zeros((n_features, 1))
            eb = initial_ext_blocks
            signs = np.ones(n_features)
            for k in range(n_subgauss):
                signs[k] = -1.0
            oldsigns = np.zeros(n_features)
            if lr <= min_l_rate:
                raise ValueError("Error in Infomax ICA: unmixing_matrix matrixmight not be invertible!")
    return np.ascontiguousarray(W.T), step
