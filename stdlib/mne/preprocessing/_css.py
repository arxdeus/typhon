# Ported from mne 1.13.2 (BSD-3-Clause)
#
# cortical_signal_suppression: remove the common temporal subspace of magnetometers
# and gradiometers from a set of channels (Samuelsson et al. 2019).

import numpy as np
from .._info import _picks_to_idx
from .._containers import Evoked


def _temp_proj(ref_2: np.ndarray[float, 2], ref_1: np.ndarray[float, 2], raw_data: np.ndarray[float, 2],
               n_proj: int) -> np.ndarray[float, 2]:
    q1, r1 = np.linalg.qr(ref_1.T.copy())
    q2, r2 = np.linalg.qr(ref_2.T.copy())
    cross = np.dot(q1.T, q2)
    w, s, vt = np.linalg.svd(cross)
    proj_mat = np.dot(q1, w)
    pv = np.ascontiguousarray(proj_mat[:, :n_proj])
    weights = np.dot(raw_data, pv)
    return raw_data - np.dot(weights, pv.T)


def cortical_signal_suppression(evoked, picks = None, mag_picks = None, grad_picks = None,
                                n_proj: int = 6, *, verbose = None):
    """Apply cortical signal suppression (CSS) to evoked data (mne 1.13.2)."""
    pk = _picks_to_idx(evoked.info, picks, "data", "bads")
    mp = _picks_to_idx(evoked.info, mag_picks, "mag", "bads")
    gp = _picks_to_idx(evoked.info, grad_picks, "grad", "bads")
    out = evoked.copy()
    all_data = evoked.data
    mag = np.empty((len(mp), all_data.shape[1]))
    for a in range(len(mp)):
        mag[a, :] = all_data[mp[a], :]
    grad = np.empty((len(gp), all_data.shape[1]))
    for a in range(len(gp)):
        grad[a, :] = all_data[gp[a], :]
    d = np.empty((len(pk), all_data.shape[1]))
    for a in range(len(pk)):
        d[a, :] = all_data[pk[a], :]
    r = _temp_proj(mag, grad, d, n_proj)
    for a in range(len(pk)):
        out.data[pk[a], :] = r[a, :]
    return out
