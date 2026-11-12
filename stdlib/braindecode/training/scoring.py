# Ported from braindecode 2.0.0 (BSD-3-Clause)

import numpy as np


def trial_preds_from_window_preds(preds, i_window_in_trials, i_stop_in_trials):
    """Merge per-window predictions (n_windows, n_classes, n_preds) into one
    (n_classes, n_trial_preds) array per trial, removing overlap."""
    P = np.asarray(preds).astype(float)
    n = P.shape[0]
    if not (n == len(i_window_in_trials) == len(i_stop_in_trials)):
        raise AssertionError(f"{n}, {len(i_window_in_trials)}, {len(i_stop_in_trials)}")
    out = List[np.ndarray[float, 2]]()
    cur = List[np.ndarray[float, 2]]()
    i_last_window = -1
    has_last = False
    i_last_stop = 0
    for k in range(n):
        wp = P[k]
        iw = int(i_window_in_trials[k])
        st = int(i_stop_in_trials[k])
        if iw != i_last_window + 1:
            if iw != 0:
                raise AssertionError("window numbers in new trial should start from 0")
            if len(cur) > 0:
                out.append(np.concatenate(cur, axis=1))
                cur = List[np.ndarray[float, 2]]()
            has_last = False
        if has_last:
            need = st - i_last_stop
            wp = wp[:, wp.shape[1] - need:]
        cur.append(wp.copy())
        i_last_window = iw
        i_last_stop = st
        has_last = True
    if len(cur) > 0:
        out.append(np.concatenate(cur, axis=1))
    return out
