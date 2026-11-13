# Ported from mne 1.13.2 (BSD-3-Clause)
#
# realign_raw: realign two simultaneous recordings using shared event times.

import math
import numpy as np
from scipy.stats import pearsonr


def realign_raw(raw, other, t_raw, t_other, *, verbose = None):
    """Realign two simultaneous recordings in place (mne 1.13.2)."""
    tr = np.array(t_raw, dtype = float)
    to = np.array(t_other, dtype = float)
    if tr.ndim != 1 or tr.shape[0] != to.shape[0]:
        raise ValueError(f"t_raw and t_other must be 1D with the same shape, got shapes {tr.shape} and {to.shape}")
    n = tr.shape[0]
    # 1. zero- and first-order correction: least-squares line t_raw = zero + first * t_other
    mx = float(to.mean())
    my = float(tr.mean())
    sxx = 0.0
    sxy = 0.0
    for i in range(n):
        sxx += (to[i] - mx) * (to[i] - mx)
        sxy += (to[i] - mx) * (tr[i] - my)
    first_ord = sxy / sxx
    zero_ord = my - first_ord * mx
    res = pearsonr(to, tr)
    r = float(res.statistic)
    p = float(res.pvalue)
    if p > 0.05 or r <= 0.0:
        raise ValueError(f"Linear correlation computed as R={r:0.3f} and p={p:0.2e}, cannot resample safely")
    # 2. crop start
    if zero_ord > 0.0:
        raw.crop(zero_ord, None)
        for i in range(n):
            tr[i] -= zero_ord
    elif zero_ord < 0.0:
        t_crop = -zero_ord / first_ord
        other.crop(t_crop, None)
        for i in range(n):
            to[i] -= t_crop
    # 3. resample other using the first-order term
    sfreq_new = raw.info.sfreq * first_ord
    other.resample(sfreq_new)
    other.info.sfreq = raw.info.sfreq
    # 4. correct annotations in other
    ann = other.annotations
    ann.onset = [o * first_ord for o in ann.onset]
    ann.duration = [d * first_ord for d in ann.duration]
    # 5. crop the end of one of the recordings if necessary
    nr = raw.times.shape[0]
    no = other.times.shape[0]
    delta = float(raw.times[nr - 1]) - float(other.times[no - 1])
    if delta > 0.0:
        raw.crop(0.0, float(other.times[no - 1]))
    elif delta < 0.0:
        other.crop(0.0, float(raw.times[nr - 1]))
