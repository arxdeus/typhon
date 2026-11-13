# Ported from mne 1.13.2 (BSD-3-Clause)
#
# GradientRemover / remove_fmri_gradient_artifact: average artifact subtraction (AAS).

import numpy as np
from scipy.signal import detrend
from .._info import _picks_to_idx


class GradientRemover:
    _data: np.ndarray[float, 2]
    _tr_events: np.ndarray[int, 1]
    _w0: int
    _w1: int
    _weight_before: float
    _weight_after: float
    _spacing: int
    _corrected: Optional[np.ndarray[float, 2]]

    def __init__(self, eeg_data, tr_events, window = (4, 4), tr_tol: int = 0):
        w0 = 0
        w1 = 0
        if isinstance(window, int):
            if window % 2 != 0:
                raise ValueError(f"Integer windows must be even (received {window}).")
            w0 = window // 2
            w1 = window // 2
        else:
            if len(window) != 2:
                raise ValueError(f"Tuple windows must contain 2 elements (received {window}).")
            w0 = int(window[0])
            w1 = int(window[1])
        if w0 < 0 or w1 < 0:
            raise ValueError(f"Window must contain a positive integer. (Received {window}).")
        if w0 == 0 and w1 == 0:
            raise ValueError(f"Window must contain a positive integer. (Received {window}).")
        self._w0 = w0
        self._w1 = w1
        te = np.asarray(tr_events)
        trs = np.empty(0, dtype = int)
        if te.ndim == 2:
            if te.shape[1] == 3:
                trs = np.empty(te.shape[0], dtype = int)
                for i in range(te.shape[0]):
                    trs[i] = int(te[i, 0])
            else:
                raise ValueError(f"TRs must be a 1D array or a (N, 3) ndarray from mne. Received array of shape {te.shape}.")
        elif te.ndim == 1:
            trs = np.empty(te.shape[0], dtype = int)
            for i in range(te.shape[0]):
                trs[i] = int(te[i])
        else:
            raise ValueError(f"TRs must be a 1D array or a (N, 3) ndarray from mne. Received array of shape {te.shape}.")
        if tr_tol < 0:
            raise ValueError(f"tr_tol must be a non-negative integer (received {tr_tol}).")
        n = trs.shape[0]
        diffs = np.empty(n - 1)
        for i in range(n - 1):
            diffs[i] = float(trs[i + 1] - trs[i])
        med = float(np.median(diffs))
        for i in range(n - 1):
            if abs(diffs[i] - med) > float(tr_tol):
                raise ValueError(f"TR spacings are not consistent (median spacing {med}, tolerance {tr_tol} samples).")
        self._tr_events = trs
        self._spacing = int(round(med))
        d = np.asarray(eeg_data, dtype = float)
        if trs[n - 1] + self._spacing > d.shape[1]:
            raise ValueError(f"Last TR event is sample {trs[n - 1]} but eeg data only contains {d.shape[1]} samples. Please check your tr event markers.")
        self._data = d
        tot = w0 + w1
        self._weight_before = float(w0) / float(tot)
        self._weight_after = float(w1) / float(tot)
        self._corrected = None

    @property
    def window(self) -> Tuple[int, int]:
        return (self._w0, self._w1)

    @property
    def tr_spacing(self) -> int:
        return self._spacing

    @property
    def n_tr(self) -> int:
        return self._tr_events.shape[0]

    @property
    def n_channels(self) -> int:
        return self._data.shape[0]

    @property
    def corrected(self) -> np.ndarray[float, 2]:
        if self._corrected is not None:
            return self._corrected.__val__()
        return self.correct()

    def _check_valid_tr(self, n: int):
        if n < 0 or n >= self.n_tr:
            raise ValueError(f"Index {n} not in TR range [0, {self.n_tr - 1}]")

    def get_tr(self, n: int) -> np.ndarray[float, 2]:
        self._check_valid_tr(n)
        s = int(self._tr_events[n])
        return np.ascontiguousarray(self._data[:, s:s + self._spacing])

    def get_tr_detrended(self, n: int) -> np.ndarray[float, 2]:
        return detrend(self.get_tr(n))

    def _part(self, start: int, stop: int) -> np.ndarray[float, 2]:
        acc = np.zeros((self.n_channels, self._spacing))
        for tr in range(start, stop):
            acc = acc + self.get_tr_detrended(tr)
        return acc / float(stop - start)

    def get_tr_template(self, n: int) -> np.ndarray[float, 2]:
        self._check_valid_tr(n)
        if n < self._w0 or n >= (self.n_tr - self._w1):
            return np.zeros((self.n_channels, self._spacing))
        out = np.zeros((self.n_channels, self._spacing))
        if self._w0 > 0:
            out = out + self._weight_before * self._part(n - self._w0, n)
        if self._w1 > 0:
            out = out + self._weight_after * self._part(n + 1, n + self._w1 + 1)
        return out

    def get_tr_corrected(self, n: int) -> np.ndarray[float, 2]:
        return self.get_tr_detrended(n) - self.get_tr_template(n)

    def correct(self) -> np.ndarray[float, 2]:
        corrected = self._data.copy()
        for tr in range(self.n_tr):
            s = int(self._tr_events[tr])
            c = self.get_tr_corrected(tr)
            corrected[:, s:s + self._spacing] = c
        self._corrected = corrected
        return corrected


def remove_fmri_gradient_artifact(raw, tr_events, *, window = (4, 4), tr_tol: int = 0, picks = None,
                                  method: str = "AAS", copy: bool = True, verbose = None):
    """Remove the fMRI gradient (imaging) artifact from EEG data (mne 1.13.2)."""
    if method != "AAS":
        raise ValueError(f"method must be one of ['AAS'], got {method}")
    pk = _picks_to_idx(raw.info, picks, "data", "bads")
    out = raw.copy() if copy else raw
    nch = len(pk)
    d = np.empty((nch, out._data.shape[1]))
    for a in range(nch):
        d[a, :] = out._data[pk[a], :]
    rem = GradientRemover(d, tr_events, window, tr_tol)
    c = rem.correct()
    for a in range(nch):
        out._data[pk[a], :] = c[a, :]
    return out
