# Copyright (C) 2026 the Typhon authors
# Ported from autoreject 0.5.2 (BSD-3-Clause)

"""RANSAC bad channel detection (PREP pipeline).

Typhon notes: EEG only (MEG field mapping raises NotImplementedError);
n_jobs is accepted but the work runs serially (results are identical).
"""

import numpy as np
from numpy.random.legacy import check_random_state
from mne import pick_info
from mne.channels import _make_interpolation_matrix
from .utils import _handle_picks, _check_data, _get_channel_type


class Ransac[RS = int]:
    """RANSAC algorithm to find bad sensors and repair them."""
    n_resample: int
    min_channels: float
    min_corr: float
    unbroken_time: float
    n_jobs: int
    random_state: RS
    verbose: bool
    picks_in: Optional[List[int]]
    picks: np.ndarray[int, 1]
    ch_type: str
    ch_subsets_: List[List[str]]
    mappings_: np.ndarray[float, 2]
    corr_: np.ndarray[float, 2]
    bad_log: np.ndarray[float, 2]
    bad_chs_: List[str]

    def __init__(self, n_resample: int = 50, min_channels: float = 0.25, min_corr: float = 0.75,
                 unbroken_time: float = 0.4, n_jobs: int = 1, random_state: RS = 435656,
                 picks = None, verbose: bool = True):
        self.n_resample = n_resample
        self.min_channels = min_channels
        self.min_corr = min_corr
        self.unbroken_time = unbroken_time
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.picks_in = None if isinstance(picks, NoneType) else [int(p) for p in picks]
        self.picks = np.empty(0, dtype=int)
        self.ch_type = ""
        self.ch_subsets_ = List[List[str]]()
        self.mappings_ = np.empty((0, 0))
        self.corr_ = np.empty((0, 0))
        self.bad_log = np.empty((0, 0))
        self.bad_chs_ = List[str]()

    def _get_random_subsets(self, info) -> List[List[str]]:
        rng = check_random_state(self.random_state)
        picked_info = pick_info(info, self.picks)
        n_channels = len(picked_info.ch_names)
        n_samples = int(np.round(self.min_channels * n_channels))
        ch_list = [ch for ch in picked_info.ch_names if ch not in info.bads]
        assert len(ch_list) >= n_samples, ('too many channels marked as bad,'
                                           'cannot perform interpolation with'
                                           f'min_channels={self.min_channels}')
        subsets = List[List[str]]()
        arr = np.array(list(range(len(ch_list))), dtype=int)
        for _ in range(self.n_resample):
            sel = rng.choice(arr, size=n_samples, replace=False)
            subsets.append([ch_list[int(s)] for s in sel])
        return subsets

    def _get_mappings(self, info, ch_subsets: List[List[str]]) -> np.ndarray[float, 2]:
        picked_info = pick_info(info, self.picks)
        n_channels = picked_info.nchan
        pos = np.array([[picked_info.locs[i][0], picked_info.locs[i][1], picked_info.locs[i][2]]
                        for i in range(n_channels)])
        ch_names = picked_info.ch_names
        out = np.zeros((n_channels * len(ch_subsets), n_channels))
        for s in range(len(ch_subsets)):
            pick_from = np.array([ch_names.index(name) for name in ch_subsets[s]], dtype=int)
            if self.ch_type == 'meg':
                raise NotImplementedError("MEG RANSAC is not available in Typhon autoreject")
            m = _make_interpolation_matrix(pos[pick_from], pos, alpha=1e-5)
            for i in range(n_channels):
                for j in range(len(pick_from)):
                    out[s * n_channels + i, pick_from[j]] = m[i, j]
        return out

    def _compute_correlations(self, data: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
        mappings = self.mappings_
        n_channels = data.shape[0]
        n_times = data.shape[1]
        y_pred = data.T.dot(mappings.T)
        n_p = len(self.picks)
        # reshape (n_times, n_picks, n_resample) order='F' then median on last axis
        med = np.empty((n_times, n_p))
        buf = np.empty(self.n_resample)
        for t in range(n_times):
            for c in range(n_p):
                for r in range(self.n_resample):
                    buf[r] = y_pred[t, c + n_p * r]
                med[t, c] = np.median(buf)
        num = np.sum(data.T * med, axis=0)
        denom = (np.sqrt(np.sum(data.T ** 2, axis=0)) *
                 np.sqrt(np.sum(med ** 2, axis=0)))
        return num / denom

    def fit(self, epochs):
        """Perform RANSAC on the given epochs; bad channels in ``bad_chs_``."""
        if self.picks_in is None:
            self.picks = _handle_picks(epochs.info, None)
        else:
            self.picks = _handle_picks(epochs.info, self.picks_in.__val__())
        _check_data(epochs, self.picks, 'single_channel_type', True, self.verbose)
        self.ch_type = _get_channel_type(epochs, self.picks)
        n_epochs = len(epochs)
        self.ch_subsets_ = self._get_random_subsets(epochs.info)
        self.mappings_ = self._get_mappings(epochs.info, self.ch_subsets_)
        X = epochs.get_data(copy=False)
        self.corr_ = np.zeros((n_epochs, len(self.picks)))
        for e in range(n_epochs):
            self.corr_[e, :] = self._compute_correlations(X[e][self.picks, :])
        if self.verbose:
            print('[Done]')
        self.bad_log = np.zeros_like(self.corr_)
        self.bad_log[self.corr_ < self.min_corr] = 1.0
        bad_log = self.bad_log.sum(axis=0)
        self.bad_chs_ = List[str]()
        for p in range(bad_log.shape[0]):
            if bad_log[p] > self.unbroken_time * n_epochs:
                self.bad_chs_.append(epochs.info.ch_names[int(self.picks[p])])
        return self

    def transform(self, epochs):
        epochs = epochs.copy()
        _check_data(epochs, self.picks, 'single_channel_type', True, self.verbose)
        epochs.info.bads = list(self.bad_chs_)
        epochs.interpolate_bads(reset_bads=True)
        return epochs

    def fit_transform(self, epochs):
        return self.fit(epochs).transform(epochs)
