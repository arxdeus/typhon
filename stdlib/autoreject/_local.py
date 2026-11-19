# Copyright (C) 2026 the Typhon authors
# Ported from autoreject 0.5.2 (BSD-3-Clause)

"""Local (sensor-level) rejection: _AutoReject and its cross-validation.

Typhon notes: the custom ``thresh_func`` argument is not supported, the
thresholds always come from compute_thresholds with the stored
method/random_state. Epochs are EEG (see autoreject.utils).
"""

import numpy as np
from .utils import (_handle_picks, _check_data, _get_picks_by_type, _interp_matrices,
                    _pprint, _verbose_bool)
from ._thresholds import _compute_thresholds, _slicemean


class RejectLog:
    """The rejection log: labels 0 good, 1 bad, 2 bad and interpolated,
    NaN for channels that were not considered."""
    bad_epochs: np.ndarray[bool, 1]
    labels: np.ndarray[float, 2]
    ch_names: List[str]

    def __init__(self, bad_epochs, labels, ch_names):
        self.bad_epochs = np.asarray(bad_epochs).astype(bool)
        self.labels = np.asarray(labels, dtype=float)
        self.ch_names = [str(c) for c in ch_names]
        assert self.bad_epochs.shape[0] == self.labels.shape[0]
        assert len(self.ch_names) == self.labels.shape[1]

    def plot(self, orientation: str = 'vertical', show_names = 'auto', aspect: str = 'equal',
             show: bool = True, ax = None):
        """Plotting is not available in Typhon (no-op, returns None)."""
        if orientation != 'vertical' and orientation != 'horizontal':
            raise ValueError(f"orientation can be only 'horizontal' or 'vertical'. Got {orientation}")
        return None

    def plot_epochs(self, epochs, scalings = None, title: str = ''):
        """Plotting is not available in Typhon. Checks the inputs like the
        reference and returns None."""
        if self.labels.shape[0] != len(epochs.events):
            raise ValueError('The number of epochs should match the number of'
                             'epochs *before* autoreject. Please provide'
                             'the epochs object before running autoreject')
        if self.labels.shape[1] != len(epochs.ch_names):
            raise ValueError('The number of channels should match the number'
                             ' of channels before running autoreject.')
        return None

    def save(self, fname: str, overwrite: bool = False):
        """Save as .npz (ch_names stored as an int array of UTF-8 bytes
        joined by newlines, readable by read_reject_log)."""
        import os
        if not overwrite and os.path.isfile(fname):
            raise ValueError(f'{fname} already exists. Please set '
                             'overwrite=True if you want to overwrite it.')
        enc = '\n'.join(self.ch_names).encode()
        names = np.array([int(enc[i]) for i in range(len(enc))], dtype=int)
        np.savez_compressed(fname, ch_names=names, bad_epochs=self.bad_epochs,
                            labels=self.labels)


def read_reject_log(fname: str) -> RejectLog:
    """Read a reject log written by RejectLog.save."""
    from numpy.npio import load_npz
    d = load_npz(fname)
    names = d.get('ch_names', 1, int)
    s = bytes([int(names[i]) for i in range(names.shape[0])]).decode() if names.shape[0] > 0 else ''
    return RejectLog(d.get('bad_epochs', 1, bool), d.get('labels', 2, float), s.split('\n'))


def _get_interp_chs(labels: np.ndarray[float, 2], ch_names: List[str], picks) -> List[List[str]]:
    """Per epoch, the names of the channels to interpolate (label 2)."""
    assert labels.shape[1] == len(ch_names)
    pk = [int(p) for p in picks]
    assert labels.shape[1] > max(pk)
    not_nan = [j for j in range(labels.shape[1]) if np.any(~np.isnan(labels[:, j]))]
    if not_nan != pk:
        raise AssertionError("labels must be non-NaN exactly at the picks")
    out = List[List[str]]()
    for i in range(labels.shape[0]):
        out.append([ch_names[j] for j in range(labels.shape[1]) if labels[i, j] == 2])
    return out


def _interpolate_bad_epochs(epochs, interp_channels: List[List[str]], picks, dots = None,
                            verbose = True):
    """Repair, in place, each epoch's channels to interpolate."""
    assert len(epochs) == len(interp_channels)
    keys = List[List[str]]()
    groups = Dict[str, List[int]]()
    for e in range(len(interp_channels)):
        chs = interp_channels[e]
        if len(chs) > 0:
            k = '\x00'.join(chs)
            if k not in groups:
                groups[k] = List[int]()
                keys.append(chs)
            groups[k].append(e)
    if len(keys) == 0:
        return
    pk = [int(p) for p in picks]
    pa = np.array(pk, dtype=int)
    pick_of = Dict[str, int]()
    for p in pk:
        pick_of[epochs.ch_names[p]] = p
    mats = _interp_matrices(epochs.info, pk, dots, keys, 'accurate', 'Repairing epochs', verbose)
    d = epochs._data
    for g in range(len(keys)):
        bad_picks = [pick_of[ch] for ch in keys[g]]
        mat = mats[g]
        for e in groups['\x00'.join(keys[g])]:
            res = mat @ d[e][pa, :]
            for k in range(len(bad_picks)):
                d[e, bad_picks[k], :] = res[k]


class _AutoReject:
    """Automatically reject bad epochs and repair bad trials for one
    channel type, given n_interpolate (rho) and consensus (kappa / Q)."""
    n_interpolate: int
    consensus: float
    picks: Optional[List[int]]
    thresh_method: str
    random_state: Optional[int]
    verbose: bool
    dots: Optional[int]
    picks_: np.ndarray[int, 1]
    threshes_: Dict[str, float]
    consensus_: Dict[str, float]
    n_interpolate_: Dict[str, int]
    mean_: np.ndarray[float, 2]

    def __init__(self, n_interpolate: int = 0, consensus: float = 0.1, thresh_func = None,
                 picks = None, thresh_method: str = 'bayesian_optimization', dots = None,
                 verbose = True, random_state: Optional[int] = None):
        if not (0 <= consensus <= 1):
            raise ValueError(f'"consensus" must be between 0 and 1. You gave me {consensus}.')
        self.consensus = consensus
        self.n_interpolate = n_interpolate
        self.picks = None if isinstance(picks, NoneType) else [int(p) for p in picks]
        self.thresh_method = thresh_method
        self.random_state = random_state
        self.verbose = _verbose_bool(verbose)
        self.dots = None
        self.picks_ = np.empty(0, dtype=int)
        self.threshes_ = Dict[str, float]()
        self.consensus_ = Dict[str, float]()
        self.n_interpolate_ = Dict[str, int]()
        self.mean_ = np.empty((0, 0))

    def __repr__(self) -> str:
        name = '_AutoReject'
        pr = 'None' if self.picks is None else str(self.picks.__val__())
        params = [('n_interpolate', str(self.n_interpolate)), ('consensus', str(self.consensus)),
                  ('verbose', str(self.verbose)), ('picks', pr)]
        return f'{name}({_pprint(params, len(name))})'

    def _vote_bad_epochs(self, epochs, picks) -> Tuple[np.ndarray[float, 2], np.ndarray[int, 1]]:
        """Each channel votes for an epoch as good or bad."""
        pk = [int(p) for p in picks]
        n_ep = len(epochs)
        labels = np.full((n_ep, len(epochs.ch_names)), np.nan)
        deltas = np.ptp(epochs.get_data(pk), axis=-1).T
        for ci in range(len(pk)):
            th = self.threshes_[epochs.ch_names[pk[ci]]]
            for e in range(n_ep):
                labels[e, pk[ci]] = 1.0 if deltas[ci, e] > th else 0.0
        counts = np.zeros(n_ep, dtype=int)
        for e in range(n_ep):
            c = 0
            for j in range(labels.shape[1]):
                if labels[e, j] == 1:
                    c += 1
            counts[e] = c
        return labels, counts

    def _get_epochs_interpolation(self, epochs, labels: np.ndarray[float, 2], picks,
                                  n_interpolate: int, verbose = True) -> np.ndarray[float, 2]:
        """Mark (label 2) the channels to interpolate in each epoch: all bad
        ones if at most n_interpolate, else the n_interpolate worst."""
        assert labels.shape[0] == len(epochs)
        assert labels.shape[1] == len(epochs.ch_names)
        labels = labels.copy()
        pk = [int(p) for p in picks]
        pick_set = Set[int](pk)
        nch = labels.shape[1]
        all_peaks = np.empty((0, 0))
        have_peaks = False
        for e in range(len(epochs)):
            n_bads = 0.0
            for p in pk:
                n_bads += labels[e, p]
            if n_bads == 0:
                continue
            mask = np.array([labels[e, j] == 1 for j in range(nch)])
            if n_bads > n_interpolate:
                if not have_peaks:
                    all_peaks = np.ptp(epochs.get_data(copy=False), axis=-1)
                    have_peaks = True
                peaks = all_peaks[e].copy()
                for j in range(nch):
                    if j not in pick_set or not mask[j]:
                        peaks[j] = -np.inf
                order = np.argsort(peaks)[::-1]
                for k in range(n_interpolate, nch):
                    mask[int(order[k])] = False
            for j in range(nch):
                if mask[j]:
                    labels[e, j] = 2.0
        return labels

    def _get_bad_epochs(self, bad_sensor_counts: np.ndarray[int, 1], ch_type: str,
                        picks) -> np.ndarray[bool, 1]:
        """Mask of the epochs where at least consensus * n_channels vote bad."""
        sorted_epoch_idx = np.argsort(bad_sensor_counts)[::-1]
        counts = np.sort(bad_sensor_counts)[::-1]
        n_consensus = self.consensus_[ch_type] * len(picks)
        bad_epochs = np.zeros(counts.shape[0], dtype=bool)
        if np.max(counts) >= n_consensus:
            n_drop = 0
            for i in range(counts.shape[0]):
                if counts[i] >= n_consensus:
                    n_drop += 1
            for i in range(n_drop):
                bad_epochs[int(sorted_epoch_idx[i])] = True
        return bad_epochs

    def get_reject_log(self, epochs, threshes = None, picks = None) -> RejectLog:
        pk = self.picks_ if isinstance(picks, NoneType) else _handle_picks(epochs.info, picks)
        by_type = _get_picks_by_type(epochs.info, pk)
        assert len(by_type) == 1
        ch_type, this_picks = by_type[0]
        labels, counts = self._vote_bad_epochs(epochs, this_picks)
        labels = self._get_epochs_interpolation(epochs, labels, this_picks,
                                                self.n_interpolate_[ch_type])
        bad_epochs = self._get_bad_epochs(counts, ch_type, this_picks)
        return RejectLog(bad_epochs, labels, list(epochs.ch_names))

    def fit(self, epochs):
        """Compute the thresholds and the clean mean."""
        pin = None if self.picks is None else self.picks.__val__()
        self.picks_ = _handle_picks(epochs.info, pin)
        _check_data(epochs, self.picks_, 'single_channel_type', True, self.verbose)
        by_type = _get_picks_by_type(epochs.info, self.picks_)
        assert len(by_type) == 1
        ch_type, this_picks = by_type[0]
        self.consensus_ = {ch_type: self.consensus}
        self.n_interpolate_ = {ch_type: self.n_interpolate}
        self.threshes_ = _compute_thresholds(epochs.copy(), self.thresh_method, self.random_state,
                                             [int(p) for p in self.picks_], True, None,
                                             self.verbose, 1)
        reject_log = self.get_reject_log(epochs, None, [int(p) for p in self.picks_])
        epochs_copy = epochs.copy()
        interp_channels = _get_interp_chs(reject_log.labels, reject_log.ch_names, this_picks)
        _interpolate_bad_epochs(epochs_copy, interp_channels, self.picks_, None, self.verbose)
        good = [i for i in range(len(epochs)) if not reject_log.bad_epochs[i]]
        self.mean_ = _slicemean(epochs_copy.get_data(copy=False), good)
        return self

    def score(self, X, y = None) -> float:
        if np.any(np.isnan(self.mean_)):
            return -np.inf
        return -np.sqrt(np.mean((np.median(X, axis=0) - self.mean_) ** 2))


def _run_local_reject_cv(epochs, thresh_method: str, random_state: Optional[int], picks_,
                         n_interpolate: np.ndarray[int, 1], cv, consensus: np.ndarray[float, 1],
                         dots, verbose):
    """Fit the thresholds once, then the CV loss over (consensus, n_interp)."""
    n_folds = cv.get_n_splits()
    loss = np.zeros((consensus.shape[0], n_interpolate.shape[0], n_folds))
    local_reject = _AutoReject(picks=picks_, verbose=verbose, thresh_method=thresh_method,
                               random_state=random_state)
    local_reject.fit(epochs)
    assert len(local_reject.consensus_) == 1
    ch_type = next(iter(local_reject.consensus_.keys()))
    labels, bad_sensor_counts = local_reject._vote_bad_epochs(epochs, picks_)
    n_channels = len(picks_)
    X = epochs.get_data(picks_)
    splits = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
    for tr, te in cv.split(X):
        splits.append((np.asarray(tr, dtype=int), np.asarray(te, dtype=int)))
    medians = [np.median(X[te], axis=0) for _, te in splits]
    for jdx in range(n_interpolate.shape[0]):
        n_interp = int(n_interpolate[jdx])
        local_reject.n_interpolate_[ch_type] = n_interp
        labels = local_reject._get_epochs_interpolation(epochs, labels, picks_, n_interp)
        interp_channels = _get_interp_chs(labels, list(epochs.ch_names), picks_)
        epochs_interp = epochs.copy()
        _interpolate_bad_epochs(epochs_interp, interp_channels, picks_, dots, verbose)
        X_interp = epochs_interp.get_data(picks_)
        for fold in range(len(splits)):
            train = splits[fold][0]
            X_interp_train = X_interp[train]
            counts_train = bad_sensor_counts[train]
            median = medians[fold]
            for idx in range(consensus.shape[0]):
                this_consensus = float(consensus[idx])
                if this_consensus * n_channels <= n_interp:
                    loss[idx, jdx, fold] = np.inf
                    continue
                local_reject.consensus_[ch_type] = this_consensus
                bad_epochs = local_reject._get_bad_epochs(counts_train, ch_type, picks_)
                good = [i for i in range(bad_epochs.shape[0]) if not bad_epochs[i]]
                mean_ = _slicemean(X_interp_train, good)
                local_reject.mean_ = mean_
                loss[idx, jdx, fold] = (np.inf if np.any(np.isnan(mean_))
                                        else np.sqrt(np.mean((median - mean_) ** 2)))
    return local_reject, loss
