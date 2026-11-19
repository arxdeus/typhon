# Copyright (C) 2026 the Typhon authors
# Ported from autoreject 0.5.2 (BSD-3-Clause)

"""AutoReject: find n_interpolate and consensus by cross-validation.

Typhon notes: ``save``/``read_auto_reject`` use a JSON file instead of
HDF5 (h5io is not available); any file name is accepted. n_jobs runs
serially. MEG data is not supported (see autoreject.utils).
"""

import numpy as np
import json
from sklearn.model_selection import KFold
from .utils import _handle_picks, _check_data, _get_picks_by_type, _pprint, _verbose_bool
from ._local import (_AutoReject, RejectLog, _run_local_reject_cv, _get_interp_chs,
                     _interpolate_bad_epochs)


class AutoReject:
    """Efficiently find n_interpolate (rho) and consensus (kappa / Q).

    n_interpolate defaults to [1, 4, min(n_picks - 1, 32)], consensus to
    np.linspace(0, 1, 11), cv to KFold(10)."""
    n_interpolate: Optional[np.ndarray[int, 1]]
    consensus: np.ndarray[float, 1]
    thresh_method: str
    cv: int
    verbose: bool
    picks: Optional[List[int]]
    n_jobs: int
    random_state: Optional[int]
    dots: Optional[int]
    picks_: np.ndarray[int, 1]
    n_interpolate_: Dict[str, int]
    consensus_: Dict[str, float]
    threshes_: Dict[str, float]
    loss_: Dict[str, np.ndarray[float, 3]]
    local_reject_: Dict[str, _AutoReject]
    _fitted: bool

    def __init__(self, n_interpolate = None, consensus = None, thresh_func = None, cv: int = 10,
                 picks = None, thresh_method: str = 'bayesian_optimization', n_jobs: int = 1,
                 random_state: Optional[int] = None, verbose = True):
        self.n_interpolate = (None if isinstance(n_interpolate, NoneType)
                              else np.asarray(n_interpolate).astype(int))
        self.consensus = (np.linspace(0, 1.0, 11) if isinstance(consensus, NoneType)
                          else np.asarray(consensus).astype(float))
        self.thresh_method = thresh_method
        self.cv = cv
        self.verbose = _verbose_bool(verbose)
        self.picks = None if isinstance(picks, NoneType) else [int(p) for p in picks]
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.dots = None
        self.picks_ = np.empty(0, dtype=int)
        self.n_interpolate_ = Dict[str, int]()
        self.consensus_ = Dict[str, float]()
        self.threshes_ = Dict[str, float]()
        self.loss_ = Dict[str, np.ndarray[float, 3]]()
        self.local_reject_ = Dict[str, _AutoReject]()
        self._fitted = False

    def __repr__(self) -> str:
        name = 'AutoReject'
        ni = 'None' if self.n_interpolate is None else str(self.n_interpolate.__val__())
        params = [('n_interpolate', ni), ('consensus', str(self.consensus)),
                  ('cv', str(self.cv)), ('verbose', str(self.verbose)),
                  ('picks', 'None' if self.picks is None else str(self.picks.__val__())),
                  ('thresh_method', repr(self.thresh_method)),
                  ('random_state', 'None' if self.random_state is None else str(self.random_state.__val__())),
                  ('n_jobs', str(self.n_jobs))]
        return f'{name}({_pprint(params, len(name))})'

    def fit(self, epochs):
        """Fit the thresholds, consensus and n_interpolate on epochs."""
        pin = None if self.picks is None else self.picks.__val__()
        self.picks_ = np.sort(_handle_picks(epochs.info, pin))
        _check_data(epochs, self.picks_, 'data_channels', True, self.verbose)
        cv = KFold(n_splits=self.cv)
        by_type = _get_picks_by_type(epochs.info, self.picks_)
        if self.n_interpolate is None:
            if len(self.picks_) < 4:
                raise ValueError('Too few channels. autoreject is unlikely to be effective')
            max_interp = min(len(self.picks_) - 1, 32)
            self.n_interpolate = np.array([1, 4, max_interp], dtype=int)
        ni = self.n_interpolate.__val__()
        self.n_interpolate_ = Dict[str, int]()
        self.consensus_ = Dict[str, float]()
        self.threshes_ = Dict[str, float]()
        self.loss_ = Dict[str, np.ndarray[float, 3]]()
        self.local_reject_ = Dict[str, _AutoReject]()
        for ch_type, this_picks in by_type:
            if self.verbose:
                print(f'Running autoreject on ch_type={ch_type}')
            lr, loss = _run_local_reject_cv(epochs, self.thresh_method, self.random_state,
                                            this_picks, ni, cv, self.consensus, None,
                                            self.verbose)
            self.threshes_.update(lr.threshes_)
            mloss = loss.mean(axis=-1)
            flat = int(mloss.argmin())
            best_idx = flat // mloss.shape[1]
            best_jdx = flat % mloss.shape[1]
            self.consensus_[ch_type] = float(self.consensus[best_idx])
            self.n_interpolate_[ch_type] = int(ni[best_jdx])
            self.loss_[ch_type] = loss
            lr.consensus_[ch_type] = self.consensus_[ch_type]
            lr.n_interpolate_[ch_type] = self.n_interpolate_[ch_type]
            self.local_reject_[ch_type] = lr
            if self.verbose:
                c = self.consensus_[ch_type]
                print(f'\n\n\n\nEstimated consensus={c:0.2f} and n_interpolate={self.n_interpolate_[ch_type]}')
        self._fitted = True
        return self

    def get_reject_log(self, epochs, picks = None) -> RejectLog:
        """Rejection log; bad_epochs is the union over channel types."""
        ch_names = list(epochs.ch_names)
        reject_log = RejectLog(np.zeros(len(epochs), dtype=bool),
                               np.full((len(epochs), len(ch_names)), np.nan), ch_names)
        pk = self.picks_ if isinstance(picks, NoneType) else _handle_picks(epochs.info, picks)
        for ch_type, this_picks in _get_picks_by_type(epochs.info, pk):
            lg = self.local_reject_[ch_type].get_reject_log(epochs, None, this_picks)
            for p in this_picks:
                reject_log.labels[:, p] = lg.labels[:, p]
            reject_log.bad_epochs = np.logical_or(reject_log.bad_epochs, lg.bad_epochs)
            reject_log.ch_names = lg.ch_names
        return reject_log

    def _transform(self, epochs, reject_log: RejectLog):
        if not self._fitted:
            raise ValueError('Please run autoreject.fit() method first')
        _check_data(epochs, self.picks_, 'data_channels', True, self.verbose)
        epochs_clean = epochs.copy()
        _apply_interp(reject_log, epochs_clean, self.threshes_, self.picks_, self.verbose)
        _apply_drop(reject_log, epochs_clean, self.threshes_, self.picks_, self.verbose)
        return epochs_clean

    def transform(self, epochs, return_log: Literal[bool] = False, reject_log = None):
        """Repair and drop bad epochs; returns a cleaned copy (and the log)."""
        if not self._fitted:
            raise ValueError('Please run autoreject.fit() method first')
        log = RejectLog(np.zeros(0, dtype=bool), np.zeros((0, 0)), List[str]())
        if isinstance(reject_log, NoneType):
            log = self.get_reject_log(epochs)
        elif isinstance(reject_log, RejectLog):
            log = reject_log
        else:
            compile_error("reject_log must be an instance of RejectLog")
        out = self._transform(epochs, log)
        if return_log:
            return out, log
        else:
            return out

    def fit_transform(self, epochs, return_log: Literal[bool] = False):
        return self.fit(epochs).transform(epochs, return_log=return_log)

    def _state(self) -> json.JSON:
        st = json.JSON.object()
        st["consensus"] = json.JSON([float(c) for c in self.consensus])
        st["n_interpolate"] = (json.JSON() if self.n_interpolate is None
                               else json.JSON([int(v) for v in self.n_interpolate.__val__()]))
        st["picks"] = json.JSON() if self.picks is None else json.JSON(self.picks.__val__())
        st["verbose"] = json.JSON(self.verbose)
        st["n_jobs"] = json.JSON(self.n_jobs)
        st["cv"] = json.JSON(self.cv)
        st["random_state"] = (json.JSON() if self.random_state is None
                              else json.JSON(self.random_state.__val__()))
        st["thresh_method"] = json.JSON(self.thresh_method)
        if self._fitted:
            st["threshes_"] = json.JSON(self.threshes_)
            st["n_interpolate_"] = json.JSON(self.n_interpolate_)
            st["consensus_"] = json.JSON(self.consensus_)
            st["picks_"] = json.JSON([int(p) for p in self.picks_])
            lo = json.JSON.object()
            for k, v in self.loss_.items():
                o = json.JSON.object()
                o["shape"] = json.JSON([v.shape[0], v.shape[1], v.shape[2]])
                o["data"] = json.JSON([float(x) if np.isfinite(x) else 1e308 * 10 for x in v.ravel()])
                lo[k] = o
            st["loss_"] = lo
            lr = json.JSON.object()
            for k, v in self.local_reject_.items():
                o = json.JSON.object()
                o["consensus"] = json.JSON(v.consensus)
                o["n_interpolate"] = json.JSON(v.n_interpolate)
                o["picks"] = json.JSON() if v.picks is None else json.JSON(v.picks.__val__())
                o["verbose"] = json.JSON(v.verbose)
                o["threshes_"] = json.JSON(v.threshes_)
                o["n_interpolate_"] = json.JSON(v.n_interpolate_)
                o["consensus_"] = json.JSON(v.consensus_)
                lr[k] = o
            st["local_reject_"] = lr
        return st

    def save(self, fname: str, overwrite: bool = False):
        """Save the AutoReject state (JSON format in Typhon)."""
        import os
        if not overwrite and os.path.isfile(fname):
            raise ValueError(f'{fname} already exists. Please make overwrite=True'
                             'if you want to overwrite this file')
        with open(fname, 'w') as f:
            f.write(json.dumps(self._state()))


def _f_dict(j: json.JSON) -> Dict[str, float]:
    out = Dict[str, float]()
    for k, v in j.items():
        out[k] = v.as_float()
    return out


def _i_dict(j: json.JSON) -> Dict[str, int]:
    out = Dict[str, int]()
    for k, v in j.items():
        out[k] = v.as_int()
    return out


def _opt_ilist(j: json.JSON) -> Optional[List[int]]:
    if j.is_null():
        return None
    return [v.as_int() for v in j.values()]


def read_auto_reject(fname: str) -> AutoReject:
    """Read an AutoReject object written by AutoReject.save."""
    with open(fname) as f:
        st = json.loads(f.read())
    rs: Optional[int] = None if st["random_state"].is_null() else st["random_state"].as_int()
    ar = AutoReject(None, [v.as_float() for v in st["consensus"].values()], None,
                    st["cv"].as_int(), _opt_ilist(st["picks"]), st["thresh_method"].as_str(),
                    st["n_jobs"].as_int(), rs, st["verbose"].as_bool())
    ni = _opt_ilist(st["n_interpolate"])
    if ni is not None:
        ar.n_interpolate = np.array(ni.__val__(), dtype=int)
    if "threshes_" in st:
        ar.threshes_ = _f_dict(st["threshes_"])
        ar.n_interpolate_ = _i_dict(st["n_interpolate_"])
        ar.consensus_ = _f_dict(st["consensus_"])
        ar.picks_ = np.array([v.as_int() for v in st["picks_"].values()], dtype=int)
        for k, o in st["loss_"].items():
            sh = [v.as_int() for v in o["shape"].values()]
            data = np.array([v.as_float() for v in o["data"].values()])
            ar.loss_[k] = data.reshape((sh[0], sh[1], sh[2]))
        for k, o in st["local_reject_"].items():
            lr = _AutoReject(o["n_interpolate"].as_int(), o["consensus"].as_float(), None,
                             _opt_ilist(o["picks"]), verbose=o["verbose"].as_bool())
            lr.threshes_ = _f_dict(o["threshes_"])
            lr.n_interpolate_ = _i_dict(o["n_interpolate_"])
            lr.consensus_ = _f_dict(o["consensus_"])
            lr.picks_ = np.array(lr.picks.__val__() if lr.picks is not None else List[int](), dtype=int)
            ar.local_reject_[k] = lr
        ar._fitted = True
    return ar


def _check_fit(epochs, threshes_: Dict[str, float], picks_):
    for p in picks_:
        if epochs.ch_names[int(p)] not in threshes_:
            raise ValueError('You are passing channels which were not present '
                             'at fit-time. Please fit it again, this time '
                             'correctly.')


def _apply_interp(reject_log: RejectLog, epochs, threshes_: Dict[str, float], picks_, verbose):
    _check_fit(epochs, threshes_, picks_)
    interp_channels = _get_interp_chs(reject_log.labels, reject_log.ch_names, picks_)
    _interpolate_bad_epochs(epochs, interp_channels, picks_, None, verbose)


def _apply_drop(reject_log: RejectLog, epochs, threshes_: Dict[str, float], picks_, verbose):
    _check_fit(epochs, threshes_, picks_)
    if np.any(reject_log.bad_epochs):
        if np.all(reject_log.bad_epochs):
            import warnings
            warnings.warn('All epochs were marked as bad by autoreject and will be dropped.')
        epochs.drop([int(i) for i in np.nonzero(reject_log.bad_epochs)[0]], reason='AUTOREJECT')
    elif verbose:
        print("No bad epochs were found for your data. Returning "
              "a copy of the data you wanted to clean. Interpolation "
              "may have been done.")
