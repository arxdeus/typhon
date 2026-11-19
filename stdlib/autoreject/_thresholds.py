# Copyright (C) 2026 the Typhon authors
# Ported from autoreject 0.5.2 (BSD-3-Clause)

"""Global and channel-level peak-to-peak threshold estimation.

Typhon notes: n_jobs is accepted but runs serially (identical results).
``random_state=None`` draws a seed from numpy's global RandomState (the
reference is irreproducible in that case too). A custom ``thresh_func``
is not supported (the default compute_thresholds is always used).
"""

import numpy as np
from numpy.random.legacy import check_random_state
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from .utils import _clean_by_interp, _handle_picks, _check_data, _get_picks_by_type, _verbose_bool
from .bayesopt import expected_improvement, bayes_opt


def _seed(random_state) -> int:
    if isinstance(random_state, NoneType):
        return int(np.random.randint(0, 2147483647))
    elif isinstance(random_state, Optional):
        if random_state is None:
            return int(np.random.randint(0, 2147483647))
        return int(random_state.__val__())
    else:
        return int(random_state)


def _as_cv(cv):
    if isinstance(cv, NoneType):
        return KFold(5)
    elif isinstance(cv, int):
        return KFold(cv)
    else:
        return cv


def _slicemean(X: np.ndarray[float, 3], idx: List[int]) -> np.ndarray[float, 2]:
    """np.mean(X[idx], axis=0), or all-NaN when idx is empty."""
    if len(idx) == 0:
        return np.full((X.shape[1], X.shape[2]), np.nan)
    return np.mean(X[np.array(idx, dtype=int)], axis=0)


def _slicemean2(X: np.ndarray[float, 2], idx: List[int]) -> np.ndarray[float, 1]:
    if len(idx) == 0:
        return np.full(X.shape[1], np.nan)
    return np.mean(X[np.array(idx, dtype=int)], axis=0)


def _rmse_score(med, mean_) -> float:
    if np.any(np.isnan(mean_)):
        return -np.inf
    return -np.sqrt(np.mean((med - mean_) ** 2))


class _GlobalAutoReject:
    """Global (all channels) peak-to-peak rejection estimator."""
    n_channels: Optional[int]
    n_times: Optional[int]
    thresh: float
    mean_: np.ndarray[float, 2]

    def __init__(self, n_channels: Optional[int] = None, n_times: Optional[int] = None,
                 thresh: float = 40e-6):
        self.thresh = thresh
        self.n_channels = n_channels
        self.n_times = n_times
        self.mean_ = np.empty((0, 0))

    def get_params(self, deep: bool = True) -> Dict[str, float]:
        return {"thresh": self.thresh}

    def set_params(self, thresh: float):
        self.thresh = thresh
        return self

    def _reshape(self, X) -> np.ndarray[float, 3]:
        Xa = np.asarray(X, dtype=float)
        if self.n_channels is None or self.n_times is None:
            raise ValueError('Cannot fit without knowing n_channels and n_times')
        nc = self.n_channels.__val__()
        nt = self.n_times.__val__()
        return Xa.reshape((Xa.size // (nc * nt), nc, nt))

    def fit(self, X, y = None):
        X3 = self._reshape(X)
        keep = List[int]()
        for i in range(X3.shape[0]):
            if np.max(np.ptp(X3[i], axis=1)) <= self.thresh:
                keep.append(i)
        self.mean_ = _slicemean(X3, keep)
        return self

    def score(self, X, y = None) -> float:
        X3 = self._reshape(X)
        return _rmse_score(np.median(X3, axis=0), self.mean_)


class _ChannelAutoReject:
    """Single channel peak-to-peak rejection estimator."""
    thresh: float
    deltas_: np.ndarray[float, 1]
    mean_: np.ndarray[float, 1]

    def __init__(self, thresh: float = 40e-6):
        self.thresh = thresh
        self.deltas_ = np.empty(0)
        self.mean_ = np.empty(0)

    def fit(self, X: np.ndarray[float, 2], y = None):
        deltas = np.ptp(X, axis=1)
        self.deltas_ = deltas
        lim = max(self.thresh, float(deltas.min()))
        keep = [i for i in range(deltas.shape[0]) if deltas[i] <= lim]
        self.mean_ = _slicemean2(X, keep)
        return self

    def score(self, X: np.ndarray[float, 2], y = None) -> float:
        return _rmse_score(np.median(X, axis=0), self.mean_)


def _fold_list(cv, X, y) -> List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]:
    out = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
    if isinstance(y, NoneType):
        for tr, te in cv.split(X):
            out.append((np.asarray(tr, dtype=int), np.asarray(te, dtype=int)))
    else:
        for tr, te in cv.split(X, y):
            out.append((np.asarray(tr, dtype=int), np.asarray(te, dtype=int)))
    return out


def validation_curve(epochs, y = None, param_name: str = "thresh", param_range = None,
                     cv = None, return_param_range: Literal[bool] = False, n_jobs: int = 1):
    """Validation curve on epochs for global autoreject.

    Returns (train_scores, test_scores[, param_range]) of shape
    (n_params, n_folds)."""
    data_picks = _handle_picks(epochs.info, None)
    X = epochs.get_data(data_picks)
    n_epochs, n_channels, n_times = X.shape
    pr = np.empty(0)
    if isinstance(param_range, NoneType):
        ptps = np.ptp(X, axis=2)
        pr = np.linspace(ptps.min(), ptps.max(), 15)
    else:
        pr = np.asarray(param_range, dtype=float)
    X2 = X.reshape((n_epochs, n_channels * n_times))
    folds = _fold_list(_as_cv(cv), X2, y)
    trs = np.empty((pr.shape[0], len(folds)))
    tes = np.empty((pr.shape[0], len(folds)))
    for fi in range(len(folds)):
        tr, te = folds[fi]
        for k in range(pr.shape[0]):
            est = _GlobalAutoReject(n_channels, n_times, float(pr[k]))
            est.fit(X2[tr])
            tes[k, fi] = est.score(X2[te])
            trs[k, fi] = est.score(X2[tr])
    if return_param_range:
        return trs, tes, pr
    else:
        return trs, tes


class _GlobalObjective:
    """-mean CV score of _GlobalAutoReject at the largest candidate <= thresh (cached)."""
    X: np.ndarray[float, 3]
    all_threshes: np.ndarray[float, 1]
    folds: List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]
    cache: Dict[float, float]

    def __init__(self, X: np.ndarray[float, 3], all_threshes: np.ndarray[float, 1], folds):
        self.X = X
        self.all_threshes = all_threshes
        self.folds = folds
        self.cache = Dict[float, float]()

    def __call__(self, thresh: float) -> float:
        idx = 0
        for i in range(self.all_threshes.shape[0]):
            if thresh - self.all_threshes[i] >= 0:
                idx = i
        t = float(self.all_threshes[idx])
        if t not in self.cache:
            sc = np.empty(len(self.folds))
            k = 0
            for tr, te in self.folds:
                est = _GlobalAutoReject(self.X.shape[1], self.X.shape[2], t)
                est.fit(self.X[tr])
                sc[k] = est.score(self.X[te])
                k += 1
            self.cache[t] = -np.mean(sc)
        return self.cache[t]


class _LossLookup:
    """Loss of the largest candidate threshold that is not above thresh."""
    losses: np.ndarray[float, 1]
    all_threshes: np.ndarray[float, 1]

    def __init__(self, losses: np.ndarray[float, 1], all_threshes: np.ndarray[float, 1]):
        self.losses = losses
        self.all_threshes = all_threshes

    def __call__(self, thresh: float) -> float:
        return float(self.losses[int(np.searchsorted(self.all_threshes, thresh, side='right')) - 1])


def _type_picks(info, ch_type: str) -> List[int]:
    return [i for i in range(info.nchan)
            if info.ch_types[i] == ch_type and info.ch_names[i] not in info.bads]


def get_rejection_threshold(epochs, decim: int = 1, random_state = None,
                            ch_types = None, cv = 5, verbose = True) -> Dict[str, float]:
    """Compute global rejection thresholds (one per channel type)."""
    reject = Dict[str, float]()
    types = List[str]()
    if isinstance(ch_types, NoneType):
        types = ['mag', 'grad', 'eeg', 'eog', 'hbo', 'hbr', 'ecog', 'seeg']
    elif isinstance(ch_types, str):
        types = [ch_types]
    else:
        types = [str(t) for t in ch_types]
    if decim > 1:
        epochs = epochs.copy()
        epochs.decimate(decim)
    cvo = _as_cv(cv)
    for ch_type in types:
        if ch_type not in epochs.info.ch_types:
            continue
        picks = _type_picks(epochs.info, ch_type)
        X = epochs.get_data(picks)
        n_epochs, n_channels, n_times = X.shape
        deltas = np.ptp(X, axis=2)
        all_threshes = np.sort(deltas.max(axis=1))
        if _verbose_bool(verbose):
            print(f'Estimating rejection dictionary for {ch_type}')
        func = _GlobalObjective(X, all_threshes, _fold_list(cvo, X, None))
        n = all_threshes.shape[0]
        idx = np.unique(np.concatenate((np.linspace(0, n, 5, endpoint=False).astype(int),
                                        np.array([n - 1], dtype=int))))
        best_thresh, _ = bayes_opt(func, all_threshes[idx], all_threshes,
                                   expected_improvement, max_iter=10, debug=False)
        reject[ch_type] = best_thresh
    return reject


def _thresh_cv_loss(this_data: np.ndarray[float, 2], folds, threshes: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    """Cross-validated loss of each channel-level threshold (cumulative sums)."""
    losses = np.empty((len(folds), threshes.shape[0]))
    fi = 0
    for train, test in folds:
        X = this_data[train]
        deltas = np.ptp(X, axis=1)
        order = np.argsort(deltas, kind='stable')
        deltas = deltas[order]
        means = np.cumsum(X[order], axis=0)
        means /= np.arange(1, means.shape[0] + 1).astype(float)[:, None]
        median = np.median(this_data[test], axis=0)
        loss_k = np.sqrt(np.mean((median[None, :] - means) ** 2, axis=1))
        loss_k[np.isnan(loss_k)] = np.inf
        n_keep = np.searchsorted(deltas, np.maximum(threshes, deltas[0]), side='right')
        for j in range(threshes.shape[0]):
            losses[fi, j] = loss_k[int(n_keep[j]) - 1]
        fi += 1
    return np.mean(losses, axis=0)


def _random_search_thresh(this_data: np.ndarray[float, 2], folds, lo: float, scale: float,
                          seed: int) -> float:
    """RandomizedSearchCV(_ChannelAutoReject, {thresh: uniform(lo, scale)},
    n_iter=20): best mean test score, first on ties."""
    rng = check_random_state(seed)
    best = -np.inf
    best_t = 0.0
    first = True
    for _ in range(20):
        t = lo + scale * float(rng.uniform(0.0, 1.0))
        sc = np.empty(len(folds))
        k = 0
        for tr, te in folds:
            est = _ChannelAutoReject(t)
            est.fit(this_data[tr])
            sc[k] = est.score(this_data[te])
            k += 1
        m = np.mean(sc)
        if first or m > best:
            best = m
            best_t = t
            first = False
    return best_t


def _compute_thresh(this_data: np.ndarray[float, 2], method: str = 'bayesian_optimization',
                    cv = 10, y = None, random_state = None) -> float:
    """Compute the rejection threshold for one channel."""
    all_threshes = np.sort(np.ptp(this_data, axis=1))
    folds = _fold_list(_as_cv(cv), this_data, y)
    if method == 'random_search':
        return _random_search_thresh(this_data, folds, float(all_threshes[0]),
                                     float(all_threshes[-1]), _seed(random_state))
    func = _LossLookup(_thresh_cv_loss(this_data, folds, all_threshes), all_threshes)
    n = all_threshes.shape[0]
    idx = np.unique(np.concatenate((np.linspace(0, n, 40, endpoint=False).astype(int),
                                    np.array([n - 1], dtype=int))))
    best_thresh, _ = bayes_opt(func, all_threshes[idx], all_threshes,
                               expected_improvement, max_iter=10, debug=False)
    return best_thresh


def compute_thresholds(epochs, method: str = 'bayesian_optimization', random_state = None,
                       picks = None, augment: bool = True, verbose = True,
                       n_jobs: int = 1) -> Dict[str, float]:
    """Compute the channel-level peak-to-peak thresholds."""
    return _compute_thresholds(epochs, method, random_state, picks, augment, None,
                               verbose, n_jobs)


def _compute_thresholds(epochs, method: str = 'bayesian_optimization', random_state = None,
                        picks = None, augment: bool = True, dots = None, verbose = True,
                        n_jobs: int = 1) -> Dict[str, float]:
    if method not in ('bayesian_optimization', 'random_search'):
        raise ValueError('`method` param not recognized')
    pk = _handle_picks(epochs.info, picks)
    _check_data(epochs, pk, 'data_channels', augment, verbose)
    by_type = _get_picks_by_type(epochs.info, pk)
    threshes = Dict[str, float]()
    seed = _seed(random_state)
    if len(by_type) > 1:
        for ch_type, this_picks in by_type:
            threshes.update(_compute_thresholds(epochs, method, seed, this_picks, augment,
                                                dots, verbose, n_jobs))
        return threshes
    n_epochs = len(epochs)
    data = epochs.get_data(copy=False)
    y = np.ones(n_epochs)
    if augment:
        epochs_interp = _clean_by_interp(epochs, pk, dots, verbose)
        data = np.concatenate((data, epochs_interp.get_data(copy=False)), axis=0)
        y = np.concatenate((np.zeros(n_epochs), np.ones(n_epochs)))
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
    ch_names = epochs.ch_names
    for p in pk:
        threshes[ch_names[int(p)]] = _compute_thresh(data[:, int(p), :].copy(), method, cv, y, seed)
    return threshes
