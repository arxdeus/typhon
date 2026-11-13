# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# SGDClassifier, SGDRegressor, SGDOneClassSVM (+ the shared BaseSGD logic used
# by Perceptron and PassiveAggressive*). Dense inputs only.
# random_state: None / int (sklearn's RandomState(seed) streams are reproduced).
# DEVIATIONS: class_weight is None, 'balanced' or a dict; partial_fit's
# `classes` must be given on the first call (as in sklearn).

import math
import numpy as np
from numpy.random import RandomState
from ._base import (_check_sw, _class_weight_vec, _LinearClassifierMixin, _expit, _r2)
from ._sgd_fast import (SGDLoss, plain_sgd, HINGE, SQUARED_HINGE, LOG, MODIFIED_HUBER,
                        SQUARED, HUBER, EPS_INSENSITIVE, SQ_EPS_INSENSITIVE)

MAX_INT = 2147483647
DEFAULT_EPSILON = 0.1

def _penalty_type(penalty) -> int:
    if penalty is None:
        return 0
    p = str(penalty).lower()
    if p == "none":
        return 0
    elif p == "l2":
        return 2
    elif p == "l1":
        return 1
    elif p == "elasticnet":
        return 3
    raise ValueError(f"Penalty {penalty} is not supported.")

def _lr_type(lr: str) -> int:
    if lr == "constant":
        return 1
    elif lr == "optimal":
        return 2
    elif lr == "invscaling":
        return 3
    elif lr == "adaptive":
        return 4
    elif lr == "pa1":
        return 5
    elif lr == "pa2":
        return 6
    raise ValueError(f"learning rate {lr} is not supported. ")

def _loss_fn(loss: str, epsilon: float, regressor: bool) -> SGDLoss:
    if loss == "hinge" and not regressor:
        return SGDLoss(HINGE, 1.0)
    elif loss == "squared_hinge" and not regressor:
        return SGDLoss(SQUARED_HINGE, 1.0)
    elif loss == "perceptron" and not regressor:
        return SGDLoss(HINGE, 0.0)
    elif loss == "log_loss" and not regressor:
        return SGDLoss(LOG)
    elif loss == "modified_huber" and not regressor:
        return SGDLoss(MODIFIED_HUBER)
    elif loss == "squared_error":
        return SGDLoss(SQUARED)
    elif loss == "huber":
        return SGDLoss(HUBER, epsilon)
    elif loss == "epsilon_insensitive":
        return SGDLoss(EPS_INSENSITIVE, epsilon)
    elif loss == "squared_epsilon_insensitive":
        return SGDLoss(SQ_EPS_INSENSITIVE, epsilon)
    raise ValueError(f"The loss {loss} is not supported. ")

class _Rng:
    """check_random_state(random_state) for None / int, shared across calls
    the way sklearn re-creates RandomState(int) at each use."""
    rs: RandomState
    def __init__(self, random_state: Optional[int]):
        if random_state is None:
            self.rs = np.random.RandomState(int(np.random.randint(0, MAX_INT)))
        else:
            self.rs = np.random.RandomState(random_state.__val__())

def _approximate_mode(counts, n_draws: int, rng):
    # sklearn.utils.extmath._approximate_mode with the legacy RandomState
    tot = float(np.sum(counts))
    cont = counts.astype(float) / tot * float(n_draws)
    floored = np.floor(cont)
    need = int(n_draws - floored.sum())
    if need > 0:
        rem = cont - floored
        vals = np.sort(np.unique(rem))[::-1]
        for v in vals:
            inds = np.nonzero(rem == v)[0]
            add_now = min(inds.shape[0], need)
            ch = rng.choice_noreplace(inds, add_now)
            for i in ch:
                floored[i] += 1
            need -= add_now
            if need == 0:
                break
    return floored.astype(int)

def _validation_mask(early_stopping: bool, is_clf: bool, y, sample_weight: np.ndarray[float, 1],
                     validation_fraction: float, random_state: Optional[int]) -> np.ndarray[bool, 1]:
    from sklearn.utils import _LegacyRS
    n = y.shape[0]
    mask = np.zeros(n, dtype=bool)
    if not early_stopping:
        return mask
    if random_state is None:
        raise ValueError("early_stopping in Typhon needs an integer random_state")
    rng = _LegacyRS(random_state.__val__())
    n_test = int(math.ceil(validation_fraction * float(n)))
    n_train = n - n_test
    idx_val = np.zeros(0, dtype=int)
    if is_clf:
        classes, y_ind, counts = np.unique(np.asarray(y), return_inverse=True, return_counts=True)
        nc = classes.shape[0]
        if int(np.min(counts)) < 2:
            raise ValueError("The least populated classes in y have only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.")
        if n_train < nc or n_test < nc:
            raise ValueError("The train_size / test_size should be greater or equal to the number of classes")
        order = np.argsort(y_ind, kind="stable")
        n_i = _approximate_mode(counts, n_train, rng)
        t_i = _approximate_mode(counts - n_i, n_test, rng)
        test = List[int]()
        start = 0
        for i in range(nc):
            ci = int(counts[i])
            cls = order[start:start + ci]
            pm = rng.permutation(ci)
            pic = cls[pm]
            for k in range(int(n_i[i]), int(n_i[i]) + int(t_i[i])):
                test.append(int(pic[k]))
            start += ci
        # train permutation is drawn first in sklearn
        tr_count = 0
        for i in range(nc):
            tr_count += int(n_i[i])
        rng.permutation(tr_count)
        idx_val = rng.permutation_of(np.array(test))
    else:
        perm = rng.permutation(n)
        idx_val = perm[:n_test].copy()
    any_pos = False
    for i in idx_val:
        if sample_weight[i] > 0.0:
            any_pos = True
    if not any_pos:
        raise ValueError("The sample weights for validation set are all zero, consider using a different random state.")
    if n_train == 0 or idx_val.shape[0] == 0:
        raise ValueError("Splitting samples into a train set and a validation set led to an empty set.")
    for i in idx_val:
        mask[i] = True
    return mask

class _ClfScoreCb:
    X: np.ndarray[float, 2]
    y: np.ndarray[float, 1]
    sw: np.ndarray[float, 1]
    def __init__(self, X: np.ndarray[float, 2], y: np.ndarray[float, 1], sw: np.ndarray[float, 1]):
        self.X = X
        self.y = y
        self.sw = sw
    def __call__(self, coef: np.ndarray[float, 1], intercept: float) -> float:
        # accuracy with classes [-1, 1] (weighted)
        num = 0.0
        den = 0.0
        for i in range(self.X.shape[0]):
            s = intercept
            for j in range(self.X.shape[1]):
                s += self.X[i, j] * coef[j]
            pred = 1.0 if s > 0.0 else -1.0
            den += self.sw[i]
            if pred == self.y[i]:
                num += self.sw[i]
        return num / den

class _RegScoreCb:
    X: np.ndarray[float, 2]
    y: np.ndarray[float, 1]
    sw: np.ndarray[float, 1]
    def __init__(self, X: np.ndarray[float, 2], y: np.ndarray[float, 1], sw: np.ndarray[float, 1]):
        self.X = X
        self.y = y
        self.sw = sw
    def __call__(self, coef: np.ndarray[float, 1], intercept: float) -> float:
        pred = self.X @ coef + intercept
        return _r2(self.y, pred, self.sw, True)

def _sel(X: np.ndarray[float, 2], mask: np.ndarray[bool, 1]) -> np.ndarray[float, 2]:
    return np.ascontiguousarray(X[mask])

class _SGDClassifierMixin(_LinearClassifierMixin):
    def _check_params(self, for_partial_fit: bool):
        if self.early_stopping and for_partial_fit:
            raise ValueError("early_stopping should be False with partial_fit")
        if self.learning_rate == "optimal" and self.alpha == 0.0:
            raise ValueError("alpha must be > 0 since learning_rate is 'optimal'. alpha is used to compute the optimal learning rate.")
        if self.learning_rate == "pa1" or self.learning_rate == "pa2":
            if self.loss != "hinge":
                raise ValueError(f"Learning rate '{self.learning_rate}' only works with loss 'hinge'.")
        _penalty_type(self.penalty)
        _lr_type(self.learning_rate)

    def _alloc(self, n_classes: int, n_features: int, coef_init, intercept_init):
        K = n_classes if n_classes > 2 else 1
        if coef_init is not None:
            ci = np.array(coef_init, dtype=float).reshape((K, n_features))
            self._coef = ci
        else:
            self._coef = np.zeros((K, n_features))
        if intercept_init is not None:
            self._intercept = np.array(intercept_init, dtype=float).reshape(K)
        else:
            self._intercept = np.zeros(K)
        if self.average > 0:
            self._standard_coef = self._coef
            self._average_coef = np.zeros((K, n_features))
            self._standard_intercept = self._intercept
            self._average_intercept = np.zeros(K)

    def _fit_binary_one(self, i: int, X: np.ndarray[float, 2], y, alpha: float,
                        learning_rate: str, max_iter: int, pos_weight: float, neg_weight: float,
                        sw: np.ndarray[float, 1], vmask: Optional[np.ndarray[bool, 1]],
                        rs: RandomState, loss: SGDLoss):
        n = X.shape[0]
        label_encode = loss.kind == LOG
        yi = np.ones(n)
        for s in range(n):
            if y[s] != self.classes_[i]:
                yi[s] = 0.0 if label_encode else -1.0
        binary = self.classes_.shape[0] == 2
        row = 0 if binary else i
        coef_arr = self._standard_coef if self.average > 0 else self._coef
        int_arr = self._standard_intercept if self.average > 0 else self._intercept
        coef = np.ascontiguousarray(coef_arr[row]).copy()
        intercept = float(int_arr[row])
        avg = np.zeros(1)
        avg_int = 0.0
        if self.average > 0:
            avg = np.ascontiguousarray(self._average_coef[row]).copy()
            avg_int = float(self._average_intercept[row])
        rs.randint(1, MAX_INT)  # make_dataset seed (only used by dataset.random)
        mask = np.zeros(n, dtype=bool)
        if vmask is None:
            pos = np.zeros(n, dtype=bool)
            mask = _validation_mask(self.early_stopping, True, yi, sw, self.validation_fraction,
                                    self.random_state)
        else:
            mask = vmask.__val__()
        cb = _ClfScoreCb(_sel(X, mask), yi[mask], sw[mask])
        seed = int(rs.randint(MAX_INT))
        tol = -math.inf if self.tol is None else self.tol.__val__()
        res = plain_sgd(coef, intercept, avg, self.average > 0, avg_int, loss,
                        _penalty_type(self.penalty), alpha, self._l1(), X, yi, sw, mask,
                        self.early_stopping, cb, self.n_iter_no_change, max_iter, tol,
                        self.fit_intercept, self.shuffle, seed, pos_weight, neg_weight,
                        _lr_type(learning_rate), self.eta0, self.power_t, False, self.t_, 1.0,
                        self.average)
        coef_arr[row] = coef
        if self.average > 0:
            self._average_coef[row] = avg
            self._average_intercept[row] = res.average_intercept
        return res.intercept, res.n_iter

    def _l1(self) -> float:
        return self.l1_ratio

    def _partial_fit_impl(self, X, y, alpha: float, learning_rate: str, max_iter: int,
                          classes, sample_weight, coef_init, intercept_init, loss: SGDLoss):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y)
        n, p = Xa.shape
        if not self._fitted:
            if classes is None:
                raise ValueError("classes must be passed on the first call to partial_fit.")
            self.classes_ = np.unique(np.asarray(classes))
        K = self.classes_.shape[0]
        yi = np.searchsorted(self.classes_, ya)
        self._expanded_class_weight = _class_weight_vec(self.class_weight, self.classes_, yi)
        sw = _check_sw(sample_weight, n)
        if (not self._has_coef) or coef_init is not None:
            self._alloc(K, p, coef_init, intercept_init)
            self._has_coef = True
        elif p != self._coef.shape[1]:
            raise ValueError(f"Number of features {p} does not match previous data {self._coef.shape[1]}.")
        self.n_features_in_ = p
        self._fitted = True
        if K > 2:
            vmask = _validation_mask(self.early_stopping, True, ya, sw, self.validation_fraction,
                                     self.random_state)
            rs0 = _Rng(self.random_state).rs
            seeds = rs0.randint(MAX_INT, size=K)
            n_iter = 0
            for i in range(K):
                rsi = np.random.RandomState(int(seeds[i]))
                ic, it = self._fit_binary_one(i, Xa, ya, alpha, learning_rate, max_iter,
                                              self._expanded_class_weight[i], 1.0, sw, vmask,
                                              rsi, loss)
                self._intercept[i] = ic
                n_iter = max(n_iter, it)
            self.t_ += float(n_iter * n)
            self.n_iter_ = n_iter
            if self.average > 0:
                if float(self.average) <= self.t_ - 1.0:
                    self._coef = self._average_coef
                    self._intercept = self._average_intercept
                else:
                    self._coef = self._standard_coef
                    self._standard_intercept = self._intercept
        elif K == 2:
            rs1 = _Rng(self.random_state).rs
            ic, it = self._fit_binary_one(1, Xa, ya, alpha, learning_rate, max_iter,
                                          self._expanded_class_weight[1],
                                          self._expanded_class_weight[0], sw, None, rs1, loss)
            self.t_ += float(it * n)
            self.n_iter_ = it
            if self.average > 0:
                if float(self.average) <= self.t_ - 1.0:
                    self._coef = self._average_coef
                    self._intercept = self._average_intercept
                else:
                    self._coef = self._standard_coef
                    self._standard_intercept = np.array([ic])
                    self._intercept = self._standard_intercept
            else:
                self._intercept = np.array([ic])
        else:
            raise ValueError(f"The number of classes has to be greater than one; got {K} class")
        return self

    def _fit_impl(self, X, y, alpha: float, learning_rate: str, coef_init, intercept_init,
                  sample_weight, loss: SGDLoss):
        self._check_params(False)
        self._fitted = False
        classes = np.unique(np.asarray(y))
        ci = coef_init
        ii = intercept_init
        if self.warm_start and self._has_coef:
            if coef_init is None and intercept_init is None:
                return self._fit_warm(X, y, alpha, learning_rate, classes, sample_weight, loss)
        else:
            self._has_coef = False
        self.t_ = 1.0
        return self._partial_fit_impl(X, y, alpha, learning_rate, self.max_iter, classes,
                                      sample_weight, ci, ii, loss)

    def _fit_warm(self, X, y, alpha: float, learning_rate: str, classes, sample_weight,
                  loss: SGDLoss):
        ci = self._coef.copy()
        ii = self._intercept.copy()
        self.t_ = 1.0
        return self._partial_fit_impl(X, y, alpha, learning_rate, self.max_iter, classes,
                                      sample_weight, ci, ii, loss)

    def _proba(self, X):
        if self.loss == "log_loss":
            return self._predict_proba_lr(X)
        elif self.loss == "modified_huber":
            s = self._scores(X)
            n = s.shape[0]
            if s.shape[1] == 1:
                pr = np.ones((n, 2))
                for i in range(n):
                    v = min(max(s[i, 0], -1.0), 1.0)
                    v = (v + 1.0) / 2.0
                    pr[i, 1] = v
                    pr[i, 0] -= v
                return pr
            else:
                K = s.shape[1]
                pr = np.empty((n, K))
                for i in range(n):
                    tot = 0.0
                    for k in range(K):
                        pr[i, k] = (min(max(s[i, k], -1.0), 1.0) + 1.0) / 2.0
                        tot += pr[i, k]
                    if tot == 0.0:
                        for k in range(K):
                            pr[i, k] = 1.0 / float(K)
                    else:
                        for k in range(K):
                            pr[i, k] /= tot
                return pr
        else:
            raise NotImplementedError(f"predict_(log_)proba only supported when loss='log_loss' or loss='modified_huber' ({self.loss!r} given)")

class SGDClassifier[CW = NoneType](_SGDClassifierMixin):
    loss: str
    penalty: Optional[str]
    alpha: float
    l1_ratio: float
    fit_intercept: bool
    max_iter: int
    tol: Optional[float]
    shuffle: bool
    verbose: int
    epsilon: float
    random_state: Optional[int]
    learning_rate: str
    eta0: float
    power_t: float
    early_stopping: bool
    validation_fraction: float
    n_iter_no_change: int
    class_weight: CW
    warm_start: bool
    average: int
    classes_: np.ndarray[int, 1]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _standard_coef: np.ndarray[float, 2]
    _standard_intercept: np.ndarray[float, 1]
    _average_coef: np.ndarray[float, 2]
    _average_intercept: np.ndarray[float, 1]
    _expanded_class_weight: np.ndarray[float, 1]
    t_: float
    n_iter_: int
    _fitted: bool
    _has_coef: bool
    n_features_in_: int

    def __init__(self, loss: str = "hinge", *, penalty: Optional[str] = "l2", alpha: float = 0.0001,
                 l1_ratio: float = 0.15, fit_intercept: bool = True, max_iter: int = 1000,
                 tol: Optional[float] = 1e-3, shuffle: bool = True, verbose: int = 0,
                 epsilon: float = DEFAULT_EPSILON, n_jobs = None, random_state: Optional[int] = None,
                 learning_rate: str = "optimal", eta0: float = 0.01, power_t: float = 0.5,
                 early_stopping: bool = False, validation_fraction: float = 0.1,
                 n_iter_no_change: int = 5, class_weight: CW = None, warm_start: bool = False,
                 average = False):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = int(average)
        self.classes_ = np.zeros(0, dtype=int)
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._standard_coef = np.zeros((0, 0))
        self._standard_intercept = np.zeros(0)
        self._average_coef = np.zeros((0, 0))
        self._average_intercept = np.zeros(0)
        self._expanded_class_weight = np.zeros(0)
        self.t_ = 1.0
        self.n_iter_ = 0
        self._fitted = False
        self._has_coef = False
        self.n_features_in_ = 0

    def _lossf(self) -> SGDLoss:
        return _loss_fn(self.loss, self.epsilon, False)

    def fit(self, X, y, coef_init = None, intercept_init = None, sample_weight = None):
        return self._fit_impl(X, y, self.alpha, self.learning_rate, coef_init, intercept_init,
                              sample_weight, self._lossf())

    def partial_fit(self, X, y, classes = None, sample_weight = None):
        if not self._fitted:
            self._check_params(True)
            if isinstance(self.class_weight, str):
                raise ValueError("class_weight 'balanced' is not supported for partial_fit.")
        return self._partial_fit_impl(X, y, self.alpha, self.learning_rate, 1, classes,
                                      sample_weight, None, None, self._lossf())

    def predict_proba(self, X):
        return self._proba(X)

    def predict_log_proba(self, X):
        return np.log(self._proba(X))

class Perceptron[CW = NoneType](_SGDClassifierMixin):
    loss: str
    penalty: Optional[str]
    alpha: float
    l1_ratio: float
    fit_intercept: bool
    max_iter: int
    tol: Optional[float]
    shuffle: bool
    verbose: int
    epsilon: float
    random_state: Optional[int]
    learning_rate: str
    eta0: float
    power_t: float
    early_stopping: bool
    validation_fraction: float
    n_iter_no_change: int
    class_weight: CW
    warm_start: bool
    average: int
    classes_: np.ndarray[int, 1]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _standard_coef: np.ndarray[float, 2]
    _standard_intercept: np.ndarray[float, 1]
    _average_coef: np.ndarray[float, 2]
    _average_intercept: np.ndarray[float, 1]
    _expanded_class_weight: np.ndarray[float, 1]
    t_: float
    n_iter_: int
    _fitted: bool
    _has_coef: bool
    n_features_in_: int

    def __init__(self, *, penalty: Optional[str] = None, alpha: float = 0.0001,
                 l1_ratio: float = 0.15, fit_intercept: bool = True, max_iter: int = 1000,
                 tol: Optional[float] = 1e-3, shuffle: bool = True, verbose: int = 0,
                 eta0: float = 1.0, n_jobs = None, random_state: Optional[int] = 0,
                 early_stopping: bool = False, validation_fraction: float = 0.1,
                 n_iter_no_change: int = 5, class_weight: CW = None, warm_start: bool = False):
        self.loss = "perceptron"
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = DEFAULT_EPSILON
        self.random_state = random_state
        self.learning_rate = "constant"
        self.eta0 = eta0
        self.power_t = 0.5
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        average = False
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = int(average)
        self.classes_ = np.zeros(0, dtype=int)
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._standard_coef = np.zeros((0, 0))
        self._standard_intercept = np.zeros(0)
        self._average_coef = np.zeros((0, 0))
        self._average_intercept = np.zeros(0)
        self._expanded_class_weight = np.zeros(0)
        self.t_ = 1.0
        self.n_iter_ = 0
        self._fitted = False
        self._has_coef = False
        self.n_features_in_ = 0

    def _lossf(self) -> SGDLoss:
        return SGDLoss(HINGE, 0.0)

    def fit(self, X, y, coef_init = None, intercept_init = None, sample_weight = None):
        return self._fit_impl(X, y, self.alpha, self.learning_rate, coef_init, intercept_init,
                              sample_weight, self._lossf())

    def partial_fit(self, X, y, classes = None, sample_weight = None):
        if not self._fitted:
            self._check_params(True)
        return self._partial_fit_impl(X, y, self.alpha, self.learning_rate, 1, classes,
                                      sample_weight, None, None, self._lossf())

class PassiveAggressiveClassifier[CW = NoneType](_SGDClassifierMixin):
    loss: str
    penalty: Optional[str]
    alpha: float
    l1_ratio: float
    fit_intercept: bool
    max_iter: int
    tol: Optional[float]
    shuffle: bool
    verbose: int
    epsilon: float
    random_state: Optional[int]
    learning_rate: str
    eta0: float
    power_t: float
    early_stopping: bool
    validation_fraction: float
    n_iter_no_change: int
    class_weight: CW
    warm_start: bool
    average: int
    classes_: np.ndarray[int, 1]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _standard_coef: np.ndarray[float, 2]
    _standard_intercept: np.ndarray[float, 1]
    _average_coef: np.ndarray[float, 2]
    _average_intercept: np.ndarray[float, 1]
    _expanded_class_weight: np.ndarray[float, 1]
    t_: float
    n_iter_: int
    _fitted: bool
    _has_coef: bool
    n_features_in_: int
    C: float
    pa_loss: str

    def __init__(self, *, C: float = 1.0, fit_intercept: bool = True, max_iter: int = 1000,
                 tol: Optional[float] = 1e-3, early_stopping: bool = False,
                 validation_fraction: float = 0.1, n_iter_no_change: int = 5,
                 shuffle: bool = True, verbose: int = 0, loss: str = "hinge", n_jobs = None,
                 random_state: Optional[int] = None, warm_start: bool = False,
                 class_weight: CW = None, average = False):
        self.loss = "hinge"
        self.penalty = None
        self.alpha = 0.0001
        self.l1_ratio = 0.15
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = DEFAULT_EPSILON
        self.random_state = random_state
        self.learning_rate = "pa1" if loss == "hinge" else "pa2"
        self.eta0 = C
        self.power_t = 0.5
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = int(average)
        self.classes_ = np.zeros(0, dtype=int)
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._standard_coef = np.zeros((0, 0))
        self._standard_intercept = np.zeros(0)
        self._average_coef = np.zeros((0, 0))
        self._average_intercept = np.zeros(0)
        self._expanded_class_weight = np.zeros(0)
        self.t_ = 1.0
        self.n_iter_ = 0
        self._fitted = False
        self._has_coef = False
        self.n_features_in_ = 0
        self.C = C
        self.pa_loss = loss

    def _lossf(self) -> SGDLoss:
        return SGDLoss(HINGE, 1.0)

    def fit(self, X, y, coef_init = None, intercept_init = None):
        self.eta0 = self.C
        self.learning_rate = "pa1" if self.pa_loss == "hinge" else "pa2"
        return self._fit_impl(X, y, 1.0, self.learning_rate, coef_init, intercept_init, None,
                              self._lossf())

    def partial_fit(self, X, y, classes = None):
        if not self._fitted:
            self._check_params(True)
            if isinstance(self.class_weight, str):
                raise ValueError("class_weight 'balanced' is not supported for partial_fit.")
        self.eta0 = self.C
        self.learning_rate = "pa1" if self.pa_loss == "hinge" else "pa2"
        return self._partial_fit_impl(X, y, 1.0, self.learning_rate, 1, classes, None, None,
                                      None, self._lossf())

class _SGDRegressorMixin:
    @property
    def coef_(self):
        return self._coef.copy()

    @property
    def intercept_(self):
        return self._intercept.copy()

    def _check_params(self, for_partial_fit: bool):
        if self.early_stopping and for_partial_fit:
            raise ValueError("early_stopping should be False with partial_fit")
        if self.learning_rate == "optimal" and self.alpha == 0.0:
            raise ValueError("alpha must be > 0 since learning_rate is 'optimal'. alpha is used to compute the optimal learning rate.")
        if self.learning_rate == "pa1" or self.learning_rate == "pa2":
            if self.loss != "epsilon_insensitive":
                raise ValueError(f"Learning rate '{self.learning_rate}' only works with loss 'epsilon_insensitive'.")
        _penalty_type(self.penalty)
        _lr_type(self.learning_rate)

    def _partial_fit_impl(self, X, y, alpha: float, learning_rate: str, max_iter: int,
                          sample_weight, coef_init, intercept_init, loss: SGDLoss):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
        n, p = Xa.shape
        sw = _check_sw(sample_weight, n)
        if not self._has_coef:
            self._coef = np.zeros(p) if coef_init is None else np.array(coef_init, dtype=float).ravel()
            self._intercept = np.zeros(1) if intercept_init is None else np.array(intercept_init, dtype=float).reshape(1)
            if self.average > 0:
                self._standard_coef = self._coef
                self._standard_intercept = self._intercept
                self._average_coef = np.zeros(p)
                self._average_intercept = np.zeros(1)
            self._has_coef = True
        self.n_features_in_ = p
        if self.average > 0 and self._average_coef.shape[0] != p:
            self._average_coef = np.zeros(p)
            self._average_intercept = np.zeros(1)
        mask = _validation_mask(self.early_stopping, False, ya, sw, self.validation_fraction,
                                self.random_state)
        cb = _RegScoreCb(_sel(Xa, mask), ya[mask], sw[mask])
        rs = _Rng(self.random_state).rs
        seed = int(rs.randint(0, MAX_INT))
        rs.randint(1, MAX_INT)  # make_dataset
        tol = -math.inf if self.tol is None else self.tol.__val__()
        coef = (self._standard_coef if self.average > 0 else self._coef).copy()
        intercept = float((self._standard_intercept if self.average > 0 else self._intercept)[0])
        avg = self._average_coef.copy() if self.average > 0 else np.zeros(1)
        avg_int = float(self._average_intercept[0]) if self.average > 0 else 0.0
        res = plain_sgd(coef, intercept, avg, self.average > 0, avg_int, loss,
                        _penalty_type(self.penalty), alpha, self.l1_ratio, Xa, ya, sw, mask,
                        self.early_stopping, cb, self.n_iter_no_change, max_iter, tol,
                        self.fit_intercept, self.shuffle, seed, 1.0, 1.0,
                        _lr_type(learning_rate), self.eta0, self.power_t, False, self.t_, 1.0,
                        self.average)
        self.n_iter_ = res.n_iter
        self.t_ += float(res.n_iter * n)
        if self.average > 0:
            self._standard_coef = coef
            self._average_coef = avg
            self._average_intercept = np.array([res.average_intercept])
            self._standard_intercept = np.array([res.intercept])
            if float(self.average) <= self.t_ - 1.0:
                self._coef = avg
                self._intercept = np.array([res.average_intercept])
            else:
                self._coef = coef
                self._intercept = np.array([res.intercept])
        else:
            self._coef = coef
            self._intercept = np.array([res.intercept])
        return self

    def _fit_impl(self, X, y, alpha: float, learning_rate: str, coef_init, intercept_init,
                  sample_weight, loss: SGDLoss):
        self._check_params(False)
        if self.warm_start and self._has_coef:
            if coef_init is None and intercept_init is None:
                ci = self._coef.copy()
                ii = self._intercept.copy()
                self._has_coef = False
                self.t_ = 1.0
                return self._partial_fit_impl(X, y, alpha, learning_rate, self.max_iter,
                                              sample_weight, ci, ii, loss)
        self._has_coef = False
        self.t_ = 1.0
        return self._partial_fit_impl(X, y, alpha, learning_rate, self.max_iter, sample_weight,
                                      coef_init, intercept_init, loss)

    def predict(self, X):
        Xa = np.asarray(X, dtype=float)
        return Xa @ self._coef + self._intercept[0]

    def score(self, X, y, sample_weight = None) -> float:
        n = np.asarray(X).shape[0]
        return _r2(y, self.predict(X), _check_sw(sample_weight, n), sample_weight is not None)

class SGDRegressor(_SGDRegressorMixin):
    loss: str
    penalty: Optional[str]
    alpha: float
    l1_ratio: float
    fit_intercept: bool
    max_iter: int
    tol: Optional[float]
    shuffle: bool
    verbose: int
    epsilon: float
    random_state: Optional[int]
    learning_rate: str
    eta0: float
    power_t: float
    early_stopping: bool
    validation_fraction: float
    n_iter_no_change: int
    warm_start: bool
    average: int
    _coef: np.ndarray[float, 1]
    _intercept: np.ndarray[float, 1]
    _standard_coef: np.ndarray[float, 1]
    _standard_intercept: np.ndarray[float, 1]
    _average_coef: np.ndarray[float, 1]
    _average_intercept: np.ndarray[float, 1]
    t_: float
    n_iter_: int
    _has_coef: bool
    n_features_in_: int

    def __init__(self, loss: str = "squared_error", *, penalty: Optional[str] = "l2",
                 alpha: float = 0.0001, l1_ratio: float = 0.15, fit_intercept: bool = True,
                 max_iter: int = 1000, tol: Optional[float] = 1e-3, shuffle: bool = True,
                 verbose: int = 0, epsilon: float = DEFAULT_EPSILON,
                 random_state: Optional[int] = None, learning_rate: str = "invscaling",
                 eta0: float = 0.01, power_t: float = 0.25, early_stopping: bool = False,
                 validation_fraction: float = 0.1, n_iter_no_change: int = 5,
                 warm_start: bool = False, average = False):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = int(average)
        self._coef = np.zeros(0)
        self._intercept = np.zeros(0)
        self._standard_coef = np.zeros(0)
        self._standard_intercept = np.zeros(0)
        self._average_coef = np.zeros(0)
        self._average_intercept = np.zeros(0)
        self.t_ = 1.0
        self.n_iter_ = 0
        self._has_coef = False
        self.n_features_in_ = 0

    def fit(self, X, y, coef_init = None, intercept_init = None, sample_weight = None):
        return self._fit_impl(X, y, self.alpha, self.learning_rate, coef_init, intercept_init,
                              sample_weight, _loss_fn(self.loss, self.epsilon, True))

    def partial_fit(self, X, y, sample_weight = None):
        if not self._has_coef:
            self._check_params(True)
        return self._partial_fit_impl(X, y, self.alpha, self.learning_rate, 1, sample_weight,
                                      None, None, _loss_fn(self.loss, self.epsilon, True))

class PassiveAggressiveRegressor(_SGDRegressorMixin):
    loss: str
    penalty: Optional[str]
    alpha: float
    l1_ratio: float
    fit_intercept: bool
    max_iter: int
    tol: Optional[float]
    shuffle: bool
    verbose: int
    epsilon: float
    random_state: Optional[int]
    learning_rate: str
    eta0: float
    power_t: float
    early_stopping: bool
    validation_fraction: float
    n_iter_no_change: int
    warm_start: bool
    average: int
    _coef: np.ndarray[float, 1]
    _intercept: np.ndarray[float, 1]
    _standard_coef: np.ndarray[float, 1]
    _standard_intercept: np.ndarray[float, 1]
    _average_coef: np.ndarray[float, 1]
    _average_intercept: np.ndarray[float, 1]
    t_: float
    n_iter_: int
    _has_coef: bool
    n_features_in_: int
    C: float
    pa_loss: str

    def __init__(self, *, C: float = 1.0, fit_intercept: bool = True, max_iter: int = 1000,
                 tol: Optional[float] = 1e-3, early_stopping: bool = False,
                 validation_fraction: float = 0.1, n_iter_no_change: int = 5,
                 shuffle: bool = True, verbose: int = 0, loss: str = "epsilon_insensitive",
                 epsilon: float = DEFAULT_EPSILON, random_state: Optional[int] = None,
                 warm_start: bool = False, average = False):
        self.loss = "epsilon_insensitive"
        self.penalty = None
        self.alpha = 0.0001
        self.l1_ratio = 0.15
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = "pa1" if loss == "epsilon_insensitive" else "pa2"
        self.eta0 = C
        self.power_t = 0.25
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = int(average)
        self._coef = np.zeros(0)
        self._intercept = np.zeros(0)
        self._standard_coef = np.zeros(0)
        self._standard_intercept = np.zeros(0)
        self._average_coef = np.zeros(0)
        self._average_intercept = np.zeros(0)
        self.t_ = 1.0
        self.n_iter_ = 0
        self._has_coef = False
        self.n_features_in_ = 0
        self.C = C
        self.pa_loss = loss

    def fit(self, X, y, coef_init = None, intercept_init = None):
        self.eta0 = self.C
        self.learning_rate = "pa1" if self.pa_loss == "epsilon_insensitive" else "pa2"
        return self._fit_impl(X, y, 1.0, self.learning_rate, coef_init, intercept_init, None,
                              SGDLoss(EPS_INSENSITIVE, self.epsilon))

    def partial_fit(self, X, y):
        if not self._has_coef:
            self._check_params(True)
        self.eta0 = self.C
        self.learning_rate = "pa1" if self.pa_loss == "epsilon_insensitive" else "pa2"
        return self._partial_fit_impl(X, y, 1.0, self.learning_rate, 1, None, None, None,
                                      SGDLoss(EPS_INSENSITIVE, self.epsilon))

class SGDOneClassSVM:
    nu: float
    fit_intercept: bool
    max_iter: int
    tol: Optional[float]
    shuffle: bool
    verbose: int
    random_state: Optional[int]
    learning_rate: str
    eta0: float
    power_t: float
    warm_start: bool
    average: int
    _coef: np.ndarray[float, 1]
    _offset: np.ndarray[float, 1]
    _standard_coef: np.ndarray[float, 1]
    _standard_intercept: np.ndarray[float, 1]
    _average_coef: np.ndarray[float, 1]
    _average_intercept: np.ndarray[float, 1]
    t_: float
    n_iter_: int
    _has_coef: bool
    n_features_in_: int

    def __init__(self, nu: float = 0.5, fit_intercept: bool = True, max_iter: int = 1000,
                 tol: Optional[float] = 1e-3, shuffle: bool = True, verbose: int = 0,
                 random_state: Optional[int] = None, learning_rate: str = "optimal",
                 eta0: float = 0.01, power_t: float = 0.5, warm_start: bool = False,
                 average = False):
        self.nu = nu
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.average = int(average)
        self._coef = np.zeros(0)
        self._offset = np.zeros(0)
        self._standard_coef = np.zeros(0)
        self._standard_intercept = np.zeros(0)
        self._average_coef = np.zeros(0)
        self._average_intercept = np.zeros(0)
        self.t_ = 1.0
        self.n_iter_ = 0
        self._has_coef = False
        self.n_features_in_ = 0

    @property
    def coef_(self):
        return self._coef.copy()

    @property
    def offset_(self):
        return self._offset.copy()

    def _partial_fit_impl(self, X, max_iter: int, sample_weight, coef_init, offset_init):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        n, p = Xa.shape
        sw = _check_sw(sample_weight, n)
        if (not self._has_coef) or coef_init is not None:
            self._coef = np.zeros(p) if coef_init is None else np.array(coef_init, dtype=float).ravel()
            self._offset = np.zeros(1) if offset_init is None else np.array(offset_init, dtype=float).reshape(1)
            if self.average > 0:
                self._standard_coef = self._coef
                self._standard_intercept = 1.0 - self._offset
                self._average_coef = np.zeros(p)
                self._average_intercept = np.zeros(1)
            self._has_coef = True
        elif p != self._coef.shape[0]:
            raise ValueError(f"Number of features {p} does not match previous data {self._coef.shape[0]}.")
        self.n_features_in_ = p
        if self.average > 0 and self._average_coef.shape[0] != p:
            self._average_coef = np.zeros(p)
            self._average_intercept = np.zeros(1)
        y = np.ones(n)
        np.random.randint(1, MAX_INT)  # make_dataset(X, y, sw) uses the global RNG
        mask = np.zeros(n, dtype=bool)
        rs = _Rng(self.random_state).rs
        seed = int(rs.randint(0, MAX_INT))
        tol = -math.inf if self.tol is None else self.tol.__val__()
        coef = (self._standard_coef if self.average > 0 else self._coef).copy()
        intercept = float(self._standard_intercept[0]) if self.average > 0 else 1.0 - float(self._offset[0])
        avg = self._average_coef.copy() if self.average > 0 else np.zeros(1)
        avg_int = float(self._average_intercept[0]) if self.average > 0 else 0.0
        cb = _RegScoreCb(np.zeros((0, p)), np.zeros(0), np.zeros(0))
        res = plain_sgd(coef, intercept, avg, self.average > 0, avg_int, SGDLoss(HINGE, 1.0),
                        2, self.nu, 0.0, Xa, y, sw, mask, False, cb, 5, max_iter, tol,
                        self.fit_intercept, self.shuffle, seed, 1.0, 1.0,
                        _lr_type(self.learning_rate), self.eta0, self.power_t, True, self.t_, 1.0,
                        self.average)
        self.n_iter_ = res.n_iter
        self.t_ += float(res.n_iter * n)
        if self.average > 0:
            self._standard_coef = coef
            self._average_coef = avg
            self._average_intercept = np.array([res.average_intercept])
            self._standard_intercept = np.array([res.intercept])
            if float(self.average) <= self.t_ - 1.0:
                self._coef = avg
                self._offset = 1.0 - np.array([res.average_intercept])
            else:
                self._coef = coef
                self._offset = 1.0 - np.array([res.intercept])
        else:
            self._coef = coef
            self._offset = 1.0 - np.array([res.intercept])
        return self

    def fit(self, X, y = None, coef_init = None, offset_init = None, sample_weight = None):
        if self.learning_rate == "optimal" and self.nu == 0.0:
            raise ValueError("alpha must be > 0 since learning_rate is 'optimal'.")
        if self.warm_start and self._has_coef and coef_init is None and offset_init is None:
            ci = self._coef.copy()
            oi = self._offset.copy()
            self.t_ = 1.0
            return self._partial_fit_impl(X, self.max_iter, sample_weight, ci, oi)
        self._has_coef = False
        self.t_ = 1.0
        return self._partial_fit_impl(X, self.max_iter, sample_weight, coef_init, offset_init)

    def partial_fit(self, X, y = None, sample_weight = None):
        return self._partial_fit_impl(X, 1, sample_weight, None, None)

    def decision_function(self, X):
        Xa = np.asarray(X, dtype=float)
        return Xa @ self._coef - self._offset[0]

    def score_samples(self, X):
        return self.decision_function(X) + self._offset[0]

    def predict(self, X):
        d = self.decision_function(X)
        out = np.empty(d.shape[0], dtype=int)
        for i in range(d.shape[0]):
            out[i] = 1 if d[i] >= 0.0 else -1
        return out

    def fit_predict(self, X, y = None):
        return self.fit(X).predict(X)
