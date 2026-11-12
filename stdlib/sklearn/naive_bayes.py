# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Naive Bayes: GaussianNB, MultinomialNB, ComplementNB, BernoulliNB,
# CategoricalNB. Class labels are integers.

import numpy as np

def _logsumexp_rows(a: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
    n = a.shape[0]
    out = np.empty(n)
    for i in range(n):
        m = -float("inf")
        for j in range(a.shape[1]):
            if a[i, j] > m:
                m = a[i, j]
        if m == -float("inf") or m == float("inf"):
            out[i] = m
            continue
        s = 0.0
        for j in range(a.shape[1]):
            s += np.exp(a[i, j] - m)
        out[i] = m + np.log(s)
    return out

def _acc(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=int).ravel()
    b = np.asarray(y_pred, dtype=int).ravel()
    c = 0
    for i in range(a.shape[0]):
        if a[i] == b[i]:
            c += 1
    return float(c) / float(a.shape[0]) if a.shape[0] > 0 else 0.0

def _as2d(X) -> np.ndarray[float, 2]:
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        return np.ascontiguousarray(a.reshape((1, -1)))
    return np.ascontiguousarray(a)

def _class_pos(classes: np.ndarray[int, 1], v: int) -> int:
    i = int(np.searchsorted(classes, v))
    if i >= classes.shape[0] or classes[i] != v:
        raise ValueError(f"The target label(s) [{v}] in y do not exist in the initial classes {classes}")
    return i

def _sw(sample_weight, n: int) -> np.ndarray[float, 1]:
    if sample_weight is None:
        return np.ones(n)
    w = np.ascontiguousarray(np.asarray(sample_weight, dtype=float).ravel())
    if w.shape[0] != n:
        raise ValueError("sample_weight.shape must match X")
    return w

class _NBPredict:
    """Mixin: needs _joint_log_likelihood(X) and classes_."""
    def predict_joint_log_proba(self, X):
        return self._joint_log_likelihood(self._prep_X(X))

    def predict(self, X):
        jll = self._joint_log_likelihood(self._prep_X(X))
        idx = np.argmax(jll, axis=1)
        out = np.empty(jll.shape[0], dtype=int)
        for i in range(jll.shape[0]):
            out[i] = self.classes_[idx[i]]
        return out

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(self._prep_X(X))
        lse = _logsumexp_rows(jll)
        return jll - lse.reshape((-1, 1))

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def score(self, X, y, sample_weight = None):
        return _acc(y, self.predict(X))

class GaussianNB(_NBPredict):
    priors: Optional[List[float]]
    var_smoothing: float
    classes_: np.ndarray[int, 1]
    theta_: np.ndarray[float, 2]
    var_: np.ndarray[float, 2]
    class_count_: np.ndarray[float, 1]
    class_prior_: np.ndarray[float, 1]
    epsilon_: float
    n_features_in_: int
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, *, priors: Optional[List[float]] = None, var_smoothing: float = 1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.classes_ = np.empty(0, dtype=int)
        self.theta_ = np.empty((0, 0))
        self.var_ = np.empty((0, 0))
        self.class_count_ = np.empty(0)
        self.class_prior_ = np.empty(0)
        self.epsilon_ = 0.0
        self.n_features_in_ = 0

    def _prep_X(self, X):
        return _as2d(X)

    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.empty(0, dtype=int)
        return self._partial_fit(X, ya, np.unique(ya), sample_weight)

    def partial_fit(self, X, y, classes = None, sample_weight = None):
        ya = np.asarray(y, dtype=int).ravel()
        if classes is None:
            if self.classes_.shape[0] == 0:
                raise ValueError("classes must be passed on the first call to partial_fit.")
            return self._partial_fit(X, ya, self.classes_, sample_weight)
        else:
            cl = np.unique(np.asarray(classes, dtype=int))
            if self.classes_.shape[0] != 0:
                if cl.shape[0] != self.classes_.shape[0] or np.any(cl != self.classes_):
                    raise ValueError(f"`classes={cl}` is not the same as on last call to partial_fit, was: {self.classes_}")
            return self._partial_fit(X, ya, cl, sample_weight)

    @staticmethod
    def _update_mean_variance(n_past: float, mu: np.ndarray[float, 1],
                              var: np.ndarray[float, 1], X: np.ndarray[float, 2],
                              sw: Optional[np.ndarray[float, 1]]):
        if X.shape[0] == 0:
            return mu, var
        n_new = 0.0
        if sw is not None:
            w = sw.__val__()
            n_new = float(w.sum())
            if n_new == 0.0:
                return mu, var
            new_mu = (w @ X) / n_new
            new_var = (w @ ((X - new_mu) ** 2)) / n_new
        else:
            n_new = float(X.shape[0])
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)
        if n_past == 0.0:
            return new_mu, new_var
        n_total = n_past + n_new
        total_mu = (n_new * new_mu + n_past * mu) / n_total
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = old_ssd + new_ssd + (n_new * n_past / n_total) * (mu - new_mu) ** 2
        return total_mu, total_ssd / n_total

    def _partial_fit(self, X, ya: np.ndarray[int, 1], classes: np.ndarray[int, 1], sample_weight):
        Xa = _as2d(X)
        n, p = Xa.shape
        if ya.shape[0] != n:
            raise ValueError("Found input variables with inconsistent numbers of samples")
        first = self.classes_.shape[0] == 0
        sw_all = None if sample_weight is None else _sw(sample_weight, n)
        self.epsilon_ = self.var_smoothing * float(np.max(np.var(Xa, axis=0)))
        if first:
            self.classes_ = classes
            nc = classes.shape[0]
            self.n_features_in_ = p
            self.theta_ = np.zeros((nc, p))
            self.var_ = np.zeros((nc, p))
            self.class_count_ = np.zeros(nc)
            if self.priors is not None:
                pr = np.array(self.priors.__val__())
                if pr.shape[0] != nc:
                    raise ValueError("Number of priors must match number of classes.")
                if not np.isclose(pr.sum(), 1.0):
                    raise ValueError("The sum of the priors should be 1.")
                if np.any(pr < 0.0):
                    raise ValueError("Priors must be non-negative.")
                self.class_prior_ = pr
            else:
                self.class_prior_ = np.zeros(nc)
        else:
            if p != self.theta_.shape[1]:
                raise ValueError(f"Number of features {p} does not match previous data {self.theta_.shape[1]}.")
            self.var_ = self.var_ - self.epsilon_
        uy = np.unique(ya)
        for v in uy:
            i = _class_pos(self.classes_, int(v))
            mask = ya == v
            Xi = np.ascontiguousarray(Xa[mask])
            swi: Optional[np.ndarray[float, 1]] = None
            Ni = float(Xi.shape[0])
            if sw_all is not None:
                s = np.ascontiguousarray(sw_all[mask])
                swi = s
                Ni = float(s.sum())
            nt, nv = GaussianNB._update_mean_variance(self.class_count_[i],
                                                      np.ascontiguousarray(self.theta_[i, :]),
                                                      np.ascontiguousarray(self.var_[i, :]), Xi, swi)
            self.theta_[i, :] = nt
            self.var_[i, :] = nv
            self.class_count_[i] += Ni
        self.var_ = self.var_ + self.epsilon_
        if self.priors is None:
            self.class_prior_ = self.class_count_ / self.class_count_.sum()
        return self

    def _joint_log_likelihood(self, X: np.ndarray[float, 2]):
        nc = self.classes_.shape[0]
        n = X.shape[0]
        jll = np.empty((n, nc))
        for i in range(nc):
            jointi = np.log(self.class_prior_[i])
            vi = self.var_[i, :]
            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * vi))
            q = np.sum(((X - self.theta_[i, :]) ** 2) / vi, axis=1)
            for r in range(n):
                jll[r, i] = jointi + (n_ij - 0.5 * q[r])
        return jll

def _binarize_y(ya: np.ndarray[int, 1], classes: np.ndarray[int, 1],
                sw: Optional[np.ndarray[float, 1]]) -> np.ndarray[float, 2]:
    n = ya.shape[0]
    nc = classes.shape[0]
    Y = np.zeros((n, max(nc, 1)))
    for r in range(n):
        Y[r, _class_pos(classes, int(ya[r]))] = 1.0
    if sw is not None:
        w = sw.__val__()
        for r in range(n):
            for c in range(Y.shape[1]):
                Y[r, c] *= w[r]
    return Y

class _BaseDiscreteNB(_NBPredict):
    """Shared fit / partial_fit. Subclasses define _count, _update_feature_log_prob,
    _joint_log_likelihood and _prep_X."""

    def _check_alpha(self) -> float:
        if self.alpha < 0.0:
            raise ValueError("All values in alpha must be greater than 0.")
        if self.alpha < 1e-10 and not self.force_alpha:
            return 1e-10
        return self.alpha

    def _update_class_log_prior(self):
        nc = self.classes_.shape[0]
        if self.class_prior is not None:
            cp = np.array(self.class_prior.__val__())
            if cp.shape[0] != nc:
                raise ValueError("Number of priors must match number of classes.")
            self.class_log_prior_ = np.log(cp)
        elif self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(nc, -np.log(float(nc)))

    def fit(self, X, y, sample_weight = None):
        Xa = self._prep_X(X)
        ya = np.asarray(y, dtype=int).ravel()
        if ya.shape[0] != Xa.shape[0]:
            raise ValueError("Found input variables with inconsistent numbers of samples")
        self.classes_ = np.unique(ya)
        self.n_features_in_ = Xa.shape[1]
        sw = None if sample_weight is None else _sw(sample_weight, Xa.shape[0])
        Y = _binarize_y(ya, self.classes_, sw)
        self._init_counters(self.classes_.shape[0], Xa.shape[1])
        self._count(Xa, Y)
        self._update_feature_log_prob(self._check_alpha())
        self._update_class_log_prior()
        return self

    def partial_fit(self, X, y, classes = None, sample_weight = None):
        Xa = self._prep_X(X)
        ya = np.asarray(y, dtype=int).ravel()
        if self.classes_.shape[0] == 0:
            if classes is None:
                raise ValueError("classes must be passed on the first call to partial_fit.")
            else:
                self.classes_ = np.unique(np.asarray(classes, dtype=int))
            self.n_features_in_ = Xa.shape[1]
            self._init_counters(self.classes_.shape[0], Xa.shape[1])
        elif Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but {self.n_features_in_} were expected")
        sw = None if sample_weight is None else _sw(sample_weight, Xa.shape[0])
        Y = _binarize_y(ya, self.classes_, sw)
        self._count(Xa, Y)
        self._update_feature_log_prob(self._check_alpha())
        self._update_class_log_prior()
        return self

def _check_non_negative(X: np.ndarray[float, 2], who: str):
    if X.size > 0 and np.min(X) < 0.0:
        raise ValueError(f"Negative values in data passed to {who}")

class MultinomialNB(_BaseDiscreteNB):
    alpha: float
    force_alpha: bool
    fit_prior: bool
    class_prior: Optional[List[float]]
    classes_: np.ndarray[int, 1]
    class_count_: np.ndarray[float, 1]
    feature_count_: np.ndarray[float, 2]
    feature_log_prob_: np.ndarray[float, 2]
    class_log_prior_: np.ndarray[float, 1]
    n_features_in_: int
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, *, alpha: float = 1.0, force_alpha: bool = True,
                 fit_prior: bool = True, class_prior: Optional[List[float]] = None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes_ = np.empty(0, dtype=int)
        self.class_count_ = np.empty(0)
        self.feature_count_ = np.empty((0, 0))
        self.feature_log_prob_ = np.empty((0, 0))
        self.class_log_prior_ = np.empty(0)
        self.n_features_in_ = 0

    def _prep_X(self, X):
        return _as2d(X)

    def _init_counters(self, nc: int, p: int):
        self.class_count_ = np.zeros(nc)
        self.feature_count_ = np.zeros((nc, p))

    def _count(self, X: np.ndarray[float, 2], Y: np.ndarray[float, 2]):
        _check_non_negative(X, "MultinomialNB (input X)")
        self.feature_count_ = self.feature_count_ + Y.T @ X
        self.class_count_ = self.class_count_ + Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha: float):
        sfc = self.feature_count_ + alpha
        scc = sfc.sum(axis=1)
        self.feature_log_prob_ = np.log(sfc) - np.log(scc.reshape((-1, 1)))

    def _joint_log_likelihood(self, X: np.ndarray[float, 2]):
        return X @ self.feature_log_prob_.T + self.class_log_prior_

class ComplementNB(_BaseDiscreteNB):
    alpha: float
    force_alpha: bool
    fit_prior: bool
    class_prior: Optional[List[float]]
    norm: bool
    classes_: np.ndarray[int, 1]
    class_count_: np.ndarray[float, 1]
    feature_count_: np.ndarray[float, 2]
    feature_all_: np.ndarray[float, 1]
    feature_log_prob_: np.ndarray[float, 2]
    class_log_prior_: np.ndarray[float, 1]
    n_features_in_: int
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, *, alpha: float = 1.0, force_alpha: bool = True,
                 fit_prior: bool = True, class_prior: Optional[List[float]] = None,
                 norm: bool = False):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.norm = norm
        self.classes_ = np.empty(0, dtype=int)
        self.class_count_ = np.empty(0)
        self.feature_count_ = np.empty((0, 0))
        self.feature_all_ = np.empty(0)
        self.feature_log_prob_ = np.empty((0, 0))
        self.class_log_prior_ = np.empty(0)
        self.n_features_in_ = 0

    def _prep_X(self, X):
        return _as2d(X)

    def _init_counters(self, nc: int, p: int):
        self.class_count_ = np.zeros(nc)
        self.feature_count_ = np.zeros((nc, p))

    def _count(self, X: np.ndarray[float, 2], Y: np.ndarray[float, 2]):
        _check_non_negative(X, "ComplementNB (input X)")
        self.feature_count_ = self.feature_count_ + Y.T @ X
        self.class_count_ = self.class_count_ + Y.sum(axis=0)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _update_feature_log_prob(self, alpha: float):
        comp = self.feature_all_ + alpha - self.feature_count_
        logged = np.log(comp / comp.sum(axis=1, keepdims=True))
        if self.norm:
            self.feature_log_prob_ = logged / logged.sum(axis=1, keepdims=True)
        else:
            self.feature_log_prob_ = -logged

    def _joint_log_likelihood(self, X: np.ndarray[float, 2]):
        jll = X @ self.feature_log_prob_.T
        if self.classes_.shape[0] == 1:
            jll = jll + self.class_log_prior_
        return jll

class BernoulliNB(_BaseDiscreteNB):
    alpha: float
    force_alpha: bool
    binarize: Optional[float]
    fit_prior: bool
    class_prior: Optional[List[float]]
    classes_: np.ndarray[int, 1]
    class_count_: np.ndarray[float, 1]
    feature_count_: np.ndarray[float, 2]
    feature_log_prob_: np.ndarray[float, 2]
    class_log_prior_: np.ndarray[float, 1]
    n_features_in_: int
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, *, alpha: float = 1.0, force_alpha: bool = True,
                 binarize: Optional[float] = 0.0, fit_prior: bool = True,
                 class_prior: Optional[List[float]] = None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes_ = np.empty(0, dtype=int)
        self.class_count_ = np.empty(0)
        self.feature_count_ = np.empty((0, 0))
        self.feature_log_prob_ = np.empty((0, 0))
        self.class_log_prior_ = np.empty(0)
        self.n_features_in_ = 0

    def _prep_X(self, X):
        Xa = _as2d(X)
        if self.binarize is not None:
            th = self.binarize.__val__()
            Xa = np.where(Xa > th, 1.0, 0.0)
        return Xa

    def _init_counters(self, nc: int, p: int):
        self.class_count_ = np.zeros(nc)
        self.feature_count_ = np.zeros((nc, p))

    def _count(self, X: np.ndarray[float, 2], Y: np.ndarray[float, 2]):
        self.feature_count_ = self.feature_count_ + Y.T @ X
        self.class_count_ = self.class_count_ + Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha: float):
        sfc = self.feature_count_ + alpha
        scc = self.class_count_ + alpha * 2.0
        self.feature_log_prob_ = np.log(sfc) - np.log(scc.reshape((-1, 1)))

    def _joint_log_likelihood(self, X: np.ndarray[float, 2]):
        if X.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError(f"Expected input with {self.feature_log_prob_.shape[1]} features, got {X.shape[1]} instead")
        neg_prob = np.log(1.0 - np.exp(self.feature_log_prob_))
        jll = X @ (self.feature_log_prob_ - neg_prob).T
        return jll + (self.class_log_prior_ + neg_prob.sum(axis=1))

class CategoricalNB(_BaseDiscreteNB):
    """X holds non-negative integer category codes per feature."""
    alpha: float
    force_alpha: bool
    fit_prior: bool
    class_prior: Optional[List[float]]
    min_categories: Optional[List[int]]
    classes_: np.ndarray[int, 1]
    class_count_: np.ndarray[float, 1]
    category_count_: List[np.ndarray[float, 2]]
    feature_log_prob_: List[np.ndarray[float, 2]]
    class_log_prior_: np.ndarray[float, 1]
    n_categories_: np.ndarray[int, 1]
    n_features_in_: int
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, *, alpha: float = 1.0, force_alpha: bool = True,
                 fit_prior: bool = True, class_prior: Optional[List[float]] = None,
                 min_categories: Optional[List[int]] = None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.min_categories = min_categories
        self.classes_ = np.empty(0, dtype=int)
        self.class_count_ = np.empty(0)
        self.category_count_ = List[np.ndarray[float, 2]]()
        self.feature_log_prob_ = List[np.ndarray[float, 2]]()
        self.class_log_prior_ = np.empty(0)
        self.n_categories_ = np.empty(0, dtype=int)
        self.n_features_in_ = 0

    def _prep_X(self, X):
        Xa = _as2d(X)
        if Xa.size > 0 and np.min(Xa) < 0.0:
            raise ValueError("Negative values in data passed to CategoricalNB (input X)")
        return Xa

    def _init_counters(self, nc: int, p: int):
        self.class_count_ = np.zeros(nc)
        self.category_count_ = [np.zeros((nc, 0)) for _ in range(p)]

    def _count(self, X: np.ndarray[float, 2], Y: np.ndarray[float, 2]):
        nc = Y.shape[1]
        p = X.shape[1]
        self.class_count_ = self.class_count_ + Y.sum(axis=0)
        ncat = np.empty(p, dtype=int)
        for i in range(p):
            ncat[i] = int(np.max(X[:, i])) + 1
            if self.min_categories is not None:
                mc = self.min_categories.__val__()
                if len(mc) != p:
                    raise ValueError(f"'min_categories' should have shape ({p},) when an array-like is provided. Got ({len(mc)},) instead.")
                ncat[i] = max(ncat[i], mc[i])
        self.n_categories_ = ncat
        for i in range(p):
            cc = self.category_count_[i]
            need = ncat[i]
            if need > cc.shape[1]:
                nn = np.zeros((nc, need))
                nn[:, :cc.shape[1]] = cc
                cc = nn
            for r in range(X.shape[0]):
                k = int(X[r, i])
                for j in range(nc):
                    cc[j, k] += Y[r, j]
            self.category_count_[i] = cc

    def _update_feature_log_prob(self, alpha: float):
        out = List[np.ndarray[float, 2]]()
        for i in range(self.n_features_in_):
            scc = self.category_count_[i] + alpha
            scl = scc.sum(axis=1)
            out.append(np.log(scc) - np.log(scl.reshape((-1, 1))))
        self.feature_log_prob_ = out

    def _joint_log_likelihood(self, X: np.ndarray[float, 2]):
        nc = self.class_count_.shape[0]
        jll = np.zeros((X.shape[0], nc))
        for i in range(self.n_features_in_):
            flp = self.feature_log_prob_[i]
            for r in range(X.shape[0]):
                k = int(X[r, i])
                if k >= flp.shape[1]:
                    raise IndexError(f"index {k} is out of bounds for axis 1 with size {flp.shape[1]}")
                for j in range(nc):
                    jll[r, j] += flp[j, k]
        return jll + self.class_log_prior_
