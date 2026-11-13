# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from .base import estimator_repr as _estimator_repr
import numpy as np
import scipy.linalg as sl
from .base import _accuracy

def _empirical_cov(X):
    # sklearn.covariance.empirical_covariance (assume_centered=False)
    Xc = X - X.mean(0)
    return (Xc.T @ Xc) / float(X.shape[0])

def _ledoit_wolf_shrinkage(Xin) -> float:
    n, p = Xin.shape
    if p == 1:
        return 0.0
    X = Xin - Xin.mean(0)
    X2 = X ** 2
    emp_cov_trace = np.sum(X2, axis=0) / float(n)
    mu = np.sum(emp_cov_trace) / float(p)
    # block_size=1000: one block for p < 1000
    delta_ = float(np.sum((X.T @ X) ** 2))
    delta_ /= float(n) ** 2
    beta_ = float(np.sum(X2.T @ X2))
    beta = 1.0 / float(p * n) * (beta_ / float(n) - delta_)
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + float(p) * mu ** 2
    delta /= float(p)
    beta = min(beta, delta)
    return 0.0 if beta == 0 else beta / delta

def _ledoit_wolf(X):
    n, p = X.shape
    if p == 1:
        Xc = X - X.mean()
        return np.full((1, 1), float((Xc ** 2).mean()))
    sh = _ledoit_wolf_shrinkage(X)
    emp = _empirical_cov(X)
    mu = np.sum(np.trace(emp)) / float(p)
    out = (1.0 - sh) * emp
    for i in range(p):
        out[i, i] += sh * mu
    return out

def _cov(X, shrinkage: str, shrink_value: float):
    # shrinkage: "" (empirical), "auto" (Ledoit-Wolf on standardized X), "float"
    if shrinkage == "auto":
        from .preprocessing import StandardScaler
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        s = _ledoit_wolf(Xs)
        return sc.scale_[:, None] * s * sc.scale_[None, :]
    elif shrinkage == "float":
        emp = _empirical_cov(X)
        p = emp.shape[0]
        mu = np.trace(emp) / float(p)
        out = (1.0 - shrink_value) * emp
        for i in range(p):
            out[i, i] += shrink_value * mu
        return out
    else:
        return _empirical_cov(X)

class LinearDiscriminantAnalysis:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """solver 'svd' (default), 'lsqr', 'eigen'; shrinkage None / 'auto' /
    float for lsqr and eigen; priors from class frequencies unless given."""
    solver: str
    _shrink_kind: str
    _shrink_value: float
    priors: Optional[List[float]]
    n_components: Optional[int]
    store_covariance: bool
    tol: float
    classes_: np.ndarray[int, 1]
    priors_: np.ndarray[float, 1]
    means_: np.ndarray[float, 2]
    xbar_: np.ndarray[float, 1]
    scalings_: np.ndarray[float, 2]
    coef_: np.ndarray[float, 2]
    intercept_: np.ndarray[float, 1]
    explained_variance_ratio_: np.ndarray[float, 1]
    _max_components: int

    covariance_: np.ndarray[float, 2]

    def __init__(self, solver: str = "svd", shrinkage = None,
                 priors: Optional[List[float]] = None, n_components: Optional[int] = None,
                 store_covariance: bool = False, tol: float = 1e-4,
                 covariance_estimator = None):
        if solver != "svd" and solver != "lsqr" and solver != "eigen":
            raise ValueError(f"unknown solver {solver}")
        self._shrink_value = 0.0
        if shrinkage is None:
            self._shrink_kind = ""
        elif isinstance(shrinkage, str):
            if shrinkage != "auto":
                raise ValueError("shrinkage must be 'auto', a float or None")
            self._shrink_kind = "auto"
        else:
            self._shrink_kind = "float"
            self._shrink_value = float(shrinkage)
        if solver == "svd" and self._shrink_kind != "":
            raise NotImplementedError("shrinkage not supported with svd solver.")
        self.solver = solver
        self.covariance_ = np.empty((0, 0))
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol
        self.classes_ = np.empty(0, dtype=int)
        self.priors_ = np.empty(0)
        self.means_ = np.empty((0, 0))
        self.xbar_ = np.empty(0)
        self.scalings_ = np.empty((0, 0))
        self.coef_ = np.empty((0, 0))
        self.intercept_ = np.empty(0)
        self.explained_variance_ratio_ = np.empty(0)
        self._max_components = 0

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=int)
        n_samples, n_features = Xa.shape
        classes, yi = np.unique(ya, return_inverse=True)
        self.classes_ = classes
        n_classes = classes.shape[0]
        if n_samples == n_classes:
            raise ValueError("The number of samples must be more than the number of classes.")
        cnt = np.bincount(yi)
        if self.priors is None:
            self.priors_ = cnt.astype(float) / float(n_samples)
        else:
            pr = np.array(self.priors.__val__())
            self.priors_ = pr / pr.sum()
        mc = min(n_classes - 1, n_features)
        if self.n_components is None:
            self._max_components = mc
        else:
            self._max_components = self.n_components
        # _class_means: np.add.at then divide (row order accumulation)
        means = np.zeros((n_classes, n_features))
        for i in range(n_samples):
            for j in range(n_features):
                means[yi[i], j] += Xa[i, j]
        for k in range(n_classes):
            for j in range(n_features):
                means[k, j] /= float(cnt[k])
        self.means_ = means
        if self.solver != "svd":
            self._fit_cov_solver(Xa, ya, classes, n_classes)
            return self
        # Xc: per class (in class order), samples in original order
        Xc = np.empty((n_samples, n_features))
        r = 0
        for k in range(n_classes):
            for i in range(n_samples):
                if yi[i] == k:
                    for j in range(n_features):
                        Xc[r, j] = Xa[i, j] - means[k, j]
                    r += 1
        self.xbar_ = self.priors_ @ means
        std = np.std(Xc, axis=0)
        for j in range(n_features):
            if std[j] == 0.0:
                std[j] = 1.0
        fac = 1.0 / float(n_samples)
        Xs = np.sqrt(fac) * (Xc / std)
        _, S, Vt = sl.svd(Xs, full_matrices=False)
        rank = 0
        for s in S:
            if s > self.tol:
                rank += 1
        scalings = (Vt[:rank, :] / std).T / S[:rank]
        fac2 = 1.0 if n_classes == 1 else 1.0 / float(n_classes - 1)
        Xm = ((np.sqrt((float(n_samples) * self.priors_) * fac2)) * (means - self.xbar_).T).T @ scalings
        _, S2, Vt2 = sl.svd(Xm, full_matrices=False)
        if self._max_components == 0:
            self.explained_variance_ratio_ = np.empty(0)
        else:
            self.explained_variance_ratio_ = ((S2 ** 2) / np.sum(S2 ** 2))[:self._max_components]
        rank2 = 0
        for s in S2:
            if s > self.tol * S2[0]:
                rank2 += 1
        self.scalings_ = scalings @ Vt2.T[:, :rank2]
        coef = (means - self.xbar_) @ self.scalings_
        intercept = -0.5 * np.sum(coef ** 2, axis=1) + np.log(self.priors_)
        coef_full = coef @ self.scalings_.T
        intercept -= self.xbar_ @ coef_full.T
        if n_classes == 2:
            c = (coef_full[1, :] - coef_full[0, :]).reshape((1, -1))
            self.coef_ = np.ascontiguousarray(c)
            self.intercept_ = np.array([intercept[1] - intercept[0]])
        else:
            self.coef_ = coef_full
            self.intercept_ = intercept
        return self

    def _fit_cov_solver(self, Xa, ya, classes, n_classes: int):
        p = Xa.shape[1]
        Sw = np.zeros((p, p))
        for k in range(n_classes):
            Sw += self.priors_[k] * _cov(Xa[ya == classes[k]], self._shrink_kind, self._shrink_value)
        self.covariance_ = Sw
        if self.solver == "lsqr":
            coef = sl.lstsq(Sw, self.means_.T)[0].T
            self.xbar_ = np.empty(0)
        else:
            St = _cov(Xa, self._shrink_kind, self._shrink_value)
            Sb = St - Sw
            evals, evecs = sl.eigh(Sb, Sw)
            order = np.argsort(evals)[::-1]
            self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][:self._max_components]
            ev = np.ascontiguousarray(evecs[:, order])
            self.scalings_ = ev
            coef = (self.means_ @ ev) @ ev.T
        coef = np.ascontiguousarray(coef)
        intercept = -0.5 * np.diag(self.means_ @ coef.T) + np.log(self.priors_)
        if n_classes == 2:
            self.coef_ = np.ascontiguousarray((coef[1, :] - coef[0, :]).reshape((1, -1)))
            self.intercept_ = np.array([intercept[1] - intercept[0]])
        else:
            self.coef_ = coef
            self.intercept_ = intercept

    def decision_function(self, X):
        Xa = np.asarray(X, dtype=float)
        scores = Xa @ self.coef_.T + self.intercept_
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        n = scores.shape[0]
        out = np.empty(n, dtype=int)
        if scores.shape[1] == 1:
            for i in range(n):
                out[i] = self.classes_[1] if scores[i, 0] > 0 else self.classes_[0]
        else:
            idx = np.argmax(scores, axis=1)
            for i in range(n):
                out[i] = self.classes_[idx[i]]
        return out

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        if self.solver == "lsqr":
            raise NotImplementedError("transform not implemented for 'lsqr' solver (use 'svd' or 'eigen').")
        if self.solver == "eigen":
            return (Xa @ self.scalings_)[:, :self._max_components]
        return ((Xa - self.xbar_) @ self.scalings_)[:, :self._max_components]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def predict_proba(self, X):
        d = self.decision_function(X)
        n = d.shape[0]
        if d.shape[1] == 1:
            p = np.empty((n, 2))
            for i in range(n):
                e = 1.0 / (1.0 + np.exp(-d[i, 0]))
                p[i, 1] = e
                p[i, 0] = 1.0 - e
            return p
        else:
            m = np.max(d, axis=1, keepdims=True)
            E = np.exp(d - m)
            return E / np.sum(E, axis=1, keepdims=True)

    def predict_log_proba(self, X):
        pr = self.predict_proba(X)
        return np.log(pr)

    def score(self, X, y, sample_weight = None):
        return _accuracy(y, self.predict(X))

class QuadraticDiscriminantAnalysis:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """solver 'svd' (reg_param) or 'eigen' (shrinkage None / 'auto' / float)."""
    solver: str
    _shrink_kind: str
    _shrink_value: float
    priors: Optional[List[float]]
    reg_param: float
    store_covariance: bool
    tol: float
    classes_: np.ndarray[int, 1]
    priors_: np.ndarray[float, 1]
    means_: np.ndarray[float, 2]
    scalings_: List[np.ndarray[float, 1]]
    rotations_: List[np.ndarray[float, 2]]
    covariance_: List[np.ndarray[float, 2]]

    def __init__(self, *, solver: str = "svd", shrinkage = None,
                 priors: Optional[List[float]] = None, reg_param: float = 0.0,
                 store_covariance: bool = False, tol: float = 1e-4, covariance_estimator = None):
        if solver != "svd" and solver != "eigen":
            raise ValueError(f"unknown solver {solver}")
        self._shrink_value = 0.0
        if shrinkage is None:
            self._shrink_kind = ""
        elif isinstance(shrinkage, str):
            self._shrink_kind = "auto"
        else:
            self._shrink_kind = "float"
            self._shrink_value = float(shrinkage)
        self.solver = solver
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol
        self.classes_ = np.empty(0, dtype=int)
        self.priors_ = np.empty(0)
        self.means_ = np.empty((0, 0))
        self.scalings_ = []
        self.rotations_ = []
        self.covariance_ = []

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=int)
        n, p = Xa.shape
        classes, cnt = np.unique(ya, return_counts=True)
        nc = classes.shape[0]
        if nc < 2:
            raise ValueError(f"The number of classes has to be greater than one. Got {nc} class.")
        self.classes_ = classes
        if self.priors is None:
            self.priors_ = cnt.astype(float) / float(n)
        else:
            self.priors_ = np.array(self.priors.__val__())
        if self.solver == "svd" and self._shrink_kind != "":
            raise NotImplementedError("shrinkage not supported with 'svd' solver.")
        means = np.empty((nc, p))
        self.scalings_ = []
        self.rotations_ = []
        self.covariance_ = []
        for ci in range(nc):
            rows = [i for i in range(n) if ya[i] == classes[ci]]
            if len(rows) == 1:
                raise ValueError(f"y has only 1 sample in class {classes[ci]}, covariance is ill defined.")
            Xc = np.ascontiguousarray(Xa[np.array(rows)])
            mc = Xc.mean(0)
            means[ci] = mc
            S = np.empty(0)
            R = np.empty((0, 0))
            C = np.empty((0, 0))
            if self.solver == "svd":
                _, Sv, Vt = np.linalg.svd(Xc - mc, full_matrices=False)
                S = Sv ** 2 / float(Xc.shape[0])
                S = (1 - self.reg_param) * S + self.reg_param
                R = np.ascontiguousarray(Vt.T)
                if self.store_covariance:
                    C = S * Vt.T @ Vt
            else:
                C = _cov(Xc, self._shrink_kind, self._shrink_value)
                sc, rot = sl.eigh(C)
                order = np.argsort(sc)[::-1]
                R = np.ascontiguousarray(rot[:, order])
                S = sc[order]
            rank = 0
            for v in S:
                if v > self.tol:
                    rank += 1
            if rank < p:
                if self.solver == "svd" and Xc.shape[0] <= p:
                    raise np.linalg.LinAlgError(f"The covariance matrix of class {classes[ci]} is not full rank. When using `solver='svd'` the number of samples in each class should be more than the number of features, but class {classes[ci]} has {Xc.shape[0]} samples and {p} features. Try using `solver='eigen'` and setting the parameter `shrinkage` for regularization.")
                else:
                    mp = "shrinkage" if self.solver == "eigen" else "reg_param"
                    raise np.linalg.LinAlgError(f"The covariance matrix of class {classes[ci]} is not full rank. Increase the value of `{mp}` to reduce the collinearity.")
            self.scalings_.append(S)
            self.rotations_.append(R)
            if self.store_covariance:
                self.covariance_.append(C)
        self.means_ = means
        return self

    def _decision_function(self, X):
        Xa = np.asarray(X, dtype=float)
        nc = self.classes_.shape[0]
        n = Xa.shape[0]
        out = np.empty((n, nc))
        for i in range(nc):
            R = self.rotations_[i]
            S = self.scalings_[i]
            Xm = Xa - self.means_[i]
            X2 = np.dot(Xm, R * S ** (-0.5))
            nrm = np.sum(X2 ** 2, axis=1)
            u = float(np.sum(np.log(S)))
            lp = float(np.log(self.priors_[i]))
            for r in range(n):
                out[r, i] = -0.5 * (nrm[r] + u) + lp
        return out

    def decision_function(self, X):
        # DEVIATION: binary case returned as (n, 1)
        s = self._decision_function(X)
        if self.classes_.shape[0] == 2:
            return np.ascontiguousarray(s[:, 1:2] - s[:, 0:1])
        return s

    def predict(self, X):
        s = self._decision_function(X)
        idx = np.argmax(s, axis=1)
        out = np.empty(s.shape[0], dtype=int)
        for i in range(s.shape[0]):
            out[i] = self.classes_[idx[i]]
        return out

    def predict_log_proba(self, X):
        s = self._decision_function(X)
        ll = s - np.max(s, axis=1)[:, None]
        return ll - np.log(np.sum(np.exp(ll), axis=1))[:, None]

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def score(self, X, y, sample_weight = None):
        return _accuracy(y, self.predict(X))
