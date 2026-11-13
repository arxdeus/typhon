# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# AdaBoostClassifier (SAMME) and AdaBoostRegressor (AdaBoost.R2)
# (sklearn/ensemble/_weight_boosting.py). The base estimator defaults to a
# decision stump / depth-3 regression tree; any estimator with
# fit(X, y, sample_weight) and a `random_state` field works.

import math
import numpy as np
from ..base import clone
from ..tree._classes import DecisionTreeClassifier, DecisionTreeRegressor, _check_rs, _r2
from ._base import MAX_INT

def _set_rs(est, seed: int):
    if hasattr(est, "random_state"):
        est.random_state = seed

class AdaBoostClassifier[E]:
    estimator: E
    n_estimators: int
    learning_rate: float
    random_state: Optional[int]
    estimators_: List[E]
    estimator_weights_: np.ndarray[float, 1]
    estimator_errors_: np.ndarray[float, 1]
    classes_: np.ndarray[int, 1]
    n_classes_: int
    n_features_in_: int

    def __init__(self, estimator: E = DecisionTreeClassifier(max_depth=1), *,
                 n_estimators: int = 50, learning_rate: float = 1.0,
                 random_state: Optional[int] = None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = List[E]()
        self.estimator_weights_ = np.empty(0)
        self.estimator_errors_ = np.empty(0)
        self.classes_ = np.empty(0, dtype=int)
        self.n_classes_ = 0
        self.n_features_in_ = 0

    def _sk_clone(self):
        return AdaBoostClassifier(clone(self.estimator), n_estimators=self.n_estimators,
                                  learning_rate=self.learning_rate, random_state=self.random_state)

    @property
    def _estimator_type(self) -> str:
        return "classifier"

    def fit(self, X, y, sample_weight = None):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y)
        n = Xa.shape[0]
        self.n_features_in_ = Xa.shape[1]
        sw = np.ones(n)
        if sample_weight is not None:
            sw = np.array(np.asarray(sample_weight, dtype=float))
        sw /= np.sum(sw)
        self.estimators_ = List[E]()
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.ones(self.n_estimators)
        rs = _check_rs(self.random_state)
        eps = 2.220446049250313e-16
        zero_mask = [sw[i] == 0.0 for i in range(n)]
        for iboost in range(self.n_estimators):
            for i in range(n):
                if sw[i] < eps:
                    sw[i] = eps
                if zero_mask[i]:
                    sw[i] = 0.0
            est = clone(self.estimator)
            _set_rs(est, int(rs.randint(MAX_INT)))
            self.estimators_.append(est)
            est.fit(Xa, ya, sample_weight=sw)
            y_pred = est.predict(Xa)
            if iboost == 0:
                self.classes_ = est.classes_
                self.n_classes_ = self.classes_.shape[0]
            inc = np.empty(n)
            for i in range(n):
                inc[i] = 1.0 if y_pred[i] != ya[i] else 0.0
            err = float(np.average(inc, weights=sw))
            if err <= 0.0:
                self.estimator_weights_[iboost] = 1.0
                self.estimator_errors_[iboost] = 0.0
                break
            nc = self.n_classes_
            if err >= 1.0 - 1.0 / float(nc):
                self.estimators_.pop()
                if len(self.estimators_) == 0:
                    raise ValueError("BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit.")
                break
            w = self.learning_rate * (math.log((1.0 - err) / err) + math.log(float(nc) - 1.0))
            if iboost != self.n_estimators - 1:
                for i in range(n):
                    sw[i] = math.exp(math.log(sw[i]) + w * inc[i] * (1.0 if sw[i] > 0.0 else 0.0))
            self.estimator_weights_[iboost] = w
            self.estimator_errors_[iboost] = err
            s = float(np.sum(sw))
            if not math.isfinite(s) or s <= 0.0:
                break
            if iboost < self.n_estimators - 1:
                sw /= s
        return self

    def _decision_rows(self, X, limit: int):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        nc = self.n_classes_
        pred = np.zeros((n, nc))
        norm = 0.0
        for j in range(limit):
            w = self.estimator_weights_[j]
            norm += w
            p = self.estimators_[j].predict(Xa)
            for i in range(n):
                for c in range(nc):
                    pred[i, c] += w if p[i] == self.classes_[c] else -1.0 / float(nc - 1) * w
        return pred, norm

    def _finish_decision(self, pred, norm: float):
        pred /= norm
        if self.n_classes_ == 2:
            n = pred.shape[0]
            out = np.empty((n, 1))
            for i in range(n):
                out[i, 0] = -pred[i, 0] + pred[i, 1]
            return out
        return pred

    def decision_function(self, X):
        """(n, 1) for binary problems (sklearn returns (n,); the static return
        type cannot depend on n_classes), (n, n_classes) otherwise."""
        pred, _ = self._decision_rows(X, len(self.estimators_))
        tot = float(np.sum(self.estimator_weights_))
        return self._finish_decision(pred, tot)

    def decision_function_1d(self, X):
        """Binary-only (n,) decision function (sklearn's binary output shape)."""
        d = self.decision_function(X)
        if d.shape[1] != 1:
            raise ValueError("decision_function_1d is only defined for binary problems")
        return d.ravel()

    def staged_decision_function(self, X):
        for k in range(1, len(self.estimators_) + 1):
            pred, norm = self._decision_rows(X, k)
            yield self._finish_decision(pred, norm)

    def _labels(self, d):
        n = d.shape[0]
        out = np.empty(n, dtype=int)
        for i in range(n):
            if d.shape[1] == 1:
                out[i] = self.classes_[1 if d[i, 0] > 0.0 else 0]
            else:
                best = 0
                for c in range(1, d.shape[1]):
                    if d[i, c] > d[i, best]:
                        best = c
                out[i] = self.classes_[best]
        return out

    def _proba(self, d):
        n = d.shape[0]
        nc = self.n_classes_
        z = np.empty((n, nc))
        for i in range(n):
            if nc == 2:
                z[i, 0] = -d[i, 0] / 2.0
                z[i, 1] = d[i, 0] / 2.0
            else:
                for c in range(nc):
                    z[i, c] = d[i, c] / float(nc - 1)
        for i in range(n):
            mx = z[i, 0]
            for c in range(1, nc):
                if z[i, c] > mx:
                    mx = z[i, c]
            s = 0.0
            for c in range(nc):
                z[i, c] = math.exp(z[i, c] - mx)
                s += z[i, c]
            for c in range(nc):
                z[i, c] /= s
        return z

    def predict(self, X):
        pred, _ = self._decision_rows(X, len(self.estimators_))
        return self._labels(self._finish_decision(pred, float(np.sum(self.estimator_weights_))))

    def staged_predict(self, X):
        for k in range(1, len(self.estimators_) + 1):
            pred, norm = self._decision_rows(X, k)
            yield self._labels(self._finish_decision(pred, norm))

    def predict_proba(self, X):
        pred, _ = self._decision_rows(X, len(self.estimators_))
        return self._proba(self._finish_decision(pred, float(np.sum(self.estimator_weights_))))

    def staged_predict_proba(self, X):
        for k in range(1, len(self.estimators_) + 1):
            pred, norm = self._decision_rows(X, k)
            yield self._proba(self._finish_decision(pred, norm))

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def score(self, X, y, sample_weight = None) -> float:
        from ..base import _accuracy
        return _accuracy(y, self.predict(X))

    @property
    def feature_importances_(self):
        norm = float(np.sum(self.estimator_weights_))
        out = np.zeros(self.n_features_in_)
        for j in range(len(self.estimators_)):
            out += self.estimator_weights_[j] * self.estimators_[j].feature_importances_
        return out / norm

class AdaBoostRegressor[E]:
    estimator: E
    n_estimators: int
    learning_rate: float
    loss: str
    random_state: Optional[int]
    estimators_: List[E]
    estimator_weights_: np.ndarray[float, 1]
    estimator_errors_: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, estimator: E = DecisionTreeRegressor(max_depth=3), *,
                 n_estimators: int = 50, learning_rate: float = 1.0, loss: str = "linear",
                 random_state: Optional[int] = None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.estimators_ = List[E]()
        self.estimator_weights_ = np.empty(0)
        self.estimator_errors_ = np.empty(0)
        self.n_features_in_ = 0

    def _sk_clone(self):
        return AdaBoostRegressor(clone(self.estimator), n_estimators=self.n_estimators,
                                 learning_rate=self.learning_rate, loss=self.loss,
                                 random_state=self.random_state)

    @property
    def _estimator_type(self) -> str:
        return "regressor"

    def fit(self, X, y, sample_weight = None):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        n = Xa.shape[0]
        self.n_features_in_ = Xa.shape[1]
        sw = np.ones(n)
        if sample_weight is not None:
            sw = np.array(np.asarray(sample_weight, dtype=float))
        sw /= np.sum(sw)
        self.estimators_ = List[E]()
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.ones(self.n_estimators)
        rs = _check_rs(self.random_state)
        eps = 2.220446049250313e-16
        zero_mask = [sw[i] == 0.0 for i in range(n)]
        for iboost in range(self.n_estimators):
            for i in range(n):
                if sw[i] < eps:
                    sw[i] = eps
                if zero_mask[i]:
                    sw[i] = 0.0
            est = clone(self.estimator)
            _set_rs(est, int(rs.randint(MAX_INT)))
            self.estimators_.append(est)
            idx = rs.choice(np.arange(n), size=n, replace=True, p=sw)
            est.fit(Xa[idx], ya[idx])
            y_pred = est.predict(Xa)
            mask = [sw[i] > 0.0 for i in range(n)]
            rows = [i for i in range(n) if mask[i]]
            ev = np.array([abs(y_pred[i] - ya[i]) for i in rows])
            emax = float(np.max(ev))
            if emax != 0.0:
                ev /= emax
            if self.loss == "square":
                ev = ev ** 2
            elif self.loss == "exponential":
                ev = 1.0 - np.exp(-ev)
            msw = np.array([sw[i] for i in rows])
            err = float(np.sum(msw * ev))
            if err <= 0.0:
                self.estimator_weights_[iboost] = 1.0
                self.estimator_errors_[iboost] = 0.0
                break
            elif err >= 0.5:
                if len(self.estimators_) > 1:
                    self.estimators_.pop()
                break
            beta = err / (1.0 - err)
            w = self.learning_rate * math.log(1.0 / beta)
            if iboost != self.n_estimators - 1:
                for t in range(len(rows)):
                    sw[rows[t]] *= math.pow(beta, (1.0 - ev[t]) * self.learning_rate)
            self.estimator_weights_[iboost] = w
            self.estimator_errors_[iboost] = err
            s = float(np.sum(sw))
            if not math.isfinite(s) or s <= 0.0:
                break
            if iboost < self.n_estimators - 1:
                sw /= s
        return self

    def _median_predict(self, X, limit: int):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        preds = np.empty((n, limit))
        for j in range(limit):
            p = self.estimators_[j].predict(Xa)
            for i in range(n):
                preds[i, j] = p[i]
        out = np.empty(n)
        for i in range(n):
            order = np.argsort(preds[i])
            cdf = np.cumsum(self.estimator_weights_[order])
            half = 0.5 * cdf[limit - 1]
            m = 0
            while cdf[m] < half:
                m += 1
            out[i] = preds[i, order[m]]
        return out

    def predict(self, X):
        return self._median_predict(X, len(self.estimators_))

    def staged_predict(self, X):
        for k in range(1, len(self.estimators_) + 1):
            yield self._median_predict(X, k)

    def score(self, X, y, sample_weight = None) -> float:
        return _r2(y, self.predict(X))

    @property
    def feature_importances_(self):
        norm = float(np.sum(self.estimator_weights_))
        out = np.zeros(self.n_features_in_)
        for j in range(len(self.estimators_)):
            out += self.estimator_weights_[j] * self.estimators_[j].feature_importances_
        return out / norm
