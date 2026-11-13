# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# CalibratedClassifierCV (method 'sigmoid' / 'isotonic'; ensemble True or
# False; cv=None/int -> StratifiedKFold), _SigmoidCalibration (Platt via
# scipy L-BFGS-B on HalfBinomialLoss, as sklearn), calibration_curve.
# Integer class labels. The default estimator (None) is LinearSVC(random_state=0).

import math
import numpy as np
from .base import clone, _accuracy
from .isotonic import IsotonicRegression

def _hb_loss_grad(y: float, r: float) -> Tuple[float, float]:
    # closs_grad_half_binomial
    if r <= -37:
        e = math.exp(r)
        return e - y * r, e - y
    elif r <= -2:
        e = math.exp(r)
        return math.log1p(e) - y * r, ((1 - y) * e - y) / (1 + e)
    elif r <= 18:
        e = math.exp(-r)
        return math.log1p(e) + (1 - y) * r, ((1 - y) - y * e) / (1 + e)
    else:
        e = math.exp(-r)
        return e + (1 - y) * r, ((1 - y) - y * e) / (1 + e)

class _SigFG:
    F: np.ndarray[float, 1]
    T: np.ndarray[float, 1]
    sw: Optional[np.ndarray[float, 1]]
    _x: np.ndarray[float, 1]
    _g: np.ndarray[float, 1]
    _has: bool

    def __init__(self, F: np.ndarray[float, 1], T: np.ndarray[float, 1], sw: Optional[np.ndarray[float, 1]]):
        self.F = F
        self.T = T
        self.sw = sw
        self._x = np.zeros(0)
        self._g = np.zeros(2)
        self._has = False

    def fun(self, AB) -> float:
        a = float(AB[0]); b = float(AB[1])
        n = self.F.shape[0]
        loss = np.empty(n)
        g = np.empty(n)
        for i in range(n):
            r = -(a * self.F[i] + b)
            l, gg = _hb_loss_grad(self.T[i], r)
            if self.sw is not None:
                w = self.sw.__val__()[i]
                l *= w
                gg *= w
            loss[i] = l
            g[i] = gg
        self._g = np.array([-float(g @ self.F), -float(g.sum())])
        self._x = np.array([a, b])
        self._has = True
        return float(loss.sum())

    def grad(self, AB):
        if not (self._has and self._x[0] == AB[0] and self._x[1] == AB[1]):
            self.fun(AB)
        return self._g.copy()

def _sigmoid_calibration(predictions, y, sample_weight = None, max_abs_prediction_threshold: float = 30.0):
    from scipy.optimize import minimize
    F = np.ascontiguousarray(np.asarray(predictions, dtype=float).ravel())
    ya = np.asarray(y, dtype=float).ravel()
    scale = 1.0
    mx = float(np.max(np.abs(F)))
    if mx >= max_abs_prediction_threshold:
        scale = mx
        F = F / scale
    sw: Optional[np.ndarray[float, 1]] = None
    prior0 = 0.0; prior1 = 0.0
    if isinstance(sample_weight, NoneType):
        for v in ya:
            if v <= 0:
                prior0 += 1.0
        prior1 = float(ya.shape[0]) - prior0
    else:
        swa = np.asarray(sample_weight, dtype=float)
        sw = swa
        for i in range(ya.shape[0]):
            if ya[i] <= 0:
                prior0 += swa[i]
            else:
                prior1 += swa[i]
    T = np.empty(ya.shape[0])
    for i in range(ya.shape[0]):
        T[i] = (prior1 + 1.0) / (prior1 + 2.0) if ya[i] > 0 else 1.0 / (prior0 + 2.0)
    fg = _SigFG(F, T, sw)
    AB0 = np.array([0.0, math.log((prior0 + 1.0) / (prior1 + 1.0))])
    opts = {"gtol": 1e-6, "ftol": 64.0 * 2.220446049250313e-16}
    res = minimize(fg.fun, AB0, method="L-BFGS-B", jac=fg.grad, options=opts)
    return float(res.x[0]) / scale, float(res.x[1])

class _SigmoidCalibration:
    a_: float
    b_: float

    def __init__(self):
        self.a_ = 0.0
        self.b_ = 0.0

    def fit(self, X, y, sample_weight = None):
        self.a_, self.b_ = _sigmoid_calibration(X, y, sample_weight)
        return self

    def predict(self, T):
        from scipy.special import expit
        Ta = np.asarray(T, dtype=float).ravel()
        return expit(-(self.a_ * Ta + self.b_))

def _response(est, X) -> np.ndarray[float, 2]:
    """_get_response_values(['decision_function', 'predict_proba']) as 2-D
    (binary: one column; predict_proba binary: positive-class column)."""
    if hasattr(est, "decision_function"):
        d = np.asarray(est.decision_function(X), dtype=float)
        if d.ndim == 1:
            return np.ascontiguousarray(d.reshape(-1, 1))
        else:
            return np.ascontiguousarray(d.reshape(d.shape[0], -1))
    else:
        P = np.asarray(est.predict_proba(X), dtype=float)
        if P.shape[1] == 2:
            return np.ascontiguousarray(P[:, 1:2])
        return P

class _CalibratedClassifier[E]:
    estimator: E
    sig: List[_SigmoidCalibration]
    iso: List[IsotonicRegression]
    classes: np.ndarray[int, 1]
    method: str
    pos: List[int]

    def __init__(self, estimator: E, classes: np.ndarray[int, 1], method: str):
        self.estimator = estimator
        self.sig = []
        self.iso = []
        self.classes = classes
        self.method = method
        self.pos = []

    def fit_calibrators(self, preds: np.ndarray[float, 2], y: np.ndarray[int, 1], sw):
        k = self.classes.shape[0]
        est_classes = np.asarray(self.estimator.classes_, dtype=int)
        pos = [int(np.searchsorted(self.classes, c)) for c in est_classes]
        if preds.shape[1] == 1:
            # binary: label_binarize gives one column (positive = classes[1])
            pos = [pos[1] if len(pos) > 1 else 1]
        self.pos = pos
        for t in range(preds.shape[1]):
            ci = pos[t]
            target = np.empty(y.shape[0])
            for i in range(y.shape[0]):
                if k == 2:
                    target[i] = 1.0 if y[i] == self.classes[1] else 0.0
                else:
                    target[i] = 1.0 if y[i] == self.classes[ci] else 0.0
            col = np.ascontiguousarray(preds[:, t])
            if self.method == "isotonic":
                self.iso.append(IsotonicRegression(out_of_bounds="clip").fit(col, target, sw))
            else:
                self.sig.append(_SigmoidCalibration().fit(col, target, sw))

    def predict_proba(self, X):
        preds = _response(self.estimator, X)
        k = self.classes.shape[0]
        n = preds.shape[0]
        proba = np.zeros((n, k))
        for t in range(preds.shape[1]):
            ci = self.pos[t]
            if k == 2:
                ci = 1
            col = np.ascontiguousarray(preds[:, t])
            p = self.iso[t].predict(col) if self.method == "isotonic" else self.sig[t].predict(col)
            for i in range(n):
                proba[i, ci] = p[i]
        if k == 2:
            for i in range(n):
                proba[i, 0] = 1.0 - proba[i, 1]
        else:
            for i in range(n):
                s = 0.0
                for j in range(k):
                    s += proba[i, j]
                for j in range(k):
                    proba[i, j] = proba[i, j] / s if s != 0 else 1.0 / k
        for i in range(n):
            for j in range(k):
                if proba[i, j] > 1.0 and proba[i, j] <= 1.0 + 1e-5:
                    proba[i, j] = 1.0
        return proba

class CalibratedClassifierCV[E]:
    estimator: E
    method: str
    cv: int
    ensemble: bool
    classes_: np.ndarray[int, 1]
    calibrated_classifiers_: List[_CalibratedClassifier[E]]
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, estimator: E, *, method: str = "sigmoid", cv = None, n_jobs = None,
                 ensemble = "auto"):
        if method != "sigmoid" and method != "isotonic":
            raise NotImplementedError("CalibratedClassifierCV: method must be 'sigmoid' or 'isotonic'")
        self.estimator = estimator
        self.method = method
        self.cv = 5 if isinstance(cv, NoneType) else int(cv)
        self.ensemble = True if isinstance(ensemble, str) else bool(ensemble)
        self.classes_ = np.empty(0, dtype=int)
        self.calibrated_classifiers_ = []

    def _fit_est(self, e, X, y, sw):
        if isinstance(sw, NoneType):
            e.fit(X, y)
        else:
            e.fit(X, y, sample_weight=sw)

    def fit(self, X, y, sample_weight = None):
        from .model_selection import StratifiedKFold
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=int)
        self.classes_ = np.unique(ya)
        n = ya.shape[0]
        cnt = np.bincount(np.searchsorted(self.classes_, ya))
        if int(np.min(cnt)) < self.cv:
            raise ValueError(f"Requesting {self.cv}-fold cross-validation but provided less than {self.cv} examples for at least one class.")
        folds = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
        for tr, te in StratifiedKFold(self.cv).split(Xa, ya):
            folds.append((tr, te))
        self.calibrated_classifiers_ = []
        if self.ensemble:
            for tr, te in folds:
                e = clone(self.estimator)
                if isinstance(sample_weight, NoneType):
                    e.fit(Xa[tr], ya[tr])
                else:
                    e.fit(Xa[tr], ya[tr], sample_weight=np.asarray(sample_weight, dtype=float)[tr])
                preds = _response(e, Xa[te])
                cc = _CalibratedClassifier[E](e, self.classes_, self.method)
                if isinstance(sample_weight, NoneType):
                    cc.fit_calibrators(preds, ya[te], None)
                else:
                    cc.fit_calibrators(preds, ya[te], np.asarray(sample_weight, dtype=float)[te])
                self.calibrated_classifiers_.append(cc)
        else:
            e = clone(self.estimator)
            preds = np.empty((0, 0))
            for tr, te in folds:
                f = clone(self.estimator)
                f.fit(Xa[tr], ya[tr])
                p = _response(f, Xa[te])
                if preds.shape[0] == 0:
                    preds = np.empty((n, p.shape[1]))
                for t in range(te.shape[0]):
                    for j in range(p.shape[1]):
                        preds[te[t], j] = p[t, j]
            self._fit_est(e, Xa, ya, sample_weight)
            cc = _CalibratedClassifier[E](e, self.classes_, self.method)
            cc.fit_calibrators(preds, ya, sample_weight)
            self.calibrated_classifiers_.append(cc)
        return self

    def predict_proba(self, X):
        Xa = np.asarray(X, dtype=float)
        mean = np.zeros((Xa.shape[0], self.classes_.shape[0]))
        for cc in self.calibrated_classifiers_:
            mean += cc.predict_proba(Xa)
        mean /= float(len(self.calibrated_classifiers_))
        return mean

    def predict(self, X):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        out = np.empty(P.shape[0], dtype=int)
        for i in range(P.shape[0]):
            out[i] = self.classes_[idx[i]]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        return _accuracy(y, self.predict(X))

def calibration_curve(y_true, y_prob, *, pos_label = None, n_bins: int = 5, strategy: str = "uniform"):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_prob, dtype=float).ravel()
    if float(yp.min()) < 0 or float(yp.max()) > 1:
        raise ValueError("y_prob has values outside [0, 1].")
    labels = np.unique(yt)
    if labels.shape[0] > 2:
        raise ValueError(f"Only binary classification is supported. Provided labels {labels}.")
    pl = labels[labels.shape[0] - 1] if isinstance(pos_label, NoneType) else pos_label
    bins = np.empty(0)
    if strategy == "quantile":
        q = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(yp, q * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy must be either 'quantile' or 'uniform'.")
    inner = np.ascontiguousarray(bins[1:bins.shape[0] - 1])
    ids = np.searchsorted(inner, yp)
    sums = np.zeros(n_bins)
    trues = np.zeros(n_bins)
    tot = np.zeros(n_bins)
    for i in range(yp.shape[0]):
        b = int(ids[i])
        sums[b] += yp[i]
        trues[b] += 1.0 if yt[i] == pl else 0.0
        tot[b] += 1.0
    pt = List[float](); pp = List[float]()
    for b in range(n_bins):
        if tot[b] != 0:
            pt.append(trues[b] / tot[b])
            pp.append(sums[b] / tot[b])
    return np.array(pt), np.array(pp)
