# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

# Transfer learning: domain encoding, TLDummy, TLCenter and TLScale for SPD
# matrix input (3-D arrays). TLRotate supports manifold input only.

import numpy as np
from .geometry.base import sqrtm, invsqrtm, powm
from .geometry.distance import distance
from .geometry.mean import gmean, check_weights
from .preprocessing import Whitening
from .classification import MDM
from .geometry.geodesic import geodesic

def encode_domains(X, y, domain: List[str]):
    y = np.asarray(y)
    if y.shape[0] != len(domain):
        raise ValueError("Input lengths don't match")
    out = List[str]()
    for i in range(len(domain)):
        out.append(domain[i] + "/" + str(y[i]))
    return X, out

def decode_domains(X_enc, y_enc: List[str]):
    y = List[str]()
    dom = List[str]()
    for s in y_enc:
        parts = s.split("/")
        dom.append(parts[len(parts) - 2])
        y.append(parts[len(parts) - 1])
    return X_enc, y, dom

def _unique_sorted(v: List[str]) -> List[str]:
    s = Set[str]()
    for x in v:
        s.add(x)
    out = list(s)
    out.sort()
    return out

def _mask(dom: List[str], d: str):
    return np.array([x == d for x in dom])

class TLDummy:
    def __init__(self):
        pass

    def fit(self, X, y_enc = None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)

    def fit_transform(self, X, y_enc = None):
        return np.asarray(X, dtype=float)

class TLCenter:
    target_domain: str
    metric: str
    domains_: List[str]
    centers_: Dict[str, Whitening]

    def __init__(self, target_domain: str, metric: str = "riemann"):
        self.target_domain = target_domain
        self.metric = metric
        self.domains_ = List[str]()
        self.centers_ = Dict[str, Whitening]()

    def fit(self, X, y_enc, sample_weight = None):
        X = np.asarray(X, dtype=float)
        _, _, dom = decode_domains(X, y_enc)
        sw = check_weights(sample_weight, X.shape[0])
        self.centers_ = Dict[str, Whitening]()
        self.domains_ = _unique_sorted(dom)
        for d in self.domains_:
            idx = _mask(dom, d)
            w = Whitening(self.metric)
            w.fit(X[idx], sample_weight=sw[idx])
            self.centers_[d] = w
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        td = self.target_domain if self.target_domain != "" else self.domains_[len(self.domains_) - 1]
        return self.centers_[td].transform(X)

    def fit_transform(self, X, y_enc, sample_weight = None):
        X = np.asarray(X, dtype=float)
        self.fit(X, y_enc, sample_weight=sample_weight)
        _, _, dom = decode_domains(X, y_enc)
        out = np.zeros(X.shape)
        for d in self.domains_:
            idx = _mask(dom, d)
            out[idx] = self.centers_[d].transform(X[idx])
        return out

class TLScale:
    target_domain: str
    final_dispersion: float
    centered_data: bool
    metric: str
    domains_: List[str]
    scales_: Dict[str, float]
    _means: Dict[str, np.ndarray[float, 2]]

    def __init__(self, target_domain: str, final_dispersion: float = 1.0,
                 centered_data: bool = False, metric: str = "riemann"):
        self.target_domain = target_domain
        self.final_dispersion = final_dispersion
        self.centered_data = centered_data
        self.metric = metric
        self.domains_ = List[str]()
        self.scales_ = Dict[str, float]()
        self._means = Dict[str, np.ndarray[float, 2]]()

    def fit(self, X, y_enc, sample_weight = None):
        X = np.asarray(X, dtype=float)
        _, _, dom = decode_domains(X, y_enc)
        sw = check_weights(sample_weight, X.shape[0])
        self.scales_ = Dict[str, float]()
        self._means = Dict[str, np.ndarray[float, 2]]()
        self.domains_ = _unique_sorted(dom)
        for d in self.domains_:
            idx = _mask(dom, d)
            swd = check_weights(sw[idx], int(np.sum(np.asarray(idx, dtype=int))))
            if self.centered_data:
                m = np.eye(X.shape[2])
            else:
                m = gmean(X[idx], metric="riemann", sample_weight=swd)
            self._means[d] = m
            dist = distance(X[idx], m, metric=self.metric, squared=True)[:, 0]
            self.scales_[d] = float(np.sum(swd * dist))
        return self

    def _apply(self, X, d: str):
        m = self._means[d]
        Xc = X
        if not self.centered_data:
            mi = invsqrtm(m)
            Xc = mi @ X @ mi
        Xn = powm(Xc, float(np.sqrt(self.final_dispersion / self.scales_[d])))
        if not self.centered_data:
            ms = sqrtm(m)
            Xn = ms @ Xn @ ms
        return Xn

    def transform(self, X):
        return self._apply(np.asarray(X, dtype=float), self.target_domain)

    def fit_transform(self, X, y_enc, sample_weight = None):
        X = np.asarray(X, dtype=float)
        self.fit(X, y_enc, sample_weight=sample_weight)
        _, _, dom = decode_domains(X, y_enc)
        out = np.zeros(X.shape)
        for d in self.domains_:
            idx = _mask(dom, d)
            out[idx] = self._apply(X[idx], d)
        return out

# ---------------------------------------------------------------------------
# TLRotate (manifold input): Riemannian Procrustes analysis on orthogonal
# matrices, ported from pyriemann.optimization.grassmann.

def _rot_project(X, U):
    A = X.T @ U
    return X @ (A - A.T) / 2.0

def _rot_retract(X, v):
    Q, R = np.linalg.qr(X + v)
    for i in range(R.shape[0]):
        if R[i, i] < 0:
            Q[:, i] = -1.0 * Q[:, i]
    return Q

def _rot_loss(Q, X, Y, weights, metric: str):
    n = X.shape[0]
    P = np.empty(Y.shape)
    for i in range(n):
        P[i] = Q @ Y[i] @ Q.T
    return weights @ distance(X, P, metric=metric)

def _rot_grad(Q, X, Y, weights, metric: str):
    n = X.shape[0]
    d = X.shape[1]
    G = np.zeros((d, d))
    if metric == "euclid":
        for i in range(n):
            G = G + weights[i] * (-4.0 * (X[i] - Q @ Y[i] @ Q.T) @ Q @ Y[i])
    elif metric == "riemann":
        for i in range(n):
            Xi = X[i]
            ev, U = np.linalg.eigh(Xi)
            Xs = (U * np.sqrt(ev)) @ U.T
            Xis = (U / np.sqrt(ev)) @ U.T
            S = Xis @ (Q @ Y[i] @ Q.T) @ Xis
            S = (S + S.T) / 2.0
            w, V = np.linalg.eigh(S)
            LS = (V * np.log(w)) @ V.T
            logM = Xis @ LS @ Xs
            G = G + weights[i] * (4.0 * logM @ Q)
    else:
        raise ValueError(
            f"Unknown metric {metric}. TLRotate supports only 'euclid' and 'riemann' metrics.")
    return G

def _rot_local(Xi, Yi):
    # general eig (not eigh): the eigenvector signs select the warm start
    wX, qXc = np.linalg.eig(Xi)
    iX = np.argsort(np.real(wX))[::-1]
    qX = np.real(qXc)[:, iX]
    wY, qYc = np.linalg.eig(Yi)
    iY = np.argsort(np.real(wY))[::-1]
    qY = np.real(qYc)[:, iY]
    return qX @ qY.T

def _rot_warm_start(X, Y, weights, metric: str):
    best = 0
    bestl = 0.0
    cand0 = _rot_local(X[0], Y[0])
    cand = cand0
    for i in range(X.shape[0]):
        c = _rot_local(X[i], Y[i])
        l = float(_rot_loss(c, X, Y, weights, metric))
        if i == 0 or l < bestl:
            bestl = l
            cand = c
    return cand

def _rot_minimize(Q_ini, X_source, X_target, weights, metric: str, tol_step: float,
                  maxiter: int, maxiter_linesearch: int = 32):
    Q_1 = Q_ini.copy()
    Q = Q_ini.copy()
    F = float(_rot_loss(Q_ini, X_target, X_source, weights, metric))
    for _ in range(maxiter):
        F_1 = float(_rot_loss(Q_1, X_target, X_source, weights, metric))
        direction = _rot_project(Q_1, _rot_grad(Q_1, X_target, X_source, weights, metric))
        alpha = 1.0
        Q = _rot_retract(Q_1, -alpha * direction)
        F = float(_rot_loss(Q, X_target, X_source, weights, metric))
        dn = float(np.linalg.norm(direction))
        for _ in range(maxiter_linesearch):
            if F_1 - F > 1e-4 * alpha * dn * dn:
                break
            alpha = 0.5 * alpha
            Q = _rot_retract(Q_1, -alpha * direction)
            F = float(_rot_loss(Q, X_target, X_source, weights, metric))
        crit = float(np.linalg.norm(-alpha * direction))
        if crit <= tol_step:
            break
        Q_1 = Q
    return Q, F

def _swap01(Q):
    Z = Q.copy()
    Z[0, :] = Q[1, :]
    Z[1, :] = Q[0, :]
    return Z

def _get_rotation_manifold(X_source, X_target, weights, metric: str, tol_step: float, maxiter: int):
    if X_source.shape[0] != X_target.shape[0]:
        raise ValueError("Number of matrices in each domain doesn't match.")
    n = X_source.shape[0]
    if weights.shape[0] == 0:
        w = check_weights(None, n)
    else:
        w = check_weights(weights, n)
    Q_ini = _rot_warm_start(X_target, X_source, w, metric)
    if np.linalg.det(Q_ini) < 0:
        Q_ini = _swap01(Q_ini)
    Qp, Fp = _rot_minimize(Q_ini, X_source, X_target, w, metric, tol_step, maxiter)
    Qn, Fn = _rot_minimize(_swap01(Q_ini), X_source, X_target, w, metric, tol_step, maxiter)
    if Fp < Fn:
        return Qp
    return Qn

class TLRotate:
    # Manifold (3-D SPD input) variant only.
    target_domain: str
    metric: str
    tol_step: float
    maxiter: int
    weights: np.ndarray[float, 1]
    rotations_: Dict[str, np.ndarray[float, 2]]

    def __init__(self, target_domain: str, weights = None, metric: str = "euclid",
                 n_jobs: int = 1, expl_var: float = 0.999, n_components: int = 1,
                 n_clusters: int = 3, tol_step: float = 1e-9, maxiter: int = 10000):
        self.target_domain = target_domain
        self.metric = metric
        self.tol_step = tol_step
        self.maxiter = maxiter
        self.weights = np.empty(0) if weights is None else np.asarray(weights, dtype=float)
        self.rotations_ = Dict[str, np.ndarray[float, 2]]()

    def _class_means(self, X, lab: List[str], sw, idx):
        classes = _unique_sorted([lab[i] for i in range(len(lab)) if idx[i]])
        parts = []
        for c in classes:
            m = np.array([idx[i] and lab[i] == c for i in range(len(lab))])
            parts.append(gmean(X[m], metric="riemann", sample_weight=sw[m]))
        return np.stack(parts)

    def fit(self, X, y_enc, sample_weight = None):
        X = np.asarray(X, dtype=float)
        _, lab, dom = decode_domains(X, y_enc)
        sw = check_weights(sample_weight, X.shape[0])
        tidx = _mask(dom, self.target_domain)
        M_target = self._class_means(X, lab, sw, tidx)
        self.rotations_ = Dict[str, np.ndarray[float, 2]]()
        for d in _unique_sorted(dom):
            if d == self.target_domain:
                continue
            M_src = self._class_means(X, lab, sw, _mask(dom, d))
            self.rotations_[d] = _get_rotation_manifold(
                M_src, M_target, self.weights, self.metric, self.tol_step, self.maxiter)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)

    def fit_transform(self, X, y_enc, sample_weight = None):
        X = np.asarray(X, dtype=float)
        self.fit(X, y_enc, sample_weight=sample_weight)
        _, _, dom = decode_domains(X, y_enc)
        out = np.zeros(X.shape)
        for d in _unique_sorted(dom):
            idx = _mask(dom, d)
            if d == self.target_domain:
                out[idx] = X[idx]
            else:
                R = self.rotations_[d]
                Xd = X[idx]
                Xo = np.empty(Xd.shape)
                for i in range(Xd.shape[0]):
                    Xo[i] = R @ Xd[i] @ R.T
                out[idx] = Xo
        return out

def _decode_int_labels(y_enc: List[str]):
    out = np.empty(len(y_enc), dtype=int)
    for i in range(len(y_enc)):
        parts = y_enc[i].split("/")
        out[i] = int(parts[len(parts) - 1])
    return out

def _domain_weights(dom: List[str], domain_weight: Dict[str, float]):
    w = np.zeros(len(dom))
    for i in range(len(dom)):
        w[i] = domain_weight[dom[i]]
    return w

# TLClassifier / TLRegressor: wrap a classifier / regressor, training on the
# decoded labels of all domains (integer labels for the classifier).
class TLClassifier[EST]:
    target_domain: str
    estimator: EST
    domain_weight: Optional[Dict[str, float]]

    def __init__(self, target_domain: str, estimator: EST,
                 domain_weight: Optional[Dict[str, float]] = None):
        self.target_domain = target_domain
        self.estimator = estimator
        self.domain_weight = domain_weight

    def fit(self, X, y_enc: List[str]):
        X = np.asarray(X, dtype=float)
        _, _, dom = decode_domains(X, y_enc)
        y_dec = _decode_int_labels(y_enc)
        dw = self.domain_weight
        if dw is None:
            self.estimator.fit(X, y_dec)
        else:
            self.estimator.fit(X, y_dec, sample_weight=_domain_weights(dom, dw))
        return self

    def predict(self, X):
        return self.estimator.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self.estimator.predict_proba(np.asarray(X, dtype=float))

    def score(self, X, y_enc: List[str]):
        y_true = _decode_int_labels(y_enc)
        y_pred = np.asarray(self.predict(X), dtype=int)
        return float(np.mean(np.asarray(y_pred == y_true, dtype=float)))

class TLRegressor[EST]:
    target_domain: str
    estimator: EST
    domain_weight: Optional[Dict[str, float]]

    def __init__(self, target_domain: str, estimator: EST,
                 domain_weight: Optional[Dict[str, float]] = None):
        self.target_domain = target_domain
        self.estimator = estimator
        self.domain_weight = domain_weight

    def _targets(self, y_enc: List[str]):
        out = np.empty(len(y_enc))
        for i in range(len(y_enc)):
            parts = y_enc[i].split("/")
            out[i] = float(parts[len(parts) - 1])
        return out

    def fit(self, X, y_enc: List[str]):
        X = np.asarray(X, dtype=float)
        _, _, dom = decode_domains(X, y_enc)
        y_dec = self._targets(y_enc)
        dw = self.domain_weight
        if dw is None:
            self.estimator.fit(X, y_dec)
        else:
            self.estimator.fit(X, y_dec, sample_weight=_domain_weights(dom, dw))
        return self

    def predict(self, X):
        return self.estimator.predict(np.asarray(X, dtype=float))

    def score(self, X, y_enc: List[str]):
        y_true = self._targets(y_enc)
        y_pred = np.asarray(self.predict(X), dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1.0 - ss_res / ss_tot)

class MDWM(MDM):
    domain_tradeoff: float
    target_domain: str
    source_means_: np.ndarray[float, 3]
    target_means_: np.ndarray[float, 3]

    def __init__(self, domain_tradeoff: float, target_domain: str,
                 metric: str = "riemann", n_jobs: int = 1):
        super().__init__(metric, n_jobs)
        self.domain_tradeoff = domain_tradeoff
        self.target_domain = target_domain
        self.source_means_ = np.empty((0, 0, 0))
        self.target_means_ = np.empty((0, 0, 0))

    def fit(self, X, y_enc: List[str], sample_weight = None):
        if not (0.0 <= self.domain_tradeoff <= 1.0):
            raise ValueError(
                f"Parameter domain_tradeoff must be included in [0, 1] (Got {self.domain_tradeoff})")
        X = np.asarray(X, dtype=float)
        _, _, dom = decode_domains(X, y_enc)
        y_dec = _decode_int_labels(y_enc)
        tmask = _mask(dom, self.target_domain)
        smask = ~tmask
        X_src = X[smask]
        y_src = y_dec[smask]
        X_tgt = X[tmask]
        y_tgt = y_dec[tmask]
        self.classes_ = np.unique(y_src)
        if self.domain_tradeoff != 0.0:
            cls_t = np.unique(y_tgt)
            same = cls_t.shape[0] == self.classes_.shape[0]
            if same:
                same = bool(np.all(cls_t == self.classes_))
            if not same:
                raise ValueError("Classes in source domain must match classes in target domain.")
        sw = check_weights(sample_weight, X_src.shape[0])
        nc = self.classes_.shape[0]
        n = X.shape[1]
        self.source_means_ = np.empty((nc, n, n))
        self.target_means_ = np.empty((nc, n, n))
        for k in range(nc):
            ms = y_src == self.classes_[k]
            self.source_means_[k] = gmean(X_src[ms], metric=self.metric, sample_weight=sw[ms])
            mt = y_tgt == self.classes_[k]
            self.target_means_[k] = gmean(X_tgt[mt], metric=self.metric)
        self.covmeans_ = geodesic(self.source_means_, self.target_means_,
                                  self.domain_tradeoff, metric=self.metric)
        return self

    def score(self, X, y_enc: List[str], sample_weight = None):
        y_true = _decode_int_labels(y_enc)
        return MDM.score(self, X, y_true, sample_weight=sample_weight)
