# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

# Transfer learning: domain encoding, TLDummy, TLCenter and TLScale for SPD
# matrix input (3-D arrays). TLRotate is not provided.

import numpy as np
from .geometry.base import sqrtm, invsqrtm, powm
from .geometry.distance import distance
from .geometry.mean import gmean, check_weights
from .preprocessing import Whitening

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
