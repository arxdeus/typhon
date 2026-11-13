# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.metrics.DistanceMetric (_dist_metrics.pyx.tp, float64): one class
# holding the metric code and its parameters, with the same kernels,
# rdist/dist conversions and pairwise() as the per-metric Cython classes.

import numpy as np
import math

def _is_sp(a) -> Literal[bool]:
    return hasattr(a, "tocsr") and hasattr(a, "indptr")

def _dense2(X, name: str = "X"):
    if _is_sp(X):
        a = np.ascontiguousarray(np.asarray(X.toarray(), dtype=float))
    else:
        a = np.ascontiguousarray(np.asarray(X, dtype=float))
        if static.len(a.shape) != 2:
            raise ValueError("Expected 2D array, got 1D array instead")
    for v in a.reshape(-1):
        if not np.isfinite(v):
            raise ValueError("Input contains NaN or infinity.")
    return a

def _cd(a: float, b: float) -> float:
    if b == 0.0:
        if a == 0.0 or a != a:
            return float("nan")
        return float("inf") if a > 0.0 else -float("inf")
    return a / b

_NAMES = ["euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1", "chebyshev",
          "infinity", "seuclidean", "mahalanobis", "hamming", "canberra", "braycurtis",
          "matching", "jaccard", "dice", "kulsinski", "rogerstanimoto", "russellrao",
          "sokalmichener", "sokalsneath", "haversine"]

# codes: 0 euclid 1 manhattan 2 chebyshev 3 minkowski 4 hamming 5 canberra 6 braycurtis
# 7 jaccard 8 dice 9 rogerstanimoto 10 russellrao 11 sokalmichener 12 sokalsneath
# 13 haversine 14 seuclidean 15 mahalanobis 16 matching 17 kulsinski
def _code_of(metric: str) -> int:
    if metric == "euclidean" or metric == "l2": return 0
    if metric == "manhattan" or metric == "cityblock" or metric == "l1": return 1
    if metric == "chebyshev" or metric == "infinity": return 2
    if metric == "minkowski" or metric == "p": return 3
    if metric == "hamming": return 4
    if metric == "canberra": return 5
    if metric == "braycurtis": return 6
    if metric == "jaccard": return 7
    if metric == "dice": return 8
    if metric == "rogerstanimoto": return 9
    if metric == "russellrao": return 10
    if metric == "sokalmichener": return 11
    if metric == "sokalsneath": return 12
    if metric == "haversine": return 13
    if metric == "seuclidean": return 14
    if metric == "mahalanobis": return 15
    if metric == "matching": return 16
    if metric == "kulsinski": return 17
    raise ValueError(f"Unrecognized metric '{metric}'")

def _inv2(V):
    a = np.asarray(V, dtype=float)
    if static.len(a.shape) != 2:
        raise ValueError("V/VI must be square")
    return np.linalg.inv(a)

def _sq2(V):
    a = np.asarray(V, dtype=float)
    if static.len(a.shape) != 2:
        raise ValueError("V/VI must be square")
    return a

class DistanceMetric:
    code: int
    p: float
    vec: np.ndarray[float, 1]
    mat: np.ndarray[float, 2]
    size: int
    name: str

    def __init__(self, code: int = 0, p: float = 2.0, vec = None, mat = None, name: str = "euclidean"):
        self.code = code
        self.p = p
        self.name = name
        self.vec = np.empty(0)
        self.mat = np.empty((0, 0))
        if isinstance(vec, NoneType):
            pass
        else:
            self.vec = np.ascontiguousarray(np.asarray(vec, dtype=float).reshape(-1))
        if isinstance(mat, NoneType):
            pass
        else:
            self.mat = np.ascontiguousarray(np.asarray(mat, dtype=float))
        self.size = 0
        if code == 14:
            self.size = self.vec.shape[0]
        elif code == 15:
            self.size = self.mat.shape[0]
        elif code == 3:
            self.size = self.vec.shape[0]

    @staticmethod
    def get_metric(metric: str, dtype = None, *, p: float = 2.0, w = None, V = None, VI = None):
        """DistanceMetric.get_metric (string identifiers, float64)."""
        c = _code_of(metric)
        if c == 3:
            if p <= 0:
                raise ValueError("p must be greater than 0")
            if isinstance(w, NoneType):
                if math.isinf(p):
                    return DistanceMetric(2, p, None, None, "chebyshev")
                if p == 1:
                    return DistanceMetric(1, 1.0, None, None, "manhattan")
                if p == 2:
                    return DistanceMetric(0, 2.0, None, None, "euclidean")
                return DistanceMetric(3, p, None, None, "minkowski")
            else:
                if math.isinf(p):
                    raise ValueError("MinkowskiDistance requires finite p. For p=inf, use ChebyshevDistance.")
                wa = np.asarray(w, dtype=float).reshape(-1)
                if np.any(wa < 0):
                    raise ValueError("w cannot contain negative weights")
                return DistanceMetric(3, p, wa, None, "minkowski")
        if c == 14:
            if isinstance(V, NoneType):
                raise TypeError("SEuclideanDistance.__init__() missing 1 required positional argument: 'V'")
            else:
                return DistanceMetric(14, 2.0, V, None, metric)
        if c == 15:
            if isinstance(VI, NoneType):
                if isinstance(V, NoneType):
                    raise ValueError("Must provide either V or VI for Mahalanobis distance")
                else:
                    M0 = _inv2(V)
                    return DistanceMetric(15, 2.0, None, M0, metric)
            else:
                M1 = _sq2(VI)
                if M1.shape[0] != M1.shape[1]:
                    raise ValueError("V/VI must be square")
                return DistanceMetric(15, 2.0, None, M1, metric)
        pp = 2.0
        if c == 1:
            pp = 1.0
        elif c == 2:
            pp = float("inf")
        return DistanceMetric(c, pp, None, None, metric)

    def _validate_data(self, X: np.ndarray[float, 2]):
        if self.code == 3 and self.size > 0 and X.shape[1] != self.size:
            raise ValueError(f"MinkowskiDistance: the size of w must match the number of features ({X.shape[1]}). Currently len(w)={self.size}.")
        if self.code == 14 and X.shape[1] != self.size:
            raise ValueError("SEuclidean dist: size of V does not match")
        if self.code == 15 and X.shape[1] != self.size:
            raise ValueError("Mahalanobis dist: size of V does not match")
        if self.code == 13 and X.shape[1] != 2:
            raise ValueError("Haversine distance only valid in 2 dimensions")

    def _rd(self, A: np.ndarray[float, 2], ia: int, B: np.ndarray[float, 2], ib: int) -> float:
        c = self.code
        n = A.shape[1]
        d = 0.0
        if c == 0:
            for j in range(n):
                t = A[ia, j] - B[ib, j]
                d += t * t
            return d
        if c == 1:
            for j in range(n):
                d += abs(A[ia, j] - B[ib, j])
            return d
        if c == 2:
            for j in range(n):
                d = max(d, abs(A[ia, j] - B[ib, j]))
            return d
        if c == 3:
            if self.size > 0:
                for j in range(n):
                    d += self.vec[j] * (abs(A[ia, j] - B[ib, j]) ** self.p)
            else:
                for j in range(n):
                    d += abs(A[ia, j] - B[ib, j]) ** self.p
            return d
        if c == 4:
            ne = 0
            for j in range(n):
                if A[ia, j] != B[ib, j]:
                    ne += 1
            return float(ne) / float(n)
        if c == 5:
            for j in range(n):
                den = abs(A[ia, j]) + abs(B[ib, j])
                if den > 0.0:
                    d += abs(A[ia, j] - B[ib, j]) / den
            return d
        if c == 6:
            num = 0.0
            den = 0.0
            for j in range(n):
                num += abs(A[ia, j] - B[ib, j])
                den += abs(A[ia, j]) + abs(B[ib, j])
            if den > 0.0:
                return num / den
            return 0.0
        if c == 13:
            s0 = math.sin(0.5 * (A[ia, 0] - B[ib, 0]))
            s1 = math.sin(0.5 * (A[ia, 1] - B[ib, 1]))
            return s0 * s0 + math.cos(A[ia, 0]) * math.cos(B[ib, 0]) * s1 * s1
        if c == 14:
            for j in range(n):
                t = A[ia, j] - B[ib, j]
                d += t * t / self.vec[j]
            return d
        if c == 15:
            buf = np.empty(n)
            for j in range(n):
                buf[j] = A[ia, j] - B[ib, j]
            for i in range(n):
                tmp = 0.0
                for j in range(n):
                    tmp += self.mat[i, j] * buf[j]
                d += tmp * buf[i]
            return d
        ntt = 0
        nneq = 0
        for j in range(n):
            t1 = A[ia, j] != 0.0
            t2 = B[ib, j] != 0.0
            if t1 and t2:
                ntt += 1
            if t1 != t2:
                nneq += 1
        if c == 7:
            nnz = ntt + nneq
            if nnz == 0:
                return 0.0
            return float(nnz - ntt) * 1.0 / float(nnz)
        if c == 8:
            return _cd(float(nneq), 2.0 * float(ntt) + float(nneq))
        if c == 9 or c == 11:
            return (2.0 * float(nneq)) / (float(n) + float(nneq))
        if c == 10:
            return float(n - ntt) * 1.0 / float(n)
        if c == 12:
            return _cd(float(nneq), 0.5 * float(ntt) + float(nneq))
        if c == 16:
            return float(nneq) * 1.0 / float(n)
        return _cd(float(nneq - ntt + n) * 1.0, float(nneq + n))

    def _r2d(self, r: float) -> float:
        c = self.code
        if c == 0 or c == 14 or c == 15:
            return math.sqrt(r)
        if c == 3:
            return r ** (1.0 / self.p)
        if c == 13:
            return 2.0 * math.asin(math.sqrt(r))
        return r

    def _d2r(self, d: float) -> float:
        c = self.code
        if c == 0 or c == 14 or c == 15:
            return d * d
        if c == 3:
            return d ** self.p
        if c == 13:
            t = math.sin(0.5 * d)
            return t * t
        return d

    def rdist_to_dist(self, rdist):
        a = np.asarray(rdist, dtype=float)
        out = np.empty(a.shape)
        f = a.reshape(-1)
        o = out.reshape(-1)
        for i in range(f.shape[0]):
            o[i] = self._r2d(f[i])
        return out

    def dist_to_rdist(self, dist):
        a = np.asarray(dist, dtype=float)
        out = np.empty(a.shape)
        f = a.reshape(-1)
        o = out.reshape(-1)
        for i in range(f.shape[0]):
            o[i] = self._d2r(f[i])
        return out

    def pairwise(self, X, Y = None):
        """Pairwise distance matrix between rows of X and Y (dense or CSR input)."""
        A = _dense2(X)
        self._validate_data(A)
        if isinstance(Y, NoneType):
            n = A.shape[0]
            D = np.empty((n, n))
            for i in range(n):
                D[i, i] = self._r2d(self._rd(A, i, A, i))
                for j in range(i + 1, n):
                    d = self._r2d(self._rd(A, i, A, j))
                    D[i, j] = d
                    D[j, i] = d
            return D
        B = _dense2(Y)
        self._validate_data(B)
        if A.shape[1] != B.shape[1]:
            raise ValueError("X and Y must have the same second dimension")
        D = np.empty((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                D[i, j] = self._r2d(self._rd(A, i, B, j))
        return D

    def __repr__(self) -> str:
        return f"DistanceMetric({self.name})"
