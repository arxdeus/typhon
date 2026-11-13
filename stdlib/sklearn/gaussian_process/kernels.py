# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Gaussian-process kernels: ConstantKernel, WhiteKernel, RBF (isotropic or
# anisotropic), Matern (nu 0.5 / 1.5 / 2.5 / inf / general), RationalQuadratic,
# ExpSineSquared, DotProduct and the Sum / Product / Exponentiation
# combinators (built by +, *, **). Hyperparameters live in log space
# (theta / bounds) exactly as sklearn; bounds="fixed" removes a parameter.
# Each kernel provides: k(X, Y) / k(X) , k.grad(X) -> (K, dK (n, n, n_dims)),
# diag(X), theta (get/set), bounds (n_dims, 2), n_dims.

import math
import numpy as np

def _bnd(b) -> Tuple[bool, float, float]:
    if isinstance(b, str):
        if b != "fixed":
            raise ValueError("bounds must be a (low, high) pair or 'fixed'")
        return (True, 0.0, 0.0)
    else:
        return (False, float(b[0]), float(b[1]))

def _sq(X, Y) -> np.ndarray[float, 2]:
    n = X.shape[0]; m = Y.shape[0]; p = X.shape[1]
    D = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                d = X[i, k] - Y[j, k]
                s += d * d
            D[i, j] = s
    return D

def _as2(X) -> np.ndarray[float, 2]:
    A = np.asarray(X, dtype=float)
    if static.len(A.shape) == 1:
        return np.ascontiguousarray(A.reshape(1, -1))
    else:
        return np.ascontiguousarray(A)

class _Ops:
    def __add__(self, b):
        if isinstance(b, float) or isinstance(b, int):
            return Sum(self, ConstantKernel(float(b)))
        return Sum(self, b)

    def __radd__(self, b):
        return Sum(ConstantKernel(float(b)), self)

    def __mul__(self, b):
        if isinstance(b, float) or isinstance(b, int):
            return Product(self, ConstantKernel(float(b)))
        return Product(self, b)

    def __rmul__(self, b):
        return Product(ConstantKernel(float(b)), self)

    def __pow__(self, b):
        return Exponentiation(self, float(b))

    def __call__(self, X, Y = None):
        Xa = _as2(X)
        if isinstance(Y, NoneType):
            return self._k(Xa, Xa, True)
        else:
            return self._k(Xa, _as2(Y), False)

    def clone_with_theta(self, theta):
        from ..base import clone
        c = clone(self)
        c.theta = theta
        return c

class ConstantKernel(_Ops):
    constant_value: float
    _fixed: bool
    _lo: float
    _hi: float

    def __init__(self, constant_value: float = 1.0, constant_value_bounds = (1e-5, 1e5)):
        self.constant_value = constant_value
        self._fixed, self._lo, self._hi = _bnd(constant_value_bounds)

    @property
    def n_dims(self) -> int:
        return 0 if self._fixed else 1

    @property
    def theta(self):
        return np.empty(0) if self._fixed else np.array([math.log(self.constant_value)])

    @theta.setter
    def theta(self, t):
        if not self._fixed:
            self.constant_value = math.exp(float(t[0]))

    @property
    def bounds(self):
        return np.empty((0, 2)) if self._fixed else np.log(np.array([[self._lo, self._hi]]))

    def _k(self, X, Y, same: bool):
        return np.full((X.shape[0], Y.shape[0]), self.constant_value)

    def grad(self, X):
        Xa = _as2(X)
        n = Xa.shape[0]
        K = np.full((n, n), self.constant_value)
        G = np.full((n, n, self.n_dims), self.constant_value)
        return K, G

    def diag(self, X):
        return np.full(_as2(X).shape[0], self.constant_value)

    def __repr__(self) -> str:
        return f"{math.sqrt(self.constant_value):.3g}**2"

class WhiteKernel(_Ops):
    noise_level: float
    _fixed: bool
    _lo: float
    _hi: float

    def __init__(self, noise_level: float = 1.0, noise_level_bounds = (1e-5, 1e5)):
        self.noise_level = noise_level
        self._fixed, self._lo, self._hi = _bnd(noise_level_bounds)

    @property
    def n_dims(self) -> int:
        return 0 if self._fixed else 1

    @property
    def theta(self):
        return np.empty(0) if self._fixed else np.array([math.log(self.noise_level)])

    @theta.setter
    def theta(self, t):
        if not self._fixed:
            self.noise_level = math.exp(float(t[0]))

    @property
    def bounds(self):
        return np.empty((0, 2)) if self._fixed else np.log(np.array([[self._lo, self._hi]]))

    def _k(self, X, Y, same: bool):
        if same:
            return self.noise_level * np.eye(X.shape[0])
        return np.zeros((X.shape[0], Y.shape[0]))

    def grad(self, X):
        n = _as2(X).shape[0]
        K = self.noise_level * np.eye(n)
        G = np.zeros((n, n, self.n_dims))
        if not self._fixed:
            for i in range(n):
                G[i, i, 0] = self.noise_level
        return K, G

    def diag(self, X):
        return np.full(_as2(X).shape[0], self.noise_level)

class RBF(_Ops):
    length_scale: np.ndarray[float, 1]
    _aniso: bool
    _fixed: bool
    _lo: float
    _hi: float

    def __init__(self, length_scale = 1.0, length_scale_bounds = (1e-5, 1e5)):
        if isinstance(length_scale, float) or isinstance(length_scale, int):
            self.length_scale = np.array([float(length_scale)])
            self._aniso = False
        else:
            self.length_scale = np.asarray(length_scale, dtype=float).copy()
            self._aniso = self.length_scale.shape[0] > 1
        self._fixed, self._lo, self._hi = _bnd(length_scale_bounds)

    @property
    def n_dims(self) -> int:
        return 0 if self._fixed else self.length_scale.shape[0]

    @property
    def theta(self):
        return np.empty(0) if self._fixed else np.log(self.length_scale)

    @theta.setter
    def theta(self, t):
        if not self._fixed:
            ls = np.empty(self.length_scale.shape[0])
            for i in range(ls.shape[0]):
                ls[i] = math.exp(float(t[i]))
            self.length_scale = ls

    @property
    def bounds(self):
        m = self.n_dims
        b = np.empty((m, 2))
        for i in range(m):
            b[i, 0] = math.log(self._lo); b[i, 1] = math.log(self._hi)
        return b

    def _scaled(self, X):
        return X / self.length_scale if self._aniso else X / self.length_scale[0]

    def _k(self, X, Y, same: bool):
        D = _sq(self._scaled(X), self._scaled(Y))
        K = np.exp(-0.5 * D)
        if same:
            for i in range(X.shape[0]):
                K[i, i] = 1.0
        return K

    def grad(self, X):
        Xa = _as2(X)
        n = Xa.shape[0]
        D = _sq(self._scaled(Xa), self._scaled(Xa))
        for i in range(n):
            D[i, i] = 0.0
        K = np.exp(-0.5 * D)
        for i in range(n):
            K[i, i] = 1.0
        m = self.n_dims
        G = np.empty((n, n, m))
        if m == 0:
            return K, G
        if not self._aniso:
            for i in range(n):
                for j in range(n):
                    G[i, j, 0] = K[i, j] * D[i, j]
        else:
            for i in range(n):
                for j in range(n):
                    for k in range(m):
                        d = Xa[i, k] - Xa[j, k]
                        G[i, j, k] = d * d / self.length_scale[k] ** 2 * K[i, j]
        return K, G

    def diag(self, X):
        return np.ones(_as2(X).shape[0])

class Matern(RBF):
    nu: float

    def __init__(self, length_scale = 1.0, length_scale_bounds = (1e-5, 1e5), nu: float = 1.5):
        super().__init__(length_scale, length_scale_bounds)
        self.nu = nu

    def _kd(self, d: float) -> float:
        nu = self.nu
        if nu == 0.5:
            return math.exp(-d)
        elif nu == 1.5:
            k = d * math.sqrt(3)
            return (1.0 + k) * math.exp(-k)
        elif nu == 2.5:
            k = d * math.sqrt(5)
            return (1.0 + k + k ** 2 / 3.0) * math.exp(-k)
        elif nu == math.inf:
            return math.exp(-d ** 2 / 2.0)
        else:
            from scipy.special import kv, gamma
            if d == 0.0:
                d += 2.220446049250313e-16
            tmp = math.sqrt(2 * nu) * d
            return 2 ** (1.0 - nu) / float(gamma(nu)) * tmp ** nu * float(kv(nu, tmp))

    def _k(self, X, Y, same: bool):
        D = _sq(self._scaled(X), self._scaled(Y))
        K = np.empty(D.shape)
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                K[i, j] = self._kd(math.sqrt(D[i, j]))
        if same:
            for i in range(X.shape[0]):
                K[i, i] = 1.0
        return K

    def grad(self, X):
        Xa = _as2(X)
        n = Xa.shape[0]
        K = self._k(Xa, Xa, True)
        m = self.n_dims
        G = np.zeros((n, n, m))
        if m == 0:
            return K, G
        p = Xa.shape[1]
        nd = p if self._aniso else 1
        S = self._scaled(Xa)
        for i in range(n):
            for j in range(n):
                Dk = np.empty(nd)
                if self._aniso:
                    for k in range(p):
                        d = Xa[i, k] - Xa[j, k]
                        Dk[k] = d * d / self.length_scale[k] ** 2
                else:
                    s = 0.0
                    for k in range(p):
                        d = S[i, k] - S[j, k]
                        s += d * d
                    Dk[0] = s if i != j else 0.0
                tot = float(Dk.sum())
                for k in range(nd):
                    g = 0.0
                    if self.nu == 0.5:
                        den = math.sqrt(tot)
                        g = K[i, j] * (Dk[k] / den if den != 0 else 0.0)
                    elif self.nu == 1.5:
                        g = 3 * Dk[k] * math.exp(-math.sqrt(3 * tot))
                    elif self.nu == 2.5:
                        t = math.sqrt(5 * tot)
                        g = 5.0 / 3.0 * Dk[k] * (t + 1) * math.exp(-t)
                    elif self.nu == math.inf:
                        g = Dk[k] * K[i, j]
                    else:
                        raise NotImplementedError("Matern gradient for general nu is not ported")
                    G[i, j, k] = g
        return K, G

class RationalQuadratic(_Ops):
    length_scale: float
    alpha: float
    _lf: bool
    _llo: float
    _lhi: float
    _af: bool
    _alo: float
    _ahi: float

    def __init__(self, length_scale: float = 1.0, alpha: float = 1.0,
                 length_scale_bounds = (1e-5, 1e5), alpha_bounds = (1e-5, 1e5)):
        self.length_scale = length_scale
        self.alpha = alpha
        self._lf, self._llo, self._lhi = _bnd(length_scale_bounds)
        self._af, self._alo, self._ahi = _bnd(alpha_bounds)

    @property
    def n_dims(self) -> int:
        return (0 if self._af else 1) + (0 if self._lf else 1)

    # hyperparameter order (alphabetical, as sklearn): alpha, length_scale
    @property
    def theta(self):
        t = List[float]()
        if not self._af:
            t.append(math.log(self.alpha))
        if not self._lf:
            t.append(math.log(self.length_scale))
        return np.array(t) if len(t) > 0 else np.empty(0)

    @theta.setter
    def theta(self, t):
        i = 0
        if not self._af:
            self.alpha = math.exp(float(t[i])); i += 1
        if not self._lf:
            self.length_scale = math.exp(float(t[i]))

    @property
    def bounds(self):
        b = List[List[float]]()
        if not self._af:
            b.append([math.log(self._alo), math.log(self._ahi)])
        if not self._lf:
            b.append([math.log(self._llo), math.log(self._lhi)])
        out = np.empty((len(b), 2))
        for i in range(len(b)):
            out[i, 0] = b[i][0]; out[i, 1] = b[i][1]
        return out

    def _k(self, X, Y, same: bool):
        D = _sq(X, Y)
        K = (1 + D / (2 * self.alpha * self.length_scale ** 2)) ** (-self.alpha)
        if same:
            for i in range(X.shape[0]):
                K[i, i] = 1.0
        return K

    def grad(self, X):
        Xa = _as2(X)
        n = Xa.shape[0]
        D = _sq(Xa, Xa)
        for i in range(n):
            D[i, i] = 0.0
        base = 1 + D / (2 * self.alpha * self.length_scale ** 2)
        K = base ** (-self.alpha)
        for i in range(n):
            K[i, i] = 1.0
        G = np.empty((n, n, self.n_dims))
        c = 0
        if not self._af:
            G[:, :, c] = K * (-self.alpha * np.log(base) + D / (2 * self.length_scale ** 2 * base))
            c += 1
        if not self._lf:
            G[:, :, c] = D * K / (self.length_scale ** 2 * base)
        return K, G

    def diag(self, X):
        return np.ones(_as2(X).shape[0])

class ExpSineSquared(_Ops):
    length_scale: float
    periodicity: float
    _lf: bool
    _llo: float
    _lhi: float
    _pf: bool
    _plo: float
    _phi: float

    def __init__(self, length_scale: float = 1.0, periodicity: float = 1.0,
                 length_scale_bounds = (1e-5, 1e5), periodicity_bounds = (1e-5, 1e5)):
        self.length_scale = length_scale
        self.periodicity = periodicity
        self._lf, self._llo, self._lhi = _bnd(length_scale_bounds)
        self._pf, self._plo, self._phi = _bnd(periodicity_bounds)

    @property
    def n_dims(self) -> int:
        return (0 if self._lf else 1) + (0 if self._pf else 1)

    @property
    def theta(self):
        t = List[float]()
        if not self._lf:
            t.append(math.log(self.length_scale))
        if not self._pf:
            t.append(math.log(self.periodicity))
        return np.array(t) if len(t) > 0 else np.empty(0)

    @theta.setter
    def theta(self, t):
        i = 0
        if not self._lf:
            self.length_scale = math.exp(float(t[i])); i += 1
        if not self._pf:
            self.periodicity = math.exp(float(t[i]))

    @property
    def bounds(self):
        b = List[List[float]]()
        if not self._lf:
            b.append([math.log(self._llo), math.log(self._lhi)])
        if not self._pf:
            b.append([math.log(self._plo), math.log(self._phi)])
        out = np.empty((len(b), 2))
        for i in range(len(b)):
            out[i, 0] = b[i][0]; out[i, 1] = b[i][1]
        return out

    def _k(self, X, Y, same: bool):
        D = np.sqrt(_sq(X, Y))
        if same:
            for i in range(X.shape[0]):
                D[i, i] = 0.0
            arg = np.pi * D / self.periodicity
            return np.exp(-2 * (np.sin(arg) / self.length_scale) ** 2)
        return np.exp(-2 * (np.sin(np.pi / self.periodicity * D) / self.length_scale) ** 2)

    def grad(self, X):
        Xa = _as2(X)
        n = Xa.shape[0]
        D = np.sqrt(_sq(Xa, Xa))
        for i in range(n):
            D[i, i] = 0.0
        arg = np.pi * D / self.periodicity
        s = np.sin(arg)
        c = np.cos(arg)
        K = np.exp(-2 * (s / self.length_scale) ** 2)
        G = np.empty((n, n, self.n_dims))
        t = 0
        if not self._lf:
            G[:, :, t] = 4 / self.length_scale ** 2 * s ** 2 * K
            t += 1
        if not self._pf:
            G[:, :, t] = 4 * arg / self.length_scale ** 2 * c * s * K
        return K, G

    def diag(self, X):
        return np.ones(_as2(X).shape[0])

class DotProduct(_Ops):
    sigma_0: float
    _fixed: bool
    _lo: float
    _hi: float

    def __init__(self, sigma_0: float = 1.0, sigma_0_bounds = (1e-5, 1e5)):
        self.sigma_0 = sigma_0
        self._fixed, self._lo, self._hi = _bnd(sigma_0_bounds)

    @property
    def n_dims(self) -> int:
        return 0 if self._fixed else 1

    @property
    def theta(self):
        return np.empty(0) if self._fixed else np.array([math.log(self.sigma_0)])

    @theta.setter
    def theta(self, t):
        if not self._fixed:
            self.sigma_0 = math.exp(float(t[0]))

    @property
    def bounds(self):
        return np.empty((0, 2)) if self._fixed else np.log(np.array([[self._lo, self._hi]]))

    def _k(self, X, Y, same: bool):
        return np.inner(X, Y) + self.sigma_0 ** 2

    def grad(self, X):
        Xa = _as2(X)
        n = Xa.shape[0]
        K = np.inner(Xa, Xa) + self.sigma_0 ** 2
        G = np.full((n, n, self.n_dims), 2 * self.sigma_0 ** 2)
        return K, G

    def diag(self, X):
        Xa = _as2(X)
        return np.einsum("ij,ij->i", Xa, Xa) + self.sigma_0 ** 2

def _cat(a, b):
    if a.shape[0] == 0:
        return b.copy()
    if b.shape[0] == 0:
        return a.copy()
    return np.concatenate((a, b))

class Sum[K1, K2](_Ops):
    k1: K1
    k2: K2

    def __init__(self, k1: K1, k2: K2):
        self.k1 = k1
        self.k2 = k2

    @property
    def n_dims(self) -> int:
        return self.k1.n_dims + self.k2.n_dims

    @property
    def theta(self):
        return _cat(self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, t):
        a = self.k1.n_dims
        self.k1.theta = np.ascontiguousarray(t[:a])
        self.k2.theta = np.ascontiguousarray(t[a:])

    @property
    def bounds(self):
        b1 = self.k1.bounds; b2 = self.k2.bounds
        if b1.shape[0] == 0:
            return b2.copy()
        if b2.shape[0] == 0:
            return b1.copy()
        return np.vstack((b1, b2))

    def _k(self, X, Y, same: bool):
        return self.k1._k(X, Y, same) + self.k2._k(X, Y, same)

    def grad(self, X):
        A, Ga = self.k1.grad(X)
        B, Gb = self.k2.grad(X)
        return A + B, np.concatenate((Ga, Gb), axis=2)

    def diag(self, X):
        return self.k1.diag(X) + self.k2.diag(X)

    def _sk_clone(self):
        from ..base import clone
        return Sum(clone(self.k1), clone(self.k2))

class Product[K1, K2](_Ops):
    k1: K1
    k2: K2

    def __init__(self, k1: K1, k2: K2):
        self.k1 = k1
        self.k2 = k2

    @property
    def n_dims(self) -> int:
        return self.k1.n_dims + self.k2.n_dims

    @property
    def theta(self):
        return _cat(self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, t):
        a = self.k1.n_dims
        self.k1.theta = np.ascontiguousarray(t[:a])
        self.k2.theta = np.ascontiguousarray(t[a:])

    @property
    def bounds(self):
        b1 = self.k1.bounds; b2 = self.k2.bounds
        if b1.shape[0] == 0:
            return b2.copy()
        if b2.shape[0] == 0:
            return b1.copy()
        return np.vstack((b1, b2))

    def _k(self, X, Y, same: bool):
        return self.k1._k(X, Y, same) * self.k2._k(X, Y, same)

    def grad(self, X):
        A, Ga = self.k1.grad(X)
        B, Gb = self.k2.grad(X)
        return A * B, np.concatenate((Ga * B[:, :, None], Gb * A[:, :, None]), axis=2)

    def diag(self, X):
        return self.k1.diag(X) * self.k2.diag(X)

    def _sk_clone(self):
        from ..base import clone
        return Product(clone(self.k1), clone(self.k2))

class Exponentiation[K](_Ops):
    kernel: K
    exponent: float

    def __init__(self, kernel: K, exponent: float):
        self.kernel = kernel
        self.exponent = exponent

    @property
    def n_dims(self) -> int:
        return self.kernel.n_dims

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, t):
        self.kernel.theta = t

    @property
    def bounds(self):
        return self.kernel.bounds

    def _k(self, X, Y, same: bool):
        return self.kernel._k(X, Y, same) ** self.exponent

    def grad(self, X):
        K, G = self.kernel.grad(X)
        G *= self.exponent * K[:, :, None] ** (self.exponent - 1)
        return K ** self.exponent, G

    def diag(self, X):
        return self.kernel.diag(X) ** self.exponent

    def _sk_clone(self):
        from ..base import clone
        return Exponentiation(clone(self.kernel), self.exponent)

C = ConstantKernel
