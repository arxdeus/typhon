# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# MLPClassifier / MLPRegressor: solvers 'adam', 'sgd' (constant / invscaling /
# adaptive, momentum, Nesterov) and 'lbfgs' (scipy L-BFGS-B), activations
# identity / logistic / tanh / relu, early stopping, sample_weight. Same
# legacy RandomState stream as sklearn (Glorot-uniform init, per-epoch
# shuffles, validation split). Dense input; integer class labels.

from ..base import estimator_repr as _estimator_repr
import math
import numpy as np
from numpy.random.legacy import RandomState, check_random_state

_EPS = 2.220446049250313e-16

def _act(Z: np.ndarray[float, 2], name: str):
    if name == "identity":
        pass
    elif name == "exp":
        np.exp(Z, out=Z)
    elif name == "logistic":
        from scipy.special import expit
        Z[:, :] = expit(Z)
    elif name == "tanh":
        np.tanh(Z, out=Z)
    elif name == "relu":
        np.maximum(Z, 0.0, out=Z)
    elif name == "softmax":
        tmp = Z - Z.max(axis=1)[:, None]
        np.exp(tmp, out=Z)
        Z /= Z.sum(axis=1)[:, None]
    else:
        raise ValueError(f"unknown activation {name}")

def _deriv(Z: np.ndarray[float, 2], delta: np.ndarray[float, 2], name: str):
    if name == "logistic":
        delta *= Z
        delta *= 1 - Z
    elif name == "tanh":
        delta *= 1 - Z ** 2
    elif name == "relu":
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if Z[i, j] == 0:
                    delta[i, j] = 0.0

def _avg0(A: np.ndarray[float, 2], sw: Optional[np.ndarray[float, 1]]) -> np.ndarray[float, 1]:
    if sw is None:
        return np.average(A, axis=0)
    else:
        return np.average(A, weights=sw.__val__(), axis=0)

def _loss(name: str, Y: np.ndarray[float, 2], P: np.ndarray[float, 2], sw: Optional[np.ndarray[float, 1]]) -> float:
    from scipy.special import xlogy
    if name == "squared_error":
        return 0.5 * float(_avg0((Y - P) ** 2, sw).mean())
    elif name == "poisson":
        return float(_avg0(xlogy(Y, Y / P) - Y + P, sw).sum())
    elif name == "binary_log_loss":
        Pc = np.clip(P, _EPS, 1 - _EPS)
        return -float(_avg0(xlogy(Y, Pc) + xlogy(1 - Y, 1 - Pc), sw).sum())
    else:
        Pc = np.clip(P, _EPS, 1 - _EPS)
        if Pc.shape[1] == 1:
            Pc = np.hstack((1 - Pc, Pc))
        Yc = Y
        if Yc.shape[1] == 1:
            Yc = np.hstack((1 - Y, Y))
        return -float(_avg0(xlogy(Yc, Pc), sw).sum())

class _Opt:
    """SGDOptimizer / AdamOptimizer."""
    kind: str
    lr_init: float
    lr: float
    lr_schedule: str
    momentum: float
    nesterov: bool
    power_t: float
    beta_1: float
    beta_2: float
    epsilon: float
    t: int
    vel: List[np.ndarray[float, 2]]
    ms: List[np.ndarray[float, 2]]
    vs: List[np.ndarray[float, 2]]

    def __init__(self, kind: str, params: List[np.ndarray[float, 2]], lr_init: float, lr_schedule: str,
                 momentum: float, nesterov: bool, power_t: float, beta_1: float, beta_2: float,
                 epsilon: float):
        self.kind = kind
        self.lr_init = lr_init
        self.lr = lr_init
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.nesterov = nesterov
        self.power_t = power_t
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.vel = [np.zeros_like(p) for p in params]
        self.ms = [np.zeros_like(p) for p in params]
        self.vs = [np.zeros_like(p) for p in params]

    def update(self, params: List[np.ndarray[float, 2]], grads: List[np.ndarray[float, 2]]):
        ups = List[np.ndarray[float, 2]]()
        if self.kind == "sgd":
            ups = [self.momentum * self.vel[i] - self.lr * grads[i] for i in range(len(grads))]
            self.vel = ups
            if self.nesterov:
                ups = [self.momentum * self.vel[i] - self.lr * grads[i] for i in range(len(grads))]
        else:
            self.t += 1
            self.ms = [self.beta_1 * self.ms[i] + (1 - self.beta_1) * grads[i] for i in range(len(grads))]
            self.vs = [self.beta_2 * self.vs[i] + (1 - self.beta_2) * grads[i] ** 2 for i in range(len(grads))]
            self.lr = self.lr_init * math.sqrt(1 - self.beta_2 ** self.t) / (1 - self.beta_1 ** self.t)
            ups = [-self.lr * self.ms[i] / (np.sqrt(self.vs[i]) + self.epsilon) for i in range(len(grads))]
        for i in range(len(params)):
            params[i] += ups[i]

    def iteration_ends(self, time_step: int):
        if self.kind == "sgd" and self.lr_schedule == "invscaling":
            self.lr = float(self.lr_init) / (time_step + 1) ** self.power_t

    def trigger_stopping(self) -> bool:
        if self.kind != "sgd" or self.lr_schedule != "adaptive":
            return True
        if self.lr <= 1e-6:
            return True
        self.lr /= 5.0
        return False

class _MLPFG[M]:
    m: M
    X: np.ndarray[float, 2]
    Y: np.ndarray[float, 2]
    sw: Optional[np.ndarray[float, 1]]
    _x: np.ndarray[float, 1]
    _g: np.ndarray[float, 1]
    _f: float
    _has: bool

    def __init__(self, m: M, X: np.ndarray[float, 2], Y: np.ndarray[float, 2], sw: Optional[np.ndarray[float, 1]]):
        self.m = m; self.X = X; self.Y = Y; self.sw = sw
        self._x = np.empty(0); self._g = np.empty(0); self._f = 0.0; self._has = False

    def _eval(self, x):
        xa = np.asarray(x, dtype=float)
        if self._has:
            same = True
            for i in range(xa.shape[0]):
                if self._x[i] != xa[i]:
                    same = False
                    break
            if same:
                return
        self.m._unpack(xa)
        l, cg, ig = self.m._backprop(self.X, self.Y, self.sw)
        self._f = l
        self._g = self.m._pack(cg, ig)
        self._x = xa.copy()
        self._has = True

    def fun(self, x) -> float:
        self._eval(x)
        return self._f

    def grad(self, x):
        self._eval(x)
        return self._g.copy()

class _BaseMLP:
    hidden_layer_sizes: List[int]
    activation: str
    solver: str
    alpha: float
    batch_size: int
    learning_rate: str
    learning_rate_init: float
    power_t: float
    max_iter: int
    shuffle: bool
    random_state: Optional[int]
    tol: float
    momentum: float
    nesterovs_momentum: bool
    early_stopping: bool
    validation_fraction: float
    beta_1: float
    beta_2: float
    epsilon: float
    n_iter_no_change: int
    max_fun: int
    loss: str
    coefs_: List[np.ndarray[float, 2]]
    intercepts_: List[np.ndarray[float, 1]]
    n_layers_: int
    n_outputs_: int
    n_iter_: int
    t_: int
    loss_: float
    loss_curve_: List[float]
    validation_scores_: List[float]
    best_validation_score_: float
    best_loss_: float
    out_activation_: str
    _rs: RandomState
    _no_improve: int
    _best_coefs: List[np.ndarray[float, 2]]
    _best_ints: List[np.ndarray[float, 1]]

    def _init_mlp(self, hidden_layer_sizes, activation: str, solver: str, alpha: float, batch_size,
                  learning_rate: str, learning_rate_init: float, power_t: float, max_iter: int,
                  shuffle: bool, random_state: Optional[int], tol: float, momentum: float,
                  nesterovs_momentum: bool, early_stopping: bool, validation_fraction: float,
                  beta_1: float, beta_2: float, epsilon: float, n_iter_no_change: int,
                  max_fun: int, loss: str):
        if isinstance(hidden_layer_sizes, int):
            self.hidden_layer_sizes = [hidden_layer_sizes]
        else:
            self.hidden_layer_sizes = [int(v) for v in hidden_layer_sizes]
        for h in self.hidden_layer_sizes:
            if h <= 0:
                raise ValueError(f"hidden_layer_sizes must be > 0, got {self.hidden_layer_sizes}.")
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = -1 if isinstance(batch_size, str) else int(batch_size)
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.loss = loss
        self.coefs_ = []
        self.intercepts_ = []
        self.n_layers_ = 0
        self.n_outputs_ = 0
        self.n_iter_ = 0
        self.t_ = 0
        self.loss_ = 0.0
        self.loss_curve_ = []
        self.validation_scores_ = []
        self.best_validation_score_ = -np.inf
        self.best_loss_ = np.inf
        self.out_activation_ = "identity"
        self._rs = RandomState(0)
        self._no_improve = 0
        self._best_coefs = []
        self._best_ints = []

    def _forward(self, X: np.ndarray[float, 2]) -> List[np.ndarray[float, 2]]:
        acts = [X]
        for i in range(self.n_layers_ - 1):
            Z = acts[i] @ self.coefs_[i]
            Z += self.intercepts_[i]
            if i + 1 != self.n_layers_ - 1:
                _act(Z, self.activation)
            acts.append(Z)
        _act(acts[len(acts) - 1], self.out_activation_)
        return acts

    def _forward_fast(self, X) -> np.ndarray[float, 2]:
        A = np.ascontiguousarray(np.asarray(X, dtype=float))
        acts = self._forward(A)
        return acts[len(acts) - 1]

    def _loss_name(self) -> str:
        if self.loss == "log_loss" and self.out_activation_ == "logistic":
            return "binary_log_loss"
        return self.loss

    def _backprop(self, X: np.ndarray[float, 2], Y: np.ndarray[float, 2], sw: Optional[np.ndarray[float, 1]]):
        n = X.shape[0]
        acts = self._forward(X)
        last = self.n_layers_ - 2
        out = acts[len(acts) - 1]
        loss = _loss(self._loss_name(), Y, out, sw)
        values = 0.0
        for c in self.coefs_:
            r = c.ravel()
            values += float(np.dot(r, r))
        sw_sum = float(n) if sw is None else float(sw.__val__().sum())
        loss += 0.5 * self.alpha * values / sw_sum
        deltas = [np.empty((0, 0)) for _ in range(self.n_layers_ - 1)]
        cg = [np.empty((0, 0)) for _ in range(self.n_layers_ - 1)]
        ig = [np.empty(0) for _ in range(self.n_layers_ - 1)]
        deltas[last] = out - Y
        if sw is not None:
            deltas[last] *= sw.__val__().reshape(-1, 1)
        self._lgrad(last, sw_sum, acts, deltas, cg, ig)
        i = last
        while i > 0:
            deltas[i - 1] = deltas[i] @ self.coefs_[i].T
            _deriv(acts[i], deltas[i - 1], self.activation)
            self._lgrad(i - 1, sw_sum, acts, deltas, cg, ig)
            i -= 1
        return loss, cg, ig

    def _lgrad(self, layer: int, sw_sum: float, acts, deltas, cg, ig):
        g = acts[layer].T @ deltas[layer]
        g += self.alpha * self.coefs_[layer]
        g /= sw_sum
        cg[layer] = g
        ig[layer] = np.sum(deltas[layer], axis=0) / sw_sum

    def _pack(self, cs, ints) -> np.ndarray[float, 1]:
        parts = List[float]()
        for c in cs:
            for v in c.ravel():
                parts.append(float(v))
        for b in ints:
            for v in b:
                parts.append(float(v))
        return np.array(parts)

    def _unpack(self, x):
        pos = 0
        for i in range(len(self.coefs_)):
            a, b = self.coefs_[i].shape
            self.coefs_[i] = np.ascontiguousarray(np.asarray(x[pos:pos + a * b]).reshape(a, b))
            pos += a * b
        for i in range(len(self.intercepts_)):
            m = self.intercepts_[i].shape[0]
            self.intercepts_[i] = np.ascontiguousarray(x[pos:pos + m])
            pos += m

    def _initialize(self, units: List[int]):
        self.n_iter_ = 0
        self.t_ = 0
        self.n_layers_ = len(units)
        self.coefs_ = []
        self.intercepts_ = []
        factor = 2.0 if self.activation == "logistic" else 6.0
        for i in range(self.n_layers_ - 1):
            bound = np.sqrt(factor / (units[i] + units[i + 1]))
            self.coefs_.append(self._rs.uniform(-bound, bound, (units[i], units[i + 1])))
            self.intercepts_.append(self._rs.uniform(-bound, bound, units[i + 1]))
        self._best_coefs = [c.copy() for c in self.coefs_]
        self._best_ints = [b.copy() for b in self.intercepts_]
        self.loss_curve_ = []
        self._no_improve = 0
        self.validation_scores_ = []
        self.best_validation_score_ = -np.inf
        self.best_loss_ = np.inf

    def _fit(self, X: np.ndarray[float, 2], Y: np.ndarray[float, 2], sample_weight, stratify_y: Optional[np.ndarray[int, 1]]):
        n, p = X.shape
        sw: Optional[np.ndarray[float, 1]] = None
        if not isinstance(sample_weight, NoneType):
            sw = np.asarray(sample_weight, dtype=float) * np.ones(n)
        self.n_outputs_ = Y.shape[1]
        units = [p] + self.hidden_layer_sizes + [self.n_outputs_]
        self._rs = check_random_state(self.random_state)
        self._initialize(units)
        if self.solver == "lbfgs":
            from scipy.optimize import minimize
            fg = _MLPFG(self, X, Y, sw)
            x0 = self._pack(self.coefs_, self.intercepts_)
            opts = {"maxfun": float(self.max_fun), "maxiter": float(self.max_iter), "gtol": self.tol}
            res = minimize(fg.fun, x0, method="L-BFGS-B", jac=fg.grad, options=opts)
            self.n_iter_ = min(res.nit, self.max_iter)
            self.loss_ = float(res.fun)
            self._unpack(res.x)
        elif self.solver == "sgd" or self.solver == "adam":
            self._fit_stochastic(X, Y, sw, stratify_y)
        else:
            raise ValueError(f"unknown solver {self.solver}")
        return self

    def _split(self, n: int, stratify_y: Optional[np.ndarray[int, 1]]):
        # train_test_split(random_state=self._random_state, test_size=fraction)
        from ..utils._indexing import _approximate_mode
        nt = int(math.ceil(self.validation_fraction * n))
        ntr = n - nt
        rs = self._rs
        if stratify_y is None:
            perm = rs.permutation(n)
            return perm[nt:nt + ntr].copy(), perm[:nt].copy()
        ys = stratify_y.__val__()
        classes, y_ind, counts = np.unique(ys, return_inverse=True, return_counts=True)
        y_ind = np.asarray(y_ind).reshape(-1)
        order = np.argsort(y_ind, kind="mergesort")
        n_i = _approximate_mode(counts, ntr, rs)
        t_i = _approximate_mode(counts - n_i, nt, rs)
        train = List[int](); test = List[int]()
        start = 0
        for i in range(classes.shape[0]):
            ci = int(counts[i])
            cls = order[start:start + ci]
            pic = cls[rs.permutation(ci)]
            for k in range(int(n_i[i])):
                train.append(int(pic[k]))
            for k in range(int(n_i[i]), int(n_i[i]) + int(t_i[i])):
                test.append(int(pic[k]))
            start += ci
        return rs.permutation(np.array(train)), rs.permutation(np.array(test))

    def _fit_stochastic(self, X, Y, sw: Optional[np.ndarray[float, 1]], stratify_y: Optional[np.ndarray[int, 1]]):
        params = List[np.ndarray[float, 2]]()
        for c in self.coefs_:
            params.append(c)
        for b in self.intercepts_:
            params.append(b.reshape(1, -1))
        opt = _Opt(self.solver, params, self.learning_rate_init, self.learning_rate, self.momentum,
                   self.nesterovs_momentum, self.power_t, self.beta_1, self.beta_2, self.epsilon)
        Xtr = X; Ytr = Y; swtr = sw
        Xv = np.empty((0, 0)); Yv = np.empty((0, 0)); swv: Optional[np.ndarray[float, 1]] = None
        if self.early_stopping:
            tr, te = self._split(X.shape[0], stratify_y)
            Xtr = X[tr]; Ytr = Y[tr]; Xv = X[te]; Yv = Y[te]
            if sw is not None:
                swtr = sw.__val__()[tr]; swv = sw.__val__()[te]
            if Xv.shape[0] < 2:
                raise ValueError("The validation set is too small. Increase 'validation_fraction' or the size of your dataset.")
        n = Xtr.shape[0]
        idx = np.arange(n)
        bs = min(200, n) if self.batch_size < 0 else min(max(self.batch_size, 1), n)
        nL = len(self.coefs_)
        self.n_iter_ = 0
        for it in range(self.max_iter):
            if self.shuffle:
                perm = np.arange(n)
                self._rs.shuffle(perm)
                idx = idx[perm]
            acc = 0.0
            start = 0
            while start < n:
                end = min(start + bs, n)
                bi = idx[start:end] if self.shuffle else np.arange(start, end)
                Xb = np.ascontiguousarray(Xtr[bi]); Yb = np.ascontiguousarray(Ytr[bi])
                swb: Optional[np.ndarray[float, 1]] = None
                if swtr is not None:
                    swb = swtr.__val__()[bi]
                # rebind coefs_ / intercepts_ to the live parameter arrays
                for i in range(nL):
                    self.coefs_[i] = params[i]
                    self.intercepts_[i] = params[nL + i].reshape(-1)
                l, cg, ig = self._backprop(Xb, Yb, swb)
                acc += l * float(end - start)
                grads = List[np.ndarray[float, 2]]()
                for g in cg:
                    grads.append(g)
                for g in ig:
                    grads.append(g.reshape(1, -1))
                opt.update(params, grads)
                start = end
            for i in range(nL):
                self.coefs_[i] = params[i]
                self.intercepts_[i] = np.ascontiguousarray(params[nL + i].reshape(-1))
            self.n_iter_ += 1
            self.loss_ = acc / float(n)
            self.t_ += n
            self.loss_curve_.append(self.loss_)
            if self.early_stopping:
                vs = self._val_score(Xv, Yv, swv)
                self.validation_scores_.append(vs)
                if vs < self.best_validation_score_ + self.tol:
                    self._no_improve += 1
                else:
                    self._no_improve = 0
                if vs > self.best_validation_score_:
                    self.best_validation_score_ = vs
                    self._best_coefs = [c.copy() for c in self.coefs_]
                    self._best_ints = [b.copy() for b in self.intercepts_]
            else:
                if self.loss_ > self.best_loss_ - self.tol:
                    self._no_improve += 1
                else:
                    self._no_improve = 0
                if self.loss_ < self.best_loss_:
                    self.best_loss_ = self.loss_
            opt.iteration_ends(self.t_)
            if self._no_improve > self.n_iter_no_change:
                if opt.trigger_stopping():
                    break
                else:
                    self._no_improve = 0
        if self.early_stopping:
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_ints

class MLPClassifier(_BaseMLP):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    classes_: np.ndarray[int, 1]
    _estimator_type: ClassVar[str] = "classifier"

    def __init__(self, hidden_layer_sizes = (100,), activation: str = "relu", *, solver: str = "adam",
                 alpha: float = 1e-4, batch_size = "auto", learning_rate: str = "constant",
                 learning_rate_init: float = 1e-3, power_t: float = 0.5, max_iter: int = 200,
                 shuffle: bool = True, random_state: Optional[int] = None, tol: float = 1e-4,
                 verbose: bool = False, warm_start: bool = False, momentum: float = 0.9,
                 nesterovs_momentum: bool = True, early_stopping: bool = False,
                 validation_fraction: float = 0.1, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-8, n_iter_no_change: int = 10, max_fun: int = 15000):
        self._init_mlp(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate,
                       learning_rate_init, power_t, max_iter, shuffle, random_state, tol, momentum,
                       nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2,
                       epsilon, n_iter_no_change, max_fun, "log_loss")
        self.classes_ = np.empty(0, dtype=int)

    def fit(self, X, y, sample_weight = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=int)
        self.classes_ = np.unique(ya)
        k = self.classes_.shape[0]
        m = 1 if k <= 2 else k
        Y = np.zeros((ya.shape[0], m))
        for i in range(ya.shape[0]):
            c = int(np.searchsorted(self.classes_, ya[i]))
            if m == 1:
                Y[i, 0] = 1.0 if c == 1 else 0.0
            else:
                Y[i, c] = 1.0
        self.out_activation_ = "softmax" if k > 2 else "logistic"
        # sklearn stratifies the validation split only when n_outputs_ == 1
        if m == 1:
            return self._fit(Xa, Y, sample_weight, ya)
        else:
            none_strat: Optional[np.ndarray[int, 1]] = None
            return self._fit(Xa, Y, sample_weight, none_strat)

    def _val_score(self, Xv, Yv, swv) -> float:
        P = self._forward_fast(Xv)
        c = 0.0; t = 0.0
        for i in range(P.shape[0]):
            w = 1.0 if swv is None else swv.__val__()[i]
            ok = False
            if P.shape[1] == 1:
                ok = (P[i, 0] > 0.5) == (Yv[i, 0] > 0.5)
            else:
                ok = int(np.argmax(P[i])) == int(np.argmax(Yv[i]))
            t += w
            if ok:
                c += w
        return c / t

    def predict_proba(self, X):
        P = self._forward_fast(X)
        if P.shape[1] == 1:
            out = np.empty((P.shape[0], 2))
            for i in range(P.shape[0]):
                out[i, 0] = 1 - P[i, 0]; out[i, 1] = P[i, 0]
            return out
        return P

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        P = self._forward_fast(X)
        n = P.shape[0]
        out = np.empty(n, dtype=int)
        for i in range(n):
            if P.shape[1] == 1:
                out[i] = self.classes_[1 if P[i, 0] > 0.5 else 0]
            else:
                out[i] = self.classes_[int(np.argmax(P[i]))]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        from ..base import _accuracy
        return _accuracy(y, self.predict(X))

class MLPRegressor[D: Literal[int] = 1](_BaseMLP):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    _estimator_type: ClassVar[str] = "regressor"

    def __init__(self, loss: str = "squared_error", hidden_layer_sizes = (100,), activation: str = "relu", *, solver: str = "adam",
                 alpha: float = 1e-4, batch_size = "auto", learning_rate: str = "constant",
                 learning_rate_init: float = 1e-3, power_t: float = 0.5, max_iter: int = 200,
                 shuffle: bool = True, random_state: Optional[int] = None, tol: float = 1e-4,
                 verbose: bool = False, warm_start: bool = False, momentum: float = 0.9,
                 nesterovs_momentum: bool = True, early_stopping: bool = False,
                 validation_fraction: float = 0.1, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-8, n_iter_no_change: int = 10, max_fun: int = 15000):
        self._init_mlp(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate,
                       learning_rate_init, power_t, max_iter, shuffle, random_state, tol, momentum,
                       nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2,
                       epsilon, n_iter_no_change, max_fun, loss)

    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=float)
        if static.len(ya.shape) != D:
            from ..linear_model._base import _retype
            return _retype(self, MLPRegressor[static.len(ya.shape)]).fit(X, ya, sample_weight)
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        Y = np.ascontiguousarray(ya.reshape(ya.shape[0], -1))
        self.out_activation_ = "exp" if self.loss == "poisson" else "identity"
        none_strat: Optional[np.ndarray[int, 1]] = None
        return self._fit(Xa, Y, sample_weight, none_strat)

    def _val_score(self, Xv, Yv, swv) -> float:
        from ..metrics import r2_score
        P = self._forward_fast(Xv)
        if swv is None:
            if P.shape[1] == 1:
                return float(r2_score(np.ascontiguousarray(Yv[:, 0]), np.ascontiguousarray(P[:, 0])))
            return float(r2_score(Yv, P))
        else:
            if P.shape[1] == 1:
                return float(r2_score(np.ascontiguousarray(Yv[:, 0]), np.ascontiguousarray(P[:, 0]), sample_weight=swv.__val__()))
            return float(r2_score(Yv, P, sample_weight=swv.__val__()))

    def predict(self, X):
        P = self._forward_fast(X)
        if D == 1:
            return np.ascontiguousarray(P[:, 0])
        else:
            return P

    def score(self, X, y, sample_weight = None) -> float:
        from ..metrics import r2_score
        return float(r2_score(y, self.predict(X)))
