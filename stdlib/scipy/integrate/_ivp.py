# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# solve_ivp with the explicit Runge-Kutta methods RK23, RK45 (default) and
# DOP853, including t_eval, dense_output, events and args.
# Implicit methods Radau, BDF and LSODA live in _ivp_implicit.py / _lsoda.py.
# Deviations: complex y0, `jac_sparsity` and custom OdeSolver classes are not
# supported (ValueError). `events` may be
# a callable or a tuple/list of callables; `terminal`/`direction` are read
# from attributes of callable objects when present. When no events are given
# `t_events`/`y_events` are None (as in scipy).

import math
import numpy as np
from _dop853_coefficients import _DC, _DA, _DE3, _DE5, _DD
from _ivp_implicit import _Radau, _BDF, _LsodaSolver, _base_step

_EPS = 2.220446049250313e-16
_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0
_TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

_RK23_C = [0.0, 1 / 2, 3 / 4]
_RK23_A = [[0.0, 0.0, 0.0], [1 / 2, 0.0, 0.0], [0.0, 3 / 4, 0.0]]
_RK23_B = [2 / 9, 1 / 3, 4 / 9]
_RK23_E = [5 / 72, -1 / 12, -1 / 9, 1 / 8]
_RK23_P = [[1.0, -4 / 3, 5 / 9], [0.0, 1.0, -2 / 3], [0.0, 4 / 3, -8 / 9], [0.0, -1.0, 1.0]]

_RK45_C = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0]
_RK45_A = [[0.0, 0.0, 0.0, 0.0, 0.0],
           [1 / 5, 0.0, 0.0, 0.0, 0.0],
           [3 / 40, 9 / 40, 0.0, 0.0, 0.0],
           [44 / 45, -56 / 15, 32 / 9, 0.0, 0.0],
           [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0.0],
           [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]]
_RK45_B = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
_RK45_E = [-71 / 57600, 0.0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40]
_RK45_P = [[1.0, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
           [0.0, 0.0, 0.0, 0.0],
           [0.0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
           [0.0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
           [0.0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
           [0.0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
           [0.0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423]]

class _Interp:
    # kind: 0 = RK polynomial (Q), 1 = DOP853 (F), 2 = constant,
    # 3 = Radau (Q, n x 3), 4 = BDF (Q = D[:order+1], hh = h),
    # 5 = LSODA (Q = Nordsieck yh n x (order+1), hh = h)
    hh: float
    t_old: float
    t: float
    h: float
    y_old: np.ndarray[float, 1]
    Q: np.ndarray[float, 2]
    kind: int

    def __init__(self, t_old: float, t: float, y_old: np.ndarray[float, 1],
                 Q: np.ndarray[float, 2], kind: int):
        self.t_old = t_old
        self.t = t
        self.h = t - t_old
        self.y_old = y_old
        self.Q = Q
        self.kind = kind
        self.hh = 0.0

    def at(self, t: float) -> np.ndarray[float, 1]:
        n = self.y_old.shape[0]
        y = np.empty(n)
        if self.kind == 2:
            for j in range(n):
                y[j] = self.y_old[j]
            return y
        if self.kind == 4:
            order = self.Q.shape[0] - 1
            hh = self.hh
            p = [0.0] * order
            acc = 1.0
            for k in range(order):
                acc *= (t - (self.t - hh * k)) / (hh * (1 + k))
                p[k] = acc
            for j in range(n):
                sacc = 0.0
                for k in range(order):
                    sacc += self.Q[k + 1, j] * p[k]
                y[j] = sacc + self.Q[0, j]
            return y
        if self.kind == 5:
            m = self.Q.shape[1]
            xs = (t - self.t) / self.hh
            p = [xs ** float(k) for k in range(m)]
            for j in range(n):
                sacc = 0.0
                for k in range(m):
                    sacc += self.Q[j, k] * p[k]
                y[j] = sacc
            return y
        x = (t - self.t_old) / self.h
        if self.kind == 3:
            for j in range(n):
                y[j] = self.Q[j, 0] * x + self.Q[j, 1] * x * x + self.Q[j, 2] * x * x * x + self.y_old[j]
            return y
        if self.kind == 0:
            m = self.Q.shape[1]
            p = [0.0] * m
            acc = 1.0
            for k in range(m):
                acc *= x
                p[k] = acc
            for j in range(n):
                s = 0.0
                for k in range(m):
                    s += self.Q[j, k] * p[k]
                y[j] = self.h * s + self.y_old[j]
            return y
        # DOP853: Q holds F with shape (INTERPOLATOR_POWER, n)
        m = self.Q.shape[0]
        for j in range(n):
            v = 0.0
            for i in range(m):
                v += self.Q[m - 1 - i, j]
                if i % 2 == 0:
                    v *= x
                else:
                    v *= 1.0 - x
            y[j] = v + self.y_old[j]
        return y

class OdeSolution:
    ts: np.ndarray[float, 1]
    interpolants: List[_Interp]
    n_segments: int
    t_min: float
    t_max: float
    ascending: bool

    def __init__(self, ts: List[float], interpolants: List[_Interp]):
        self.ts = np.array(ts)
        self.interpolants = interpolants
        self.n_segments = len(interpolants)
        if len(ts) != self.n_segments + 1:
            raise ValueError("Numbers of time stamps and interpolants don't match.")
        if ts[len(ts) - 1] >= ts[0]:
            self.t_min = ts[0]
            self.t_max = ts[len(ts) - 1]
            self.ascending = True
        else:
            self.t_min = ts[len(ts) - 1]
            self.t_max = ts[0]
            self.ascending = False

    def _segment(self, t: float) -> int:
        ns = self.n_segments
        ts = self.ts
        if self.ascending:
            # searchsorted side='left'
            ind = 0
            while ind < ns + 1 and ts[ind] < t:
                ind += 1
        else:
            # ts reversed, searchsorted side='right'
            ind = 0
            while ind < ns + 1 and ts[ns - ind] <= t:
                ind += 1
        seg = min(max(ind - 1, 0), ns - 1)
        if not self.ascending:
            seg = ns - 1 - seg
        return seg

    def __call__(self, t):
        if isinstance(t, float) or isinstance(t, int):
            tf = float(t)
            return self.interpolants[self._segment(tf)].at(tf)
        else:
            ta = np.asarray(t).astype(float)
            k = ta.shape[0]
            n = self.interpolants[0].y_old.shape[0]
            out = np.empty((n, k))
            for i in range(k):
                v = self.interpolants[self._segment(ta[i])].at(ta[i])
                for j in range(n):
                    out[j, i] = v[j]
            return out

class OdeResult:
    t: np.ndarray[float, 1]
    y: np.ndarray[float, 2]
    sol: Optional[OdeSolution]
    t_events: Optional[List[np.ndarray[float, 1]]]
    y_events: Optional[List[np.ndarray[float, 2]]]
    nfev: int
    njev: int
    nlu: int
    status: int
    message: str
    success: bool

    def __init__(self, t: np.ndarray[float, 1], y: np.ndarray[float, 2],
                 sol: Optional[OdeSolution], t_events: Optional[List[np.ndarray[float, 1]]],
                 y_events: Optional[List[np.ndarray[float, 2]]], nfev: int, status: int,
                 message: str):
        self.t = t
        self.y = y
        self.sol = sol
        self.t_events = t_events
        self.y_events = y_events
        self.nfev = nfev
        self.njev = 0
        self.nlu = 0
        self.status = status
        self.message = message
        self.success = status >= 0

    def __repr__(self):
        return (f"  message: {self.message}\n  success: {self.success}\n   status: {self.status}\n"
                f"        t: {self.t}\n        y: {self.y}\n     nfev: {self.nfev}")

def _norm(x: np.ndarray[float, 1]) -> float:
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    return math.sqrt(s) / math.sqrt(float(x.shape[0]))

class _RK:
    method: int          # 0 RK23, 1 RK45, 2 DOP853
    n: int
    n_stages: int
    order_err: int
    A: List[List[float]]
    B: List[float]
    C: List[float]
    E: List[float]
    P: List[List[float]]
    t: float
    y: np.ndarray[float, 1]
    f: np.ndarray[float, 1]
    t_old: float
    y_old: np.ndarray[float, 1]
    t_bound: float
    direction: float
    h_abs: float
    h_previous: float
    max_step: float
    rtol: float
    atol: np.ndarray[float, 1]
    K: np.ndarray[float, 2]
    status: str
    nfev: int
    njev: int
    nlu: int
    error_exponent: float

    def __init__(self, method: int, t0: float, y0: np.ndarray[float, 1], t_bound: float,
                 max_step: float, rtol: float, atol: np.ndarray[float, 1]):
        self.method = method
        if method == 0:
            self.n_stages = 3
            self.order_err = 2
            self.A = _RK23_A
            self.B = _RK23_B
            self.C = _RK23_C
            self.E = _RK23_E
            self.P = _RK23_P
        elif method == 1:
            self.n_stages = 6
            self.order_err = 4
            self.A = _RK45_A
            self.B = _RK45_B
            self.C = _RK45_C
            self.E = _RK45_E
            self.P = _RK45_P
        else:
            self.n_stages = 12
            self.order_err = 7
            self.A = [[_DA[i][j] for j in range(12)] for i in range(12)]
            self.B = [_DA[12][j] for j in range(12)]
            self.C = [_DC[i] for i in range(12)]
            self.E = _DE5
            self.P = _DD
        self.n = y0.shape[0]
        self.t = t0
        self.y = y0
        self.t_old = t0
        self.y_old = y0
        self.t_bound = t_bound
        self.direction = 1.0 if t_bound - t0 >= 0 else -1.0
        if t_bound == t0:
            self.direction = 1.0
        self.max_step = max_step
        self.rtol = rtol
        self.atol = atol
        self.status = 'running'
        self.nfev = 0
        self.njev = 0
        self.nlu = 0
        self.h_previous = 0.0
        self.h_abs = 0.0
        nk = 16 if method == 2 else self.n_stages + 1
        self.K = np.zeros((nk, self.n))
        self.error_exponent = -1.0 / (self.order_err + 1)
        self.f = y0

    def fun(self, fn, t: float, y: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        self.nfev += 1
        r = np.asarray(fn(t, y)).astype(float)
        return r.reshape(self.n)

    def init_step(self, fn, first_step: Optional[float]):
        self.f = self.fun(fn, self.t, self.y)
        if first_step is not None:
            fs = first_step
            if fs <= 0:
                raise ValueError("`first_step` must be positive.")
            if fs > abs(self.t_bound - self.t):
                raise ValueError("`first_step` exceeds bounds.")
            self.h_abs = fs
            return
        # select_initial_step
        n = self.n
        if n == 0:
            self.h_abs = float('inf')
            return
        interval_length = abs(self.t_bound - self.t)
        if interval_length == 0.0:
            self.h_abs = 0.0
            return
        scale = np.empty(n)
        for i in range(n):
            scale[i] = self.atol[i] + abs(self.y[i]) * self.rtol
        d0 = _norm(self.y / scale)
        d1 = _norm(self.f / scale)
        h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        h0 = min(h0, interval_length)
        y1 = self.y + h0 * self.direction * self.f
        f1 = self.fun(fn, self.t + h0 * self.direction, y1)
        d2 = _norm((f1 - self.f) / scale) / h0
        h1 = 0.0
        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / (self.order_err + 1))
        self.h_abs = min(100 * h0, h1, interval_length, self.max_step)

    def _stage(self, fn, s: int, t: float, y: np.ndarray[float, 1], h: float,
               a: List[float], c: float):
        n = self.n
        K = self.K
        dy = np.empty(n)
        for j in range(n):
            acc = 0.0
            for k in range(s):
                acc += K[k, j] * a[k]
            dy[j] = acc * h
        fk = self.fun(fn, t + c * h, y + dy)
        for j in range(n):
            K[s, j] = fk[j]

    def _rk_step(self, fn, t: float, y: np.ndarray[float, 1], h: float):
        n = self.n
        K = self.K
        ns = self.n_stages
        for j in range(n):
            K[0, j] = self.f[j]
        for s in range(1, ns):
            self._stage(fn, s, t, y, h, self.A[s], self.C[s])
        y_new = np.empty(n)
        for j in range(n):
            acc = 0.0
            for k in range(ns):
                acc += K[k, j] * self.B[k]
            y_new[j] = y[j] + h * acc
        f_new = self.fun(fn, t + h, y_new)
        for j in range(n):
            K[ns, j] = f_new[j]
        return y_new, f_new

    def _err_norm(self, h: float, scale: np.ndarray[float, 1]) -> float:
        n = self.n
        K = self.K
        ne = self.n_stages + 1
        if self.method != 2:
            e = np.empty(n)
            for j in range(n):
                acc = 0.0
                for k in range(ne):
                    acc += K[k, j] * self.E[k]
                e[j] = acc * h / scale[j]
            return _norm(e)
        e5 = 0.0
        e3 = 0.0
        for j in range(n):
            a5 = 0.0
            a3 = 0.0
            for k in range(ne):
                a5 += K[k, j] * _DE5[k]
                a3 += K[k, j] * _DE3[k]
            a5 /= scale[j]
            a3 /= scale[j]
            e5 += a5 * a5
            e3 += a3 * a3
        if e5 == 0 and e3 == 0:
            return 0.0
        denom = e5 + 0.01 * e3
        return abs(h) * e5 / math.sqrt(denom * n)

    def _step_impl(self, fn):
        t = self.t
        y = self.y
        nxt = np.nextafter(t, self.direction * float('inf'))
        min_step = 10 * abs(nxt - t)
        h_abs = self.h_abs
        if self.h_abs > self.max_step:
            h_abs = self.max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        step_accepted = False
        step_rejected = False
        h = 0.0
        t_new = t
        y_new = y
        f_new = y
        while not step_accepted:
            if h_abs < min_step:
                return False, _TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
            h = t_new - t
            h_abs = abs(h)
            y_new, f_new = self._rk_step(fn, t, y, h)
            scale = np.empty(self.n)
            for j in range(self.n):
                scale[j] = self.atol[j] + max(abs(y[j]), abs(y_new[j])) * self.rtol
            error_norm = self._err_norm(h, scale)
            if error_norm < 1:
                factor = 0.0
                if error_norm == 0:
                    factor = _MAX_FACTOR
                else:
                    factor = min(_MAX_FACTOR, _SAFETY * error_norm ** self.error_exponent)
                if step_rejected:
                    factor = min(1.0, factor)
                h_abs *= factor
                step_accepted = True
            else:
                h_abs *= max(_MIN_FACTOR, _SAFETY * error_norm ** self.error_exponent)
                step_rejected = True
        self.h_previous = h
        self.y_old = y
        self.t = t_new
        self.y = y_new
        self.h_abs = h_abs
        self.f = f_new
        return True, ''

    def step(self, fn) -> str:
        message = ''
        if self.n == 0 or self.t == self.t_bound:
            self.t_old = self.t
            self.t = self.t_bound
            self.status = 'finished'
        else:
            t = self.t
            success, message = self._step_impl(fn)
            if not success:
                self.status = 'failed'
            else:
                self.t_old = t
                if self.direction * (self.t - self.t_bound) >= 0:
                    self.status = 'finished'
        return message

    def dense_output(self, fn) -> _Interp:
        n = self.n
        if n == 0 or self.t == self.t_old:
            return _Interp(self.t_old, self.t, self.y, np.zeros((0, 0)), 2)
        if self.method != 2:
            ne = self.n_stages + 1
            m = len(self.P[0])
            Q = np.empty((n, m))
            for j in range(n):
                for c in range(m):
                    acc = 0.0
                    for k in range(ne):
                        acc += self.K[k, j] * self.P[k][c]
                    Q[j, c] = acc
            return _Interp(self.t_old, self.t, self.y_old, Q, 0)
        K = self.K
        h = self.h_previous
        for s in range(13, 16):
            self._stage(fn, s, self.t_old, self.y_old, h, _DA[s], _DC[s])
        F = np.empty((7, n))
        for j in range(n):
            f_old = K[0, j]
            dyj = self.y[j] - self.y_old[j]
            F[0, j] = dyj
            F[1, j] = h * f_old - dyj
            F[2, j] = 2 * dyj - h * (self.f[j] + f_old)
            for r in range(4):
                acc = 0.0
                for k in range(16):
                    acc += _DD[r][k] * K[k, j]
                F[3 + r, j] = h * acc
        return _Interp(self.t_old, self.t, self.y_old, F, 1)

def _brentq(f, xa: float, xb: float, xtol: float, rtol: float, iter: int = 100) -> float:
    # port of scipy/optimize/Zeros/brentq.c
    xpre = xa
    xcur = xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0
    fpre = f(xpre)
    fcur = f(xcur)
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur
    if (fpre > 0) == (fcur > 0):
        raise ValueError("f(a) and f(b) must have different signs")
    for _ in range(iter):
        if fpre != 0 and fcur != 0 and ((fpre < 0) != (fcur < 0)):
            xblk = xpre
            fblk = fpre
            spre = xcur - xpre
            scur = spre
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre
            fpre = fcur
            fcur = fblk
            fblk = fpre
        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) < delta:
            return xcur
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            stry = 0.0
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                spre = scur
                scur = stry
            else:
                spre = sbis
                scur = sbis
        else:
            spre = sbis
            scur = sbis
        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta
        fcur = f(xcur)
    return xcur

def _no_event(t: float, y, *args) -> float:
    return 1.0

def _as_events(events):
    if isinstance(events, Tuple):
        return events
    elif isinstance(events, List):
        return events
    else:
        return (events,)

def _ev_terminal(e) -> float:
    if hasattr(e, 'terminal'):
        tv = e.terminal
        if isinstance(tv, bool):
            return 1.0 if tv else float('inf')
        tf = float(tv)
        if tf == 0:
            return float('inf')
        if int(tf) == tf and tf > 0:
            return tf
        raise ValueError("The `terminal` attribute of each event must be a boolean or positive integer.")
    return float('inf')

def _ev_direction(e) -> float:
    if hasattr(e, 'direction'):
        return float(e.direction)
    return 0.0

def _ev_values(evs, t: float, y: np.ndarray[float, 1], args) -> List[float]:
    out: List[float] = []
    for e in evs:
        out.append(float(e(t, y, *args)))
    return out

def _ev_call(evs, idx: int, t: float, y: np.ndarray[float, 1], args) -> float:
    k = 0
    r = 0.0
    for e in evs:
        if k == idx:
            r = float(e(t, y, *args))
        k += 1
    return r

_METHODS = ('RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA')

def _f1(fun, args, vectorized: Literal[bool], n: int, t: float, y: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    if vectorized:
        Y = np.empty((n, 1))
        for i in range(n):
            Y[i, 0] = y[i]
        r = np.asarray(fun(t, Y, *args)).astype(float)
        return r.reshape(n)
    r = np.asarray(fun(t, y, *args)).astype(float)
    if r.size != n:
        raise ValueError("fun returned an array of wrong size")
    return r.reshape(n)

def _fv(fun, args, vectorized: Literal[bool], n: int, t: float, Y: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    k = Y.shape[1]
    if vectorized:
        r = np.asarray(fun(t, Y, *args)).astype(float)
        return r.reshape(n, k)
    out = np.empty((n, k))
    for c in range(k):
        yc = np.empty(n)
        for i in range(n):
            yc[i] = Y[i, c]
        rc = np.asarray(fun(t, yc, *args)).astype(float).reshape(n)
        for i in range(n):
            out[i, c] = rc[i]
    return out

def _jmat(jac, args, n: int, t: float, y: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
    if jac is None:
        return np.zeros((n, n))
    elif isinstance(jac, np.ndarray) or isinstance(jac, List):
        J = np.asarray(jac).astype(float)
    else:
        J = np.asarray(jac(t, y, *args)).astype(float)
    if J.ndim != 2 or J.shape[0] != n or J.shape[1] != n:
        raise ValueError(f"`jac` is expected to have shape {(n, n)}, but actually has {J.shape}.")
    return J.reshape(n, n)

def _jac_mode(jac) -> int:
    if jac is None:
        return 0
    elif isinstance(jac, np.ndarray) or isinstance(jac, List):
        return 2
    else:
        return 1

def solve_ivp(fun, t_span, y0, method: str = 'RK45', t_eval=None, dense_output: bool = False,
              events=None, vectorized: Literal[bool] = False, args=None, first_step: Optional[float] = None,
              max_step: float = float('inf'), rtol: float = 1e-3, atol=1e-6, jac=None,
              jac_sparsity=None, min_step: float = 0.0, lband: Optional[int] = None,
              uband: Optional[int] = None):
    if method not in _METHODS:
        raise ValueError(f"`method` must be one of {_METHODS} or OdeSolver class.")
    if jac_sparsity is not None:
        raise ValueError("`jac_sparsity` is not supported")
    if lband is not None or uband is not None:
        raise ValueError("`lband`/`uband` are not supported")
    t0 = float(t_span[0])
    tf = float(t_span[1])
    if args is None:
        return _solve(fun, (), t0, tf, y0, method, t_eval, dense_output, events, first_step,
                      max_step, rtol, atol, vectorized, jac, min_step)
    else:
        if isinstance(args, Tuple):
            return _solve(fun, args, t0, tf, y0, method, t_eval, dense_output, events, first_step,
                          max_step, rtol, atol, vectorized, jac, min_step)
        else:
            raise TypeError("Supplied 'args' cannot be unpacked. Please supply `args` as a tuple")

def _solve(ufun, args, t0: float, tf: float, y0, method: str, t_eval, dense_output: bool, events,
           first_step: Optional[float], max_step: float, rtol: float, atol, vectorized: Literal[bool],
           jac, min_step: float):
    ya = np.asarray(y0).astype(float)
    if ya.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")
    yv = ya.reshape(ya.shape[0]).copy()
    n = yv.shape[0]
    for i in range(n):
        if not math.isfinite(yv[i]):
            raise ValueError("All components of the initial state `y0` must be finite.")
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")
    if rtol < 100 * _EPS:
        rtol = 100 * _EPS
    at = np.asarray(atol).astype(float)
    atv = np.empty(n)
    if at.ndim == 0:
        for i in range(n):
            atv[i] = at.item()
    else:
        if at.ndim != 1 or at.size != n:
            raise ValueError("`atol` has wrong shape.")
        af = at.reshape(n)
        for i in range(n):
            atv[i] = af[i]
    for i in range(n):
        if atv[i] < 0:
            raise ValueError("`atol` must be positive.")
    F = (lambda t, y: _f1(ufun, args, vectorized, n, t, y),
         lambda t, Y: _fv(ufun, args, vectorized, n, t, Y),
         lambda t, y: _jmat(jac, args, n, t, y))
    jm = _jac_mode(jac)
    if method == 'Radau':
        rs = _Radau(t0, yv, tf, max_step, rtol, atv, jm)
        rs.init_step(F, first_step)
        return _drive(rs, F, args, t0, tf, yv, n, t_eval, dense_output, events)
    elif method == 'BDF':
        bs = _BDF(t0, yv, tf, max_step, rtol, atv, jm)
        bs.init_step(F, first_step)
        return _drive(bs, F, args, t0, tf, yv, n, t_eval, dense_output, events)
    elif method == 'LSODA':
        ls = _LsodaSolver(t0, yv, tf, first_step, min_step, max_step, rtol, atv, jm != 0)
        return _drive(ls, F, args, t0, tf, yv, n, t_eval, dense_output, events)
    m = 1
    if method == 'RK23':
        m = 0
    elif method == 'DOP853':
        m = 2
    solver = _RK(m, t0, yv, tf, max_step, rtol, atv)
    solver.init_step(F[0], first_step)
    return _drive(solver, F, args, t0, tf, yv, n, t_eval, dense_output, events)

def _sstep(solver, F) -> str:
    if isinstance(solver, _RK):
        return solver.step(F[0])
    else:
        return _base_step(solver, F)

def _sdense(solver, F) -> _Interp:
    if isinstance(solver, _RK):
        return solver.dense_output(F[0])
    elif isinstance(solver, _Radau):
        if solver.t == solver.t_old:
            return _Interp(solver.t_old, solver.t, solver.y, np.zeros((0, 0)), 2)
        return _Interp(solver.t_old, solver.t, solver.sol_y_old, solver.sol_Q, 3)
    elif isinstance(solver, _BDF):
        if solver.t == solver.t_old:
            return _Interp(solver.t_old, solver.t, solver.y, np.zeros((0, 0)), 2)
        hh, order, Dc = solver.dense_Q()
        it = _Interp(solver.t_old, solver.t, solver.y, Dc, 4)
        it.hh = hh
        return it
    else:
        if solver.t == solver.t_old:
            return _Interp(solver.t_old, solver.t, solver.y, np.zeros((0, 0)), 2)
        hh, yh = solver.dense_data()
        it = _Interp(solver.t_old, solver.t, solver.y, yh, 5)
        it.hh = hh
        return it

def _drive(solver, F, args, t0: float, tf: float, yv: np.ndarray[float, 1], n: int, t_eval,
           dense_output: bool, events):
    has_te = t_eval is not None
    te: List[float] = []
    if t_eval is not None:
        tea = np.asarray(t_eval).astype(float)
        if tea.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")
        tel = [tea.reshape(tea.size)[i] for i in range(tea.size)]
        for v in tel:
            if v < min(t0, tf) or v > max(t0, tf):
                raise ValueError("Values in `t_eval` are not within `t_span`.")
        for i in range(len(tel) - 1):
            d = tel[i + 1] - tel[i]
            if (tf > t0 and d <= 0) or (tf < t0 and d >= 0):
                raise ValueError("Values in `t_eval` are not properly sorted.")
        te = tel if tf > t0 else tel[::-1]
    t_eval_i = 0 if (not has_te or tf > t0) else len(te)

    ts: List[float] = []
    ys: List[np.ndarray[float, 1]] = []
    ti: List[float] = [t0]
    if not has_te:
        ts.append(t0)
        ys.append(yv)
    interpolants: List[_Interp] = []

    has_ev = events is not None
    evs = _as_events(events) if events is not None else (_no_event,)
    nev = len(evs) if events is not None else 0
    max_events = [_ev_terminal(e) for e in evs] if events is not None else List[float]()
    event_dir = [_ev_direction(e) for e in evs] if events is not None else List[float]()
    event_count = [0.0] * nev
    g: List[float] = _ev_values(evs, t0, yv, args) if events is not None else List[float]()
    t_events: List[List[float]] = [List[float]() for _ in range(nev)]
    y_events: List[List[np.ndarray[float, 1]]] = [List[np.ndarray[float, 1]]() for _ in range(nev)]

    status = -2
    message = ''
    while status == -2:
        message = _sstep(solver, F)
        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break
        t_old = solver.t_old
        t = solver.t
        y = solver.y.copy()
        sol: Optional[_Interp] = None
        if dense_output:
            sol = _sdense(solver, F)
            interpolants.append(sol)
        if has_ev:
            g_new = _ev_values(evs, t, y, args)
            active: List[int] = []
            for i in range(nev):
                up = g[i] <= 0 and g_new[i] >= 0
                down = g[i] >= 0 and g_new[i] <= 0
                either = up or down
                if (up and event_dir[i] > 0) or (down and event_dir[i] < 0) or (either and event_dir[i] == 0):
                    active.append(i)
            if len(active) > 0:
                if sol is None:
                    sol = _sdense(solver, F)
                sl = sol
                for i in active:
                    event_count[i] += 1
                roots: List[float] = []
                for i in active:
                    roots.append(_brentq(lambda tt: _ev_call(evs, i, tt, sl.at(tt), args),
                                         t_old, t, 4 * _EPS, 4 * _EPS))
                terminate = False
                hit = False
                for i in active:
                    if event_count[i] >= max_events[i]:
                        hit = True
                if hit:
                    order = sorted(range(len(roots)), key=lambda k: roots[k] if t > t_old else -roots[k])
                    act2 = [active[k] for k in order]
                    roots2 = [roots[k] for k in order]
                    stop = 0
                    for k in range(len(act2)):
                        if event_count[act2[k]] >= max_events[act2[k]]:
                            stop = k
                            break
                    active = act2[:stop + 1]
                    roots = roots2[:stop + 1]
                    terminate = True
                for k in range(len(active)):
                    t_events[active[k]].append(roots[k])
                    y_events[active[k]].append(sl.at(roots[k]))
                if terminate:
                    status = 1
                    t = roots[len(roots) - 1]
                    y = sl.at(t)
            g = g_new
        if not has_te:
            donot_append = len(ts) > 1 and ts[len(ts) - 1] == t and dense_output
            if not donot_append:
                ts.append(t)
                ys.append(y)
            elif len(interpolants) > 0:
                interpolants.pop()
        else:
            step_ts: List[float] = []
            if solver.direction > 0:
                i_new = t_eval_i
                while i_new < len(te) and te[i_new] <= t:
                    i_new += 1
                step_ts = te[t_eval_i:i_new]
                t_eval_i = i_new
            else:
                i_new = t_eval_i
                while i_new > 0 and te[i_new - 1] >= t:
                    i_new -= 1
                step_ts = te[i_new:t_eval_i][::-1]
                t_eval_i = i_new
            if len(step_ts) > 0:
                if sol is None:
                    sol = _sdense(solver, F)
                for tv in step_ts:
                    ts.append(tv)
                    ys.append(sol.at(tv))
        if has_te and dense_output:
            ti.append(t)
    if status == 0:
        message = "The solver successfully reached the end of the integration interval."
    elif status == 1:
        message = "A termination event occurred."
    tarr = np.array(ts)
    yarr = np.empty((n, len(ys)))
    for k in range(len(ys)):
        for j in range(n):
            yarr[j, k] = ys[k][j]
    osol: Optional[OdeSolution] = None
    if dense_output:
        if not has_te:
            osol = OdeSolution(ts, interpolants)
        else:
            osol = OdeSolution(ti, interpolants)
    tev: List[np.ndarray[float, 1]] = []
    yev: List[np.ndarray[float, 2]] = []
    for i in range(nev):
        tev.append(np.array(t_events[i]))
        ye = np.empty((len(y_events[i]), n))
        for k in range(len(y_events[i])):
            for j in range(n):
                ye[k, j] = y_events[i][k][j]
        yev.append(ye)
    if has_ev:
        res = OdeResult(tarr, yarr, osol, tev, yev, solver.nfev, status, message)
    else:
        res = OdeResult(tarr, yarr, osol, None, None, solver.nfev, status, message)
    res.njev = solver.njev
    res.nlu = solver.nlu
    return res
