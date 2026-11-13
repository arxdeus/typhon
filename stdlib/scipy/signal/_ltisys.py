# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Linear time-invariant systems: tf2ss, ss2tf, zpk2ss, ss2zpk, abcd_normalize,
# cont2discrete, lsim, impulse, step, bode, freqresp, dlsim, dimpulse, dstep,
# dfreqresp, dbode, and minimal lti / dlti classes (state-space backed).
#
# Deviations: systems are SISO-or-MIMO state space internally; `lti` and
# `dlti` objects are always state-space based (scipy picks the representation
# from the constructor arguments). dlsim with an explicit `t` uses linear
# interpolation of u as scipy's make_interp_spline(k=1). freqresp/bode use the
# zpk form (as scipy does for StateSpace input).

import numpy as np
import math
from ._filter_design import normalize, tf2zpk, zpk2tf, freqz, freqz_zpk, _flist, _farr, _carr
from ._design_extra import freqs, freqs_zpk

def tf2ss(num, den):
    nb, na = normalize(num, den)
    M = nb.size
    K = na.size
    if M > K:
        raise ValueError('Improper transfer function. `num` is longer than `den`.')
    if M == 0 or K == 0:
        return (np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)))
    nn = np.zeros(K)
    for i in range(M):
        nn[K - M + i] = nb[i]
    if K == 1:
        D = np.empty((1, 1))
        D[0, 0] = nn[0]
        return (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), D)
    A = np.zeros((K - 1, K - 1))
    for j in range(K - 1):
        A[0, j] = -na[j + 1]
    for i in range(1, K - 1):
        A[i, i - 1] = 1.0
    B = np.zeros((K - 1, 1))
    B[0, 0] = 1.0
    C = np.empty((1, K - 1))
    for j in range(K - 1):
        C[0, j] = nn[j + 1] - nn[0] * na[j + 1]
    D = np.empty((1, 1))
    D[0, 0] = nn[0]
    return (A, B, C, D)

def _as2d(M):
    a = np.asarray(M).astype(float)
    if a.ndim == 0:
        return a.reshape((1, 1))
    elif a.ndim == 1:
        return a.reshape((1, a.shape[0]))
    else:
        return a

def abcd_normalize(A = None, B = None, C = None, D = None):
    a = _as2d(A) if A is not None else np.zeros((0, 0))
    b = _as2d(B) if B is not None else np.zeros((0, 0))
    c = _as2d(C) if C is not None else np.zeros((0, 0))
    d = _as2d(D) if D is not None else np.zeros((0, 0))
    n = a.shape[0] if a.shape[0] else (b.shape[0] if b.shape[0] else c.shape[1])
    p = b.shape[1] if b.shape[1] else d.shape[1]
    q = c.shape[0] if c.shape[0] else d.shape[0]
    if a.size == 0:
        a = np.zeros((n, n))
    if b.size == 0:
        b = np.zeros((n, p))
    if c.size == 0:
        c = np.zeros((q, n))
    if d.size == 0:
        d = np.zeros((q, p))
    if a.shape[0] != n or a.shape[1] != n:
        raise ValueError(f'Parameter A has shape {a.shape} but should be ({n}, {n})!')
    if b.shape[0] != n or b.shape[1] != p:
        raise ValueError(f'Parameter B has shape {b.shape} but should be ({n}, {p})!')
    if c.shape[0] != q or c.shape[1] != n:
        raise ValueError(f'Parameter C has shape {c.shape} but should be ({q}, {n})!')
    if d.shape[0] != q or d.shape[1] != p:
        raise ValueError(f'Parameter D has shape {d.shape} but should be ({q}, {p})!')
    return (a, b, c, d)

def _rpoly(M):
    # np.poly of a real square matrix: characteristic polynomial from the
    # eigenvalues; real when the roots come in conjugate pairs (numpy check)
    ev = np.linalg.eigvals(M).astype(complex)
    n = ev.size
    c = np.zeros(n + 1, dtype=complex)
    c[0] = complex(1.0)
    for k in range(n):
        z = ev[k]
        for i in range(k + 1, 0, -1):
            c[i] = c[i] - z * c[i - 1]
    out = np.empty(n + 1)
    for i in range(n + 1):
        out[i] = c[i].real
    return out

def ss2tf(A, B, C, D, input: int = 0):
    a, b, c, d = abcd_normalize(A, B, C, D)
    nout, nin = d.shape
    if input >= nin:
        raise ValueError('System does not have the input specified.')
    n = a.shape[0]
    den = _rpoly(a) if n > 0 else np.ones(1)
    num = np.empty((nout, n + 1))
    for k in range(nout):
        M = a.copy()
        for i in range(n):
            for j in range(n):
                M[i, j] -= b[i, input] * c[k, j]
        pk = _rpoly(M) if n > 0 else np.ones(1)
        for j in range(n + 1):
            num[k, j] = pk[j] + (d[k, input] - 1) * den[j]
    return (num, den)

def zpk2ss(z, p, k: float):
    b, a = zpk2tf(z, p, k)
    return tf2ss(b, a)

def ss2zpk(A, B, C, D, input: int = 0):
    num, den = ss2tf(A, B, C, D, input)
    return tf2zpk(num[0].copy(), den)

def _to_ss(system):
    if isinstance(system, lti) or isinstance(system, dlti):
        return (system.A, system.B, system.C, system.D)
    elif static.len(system) == 2:
        return tf2ss(system[0], system[1])
    elif static.len(system) == 3:
        return zpk2ss(system[0], system[1], float(system[2]))
    elif static.len(system) == 4:
        return abcd_normalize(system[0], system[1], system[2], system[3])
    else:
        compile_error('First argument must either be a tuple of 2 (tf), 3 (zpk), or 4 (ss) arrays.')

import internal.static as static

def _cont2discrete_ss(a, b, c, d, dt: float, method: str, alpha: Optional[float]):
    from scipy.linalg import expm, solve, block_diag
    n = a.shape[0]
    m = b.shape[1]
    if method == 'gbt' or method == 'bilinear' or method == 'tustin' or method == 'euler' \
            or method == 'forward_diff' or method == 'backward_diff':
        al = 0.0
        if method == 'gbt':
            if alpha is None:
                raise ValueError('Alpha parameter must be specified for the generalized bilinear transform (gbt) method')
            al = float(alpha)
            if al < 0 or al > 1:
                raise ValueError('Alpha parameter must be within the interval [0,1] for the gbt method')
        elif method == 'bilinear' or method == 'tustin':
            al = 0.5
        elif method == 'backward_diff':
            al = 1.0
        else:
            al = 0.0
        ima = np.eye(n) - al * dt * a
        ad = solve(ima, np.eye(n) + (1.0 - al) * dt * a)
        bd = solve(ima, dt * b)
        cd = solve(ima.T.copy(), c.T.copy()).T.copy()
        dd = d + al * (c @ bd)
        return ad, bd, cd, dd
    elif method == 'zoh':
        em = np.zeros((n + m, n + m))
        for i in range(n):
            for j in range(n):
                em[i, j] = a[i, j] * dt
            for j in range(m):
                em[i, n + j] = b[i, j] * dt
        ms = expm(em)
        return ms[:n, :n].copy(), ms[:n, n:].copy(), c.copy(), d.copy()
    elif method == 'foh':
        em = np.zeros((n + 2 * m, n + 2 * m))
        for i in range(n):
            for j in range(n):
                em[i, j] = a[i, j] * dt
            for j in range(m):
                em[i, n + j] = b[i, j] * dt
        for j in range(m):
            em[n + j, n + m + j] = 1.0
        ms = expm(em)
        ms11 = ms[:n, 0:n].copy()
        ms12 = ms[:n, n:n + m].copy()
        ms13 = ms[:n, n + m:].copy()
        return ms11, ms12 - ms13 + ms11 @ ms13, c.copy(), d + c @ ms13
    elif method == 'impulse':
        for v in d.flat:
            if abs(v) > 1e-8:
                raise ValueError('Impulse method is only applicable to strictly proper systems')
        ad = expm(a * dt)
        bd = ad @ b * dt
        return ad, bd, c.copy(), c @ b * dt
    else:
        raise ValueError(f"Unknown transformation method '{method}'")

def cont2discrete(system, dt: float, method: str = 'zoh', alpha: Optional[float] = None):
    if static.len(system) == 2:
        a, b, c, d = tf2ss(system[0], system[1])
        ad, bd, cd, dd = _cont2discrete_ss(a, b, c, d, dt, method, alpha)
        num, den = ss2tf(ad, bd, cd, dd)
        return (num, den, dt)
    elif static.len(system) == 3:
        a, b, c, d = zpk2ss(system[0], system[1], float(system[2]))
        ad, bd, cd, dd = _cont2discrete_ss(a, b, c, d, dt, method, alpha)
        z, p, k = ss2zpk(ad, bd, cd, dd)
        return (z, p, k, dt)
    elif static.len(system) == 4:
        a, b, c, d = abcd_normalize(system[0], system[1], system[2], system[3])
        ad, bd, cd, dd = _cont2discrete_ss(a, b, c, d, dt, method, alpha)
        return (ad, bd, cd, dd, dt)
    else:
        compile_error('First argument must either be a tuple of 2 (tf), 3 (zpk), or 4 (ss) arrays.')

class lti:
    A: np.ndarray[float, 2]
    B: np.ndarray[float, 2]
    C: np.ndarray[float, 2]
    D: np.ndarray[float, 2]

    def __init__(self, *system):
        a, b, c, d = _to_ss(system)
        self.A = a
        self.B = b
        self.C = c
        self.D = d

    @property
    def inputs(self) -> int:
        return self.D.shape[1]

    @property
    def outputs(self) -> int:
        return self.D.shape[0]

    @property
    def poles(self):
        return np.linalg.eigvals(self.A).astype(complex)

    @property
    def zeros(self):
        return ss2zpk(self.A, self.B, self.C, self.D)[0]

    def to_tf(self):
        return ss2tf(self.A, self.B, self.C, self.D)

    def to_zpk(self):
        return ss2zpk(self.A, self.B, self.C, self.D)

    def impulse(self, X0 = None, T = None, N: Optional[int] = None):
        return impulse(self, X0, T, N)

    def step(self, X0 = None, T = None, N: Optional[int] = None):
        return step(self, X0, T, N)

    def output(self, U, T, X0 = None):
        return lsim(self, U, T, X0)

    def bode(self, w = None, n: int = 100):
        return bode(self, w, n)

    def freqresp(self, w = None, n: int = 10000):
        return freqresp(self, w, n)

    def to_discrete(self, dt: float, method: str = 'zoh', alpha: Optional[float] = None):
        ad, bd, cd, dd = _cont2discrete_ss(self.A, self.B, self.C, self.D, dt, method, alpha)
        return dlti(ad, bd, cd, dd, dt=dt)

class dlti:
    A: np.ndarray[float, 2]
    B: np.ndarray[float, 2]
    C: np.ndarray[float, 2]
    D: np.ndarray[float, 2]
    dt: float

    def __init__(self, *system, dt: float = True):
        a, b, c, d = _to_ss(system)
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.dt = float(dt)

    @property
    def inputs(self) -> int:
        return self.D.shape[1]

    @property
    def outputs(self) -> int:
        return self.D.shape[0]

    @property
    def poles(self):
        return np.linalg.eigvals(self.A).astype(complex)

    def to_tf(self):
        return ss2tf(self.A, self.B, self.C, self.D)

    def to_zpk(self):
        return ss2zpk(self.A, self.B, self.C, self.D)

    def impulse(self, x0 = None, t = None, n: Optional[int] = None):
        return dimpulse(self, x0, t, n)

    def step(self, x0 = None, t = None, n: Optional[int] = None):
        return dstep(self, x0, t, n)

    def output(self, u, t = None, x0 = None):
        return dlsim(self, u, t, x0)

    def bode(self, w = None, n: int = 100):
        return dbode(self, w, n)

    def freqresp(self, w = None, n: int = 10000, whole: bool = False):
        return dfreqresp(self, w, n, whole)

# ---------------------------------------------------------------- continuous simulation

def _ss_of(system):
    if isinstance(system, lti):
        return (system.A, system.B, system.C, system.D)
    elif isinstance(system, dlti):
        compile_error('this function can only be used with continuous-time systems.')
    else:
        return _to_ss(system)

def _squeeze_out(Y):
    # np.squeeze of the (n_steps, n_outputs) output. The result type must be
    # static, so single-output systems are supported (1-D result) and
    # multi-output systems raise (deviation).
    if Y.shape[1] != 1:
        raise NotImplementedError('only single-output systems are supported')
    return Y[:, 0].copy()

def _lsim_core(A, B, C, D, U, has_u: bool, Tv, X0, interp: bool):
    from scipy.linalg import expm
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    n_steps = Tv.shape[0]
    xout = np.zeros((n_steps, n_states))
    if Tv[0] == 0:
        for j in range(n_states):
            xout[0, j] = X0[j]
    elif Tv[0] > 0:
        E = expm(A.T * Tv[0])
        for j in range(n_states):
            s = 0.0
            for i in range(n_states):
                s += X0[i] * E[i, j]
            xout[0, j] = s
    else:
        raise ValueError('Initial time must be nonnegative')
    if n_steps == 1:
        return xout @ C.T + (U @ D.T if has_u else np.zeros((1, C.shape[0]))), xout
    dt = Tv[1] - Tv[0]
    for i in range(1, n_steps):
        if not abs((Tv[i] - Tv[i - 1]) - dt) <= 1e-8 + 1e-5 * abs(dt):
            raise ValueError('Time steps are not equally spaced.')
    if not has_u:
        E = expm(A.T * dt)
        for i in range(1, n_steps):
            for j in range(n_states):
                s = 0.0
                for k in range(n_states):
                    s += xout[i - 1, k] * E[k, j]
                xout[i, j] = s
        return xout @ C.T, xout
    if U.shape[0] != n_steps:
        raise ValueError('U must have the same number of rows as elements in T.')
    if U.shape[1] != n_inputs:
        raise ValueError('System does not define that many inputs.')
    if not interp:
        M = np.zeros((n_states + n_inputs, n_states + n_inputs))
        for i in range(n_states):
            for j in range(n_states):
                M[i, j] = A[i, j] * dt
            for j in range(n_inputs):
                M[i, n_states + j] = B[i, j] * dt
        E = expm(M.T.copy())
        Ad = E[:n_states, :n_states].copy()
        Bd = E[n_states:, :n_states].copy()
        for i in range(1, n_steps):
            xout[i] = xout[i - 1] @ Ad + U[i - 1] @ Bd
    else:
        N = n_states + 2 * n_inputs
        M = np.zeros((N, N))
        for i in range(n_states):
            for j in range(n_states):
                M[i, j] = A[i, j] * dt
            for j in range(n_inputs):
                M[i, n_states + j] = B[i, j] * dt
        for j in range(n_inputs):
            M[n_states + j, n_states + n_inputs + j] = 1.0
        E = expm(M.T.copy())
        Ad = E[:n_states, :n_states].copy()
        Bd1 = E[n_states + n_inputs:, :n_states].copy()
        Bd0 = E[n_states:n_states + n_inputs, :n_states] - Bd1
        for i in range(1, n_steps):
            xout[i] = xout[i - 1] @ Ad + U[i - 1] @ Bd0 + U[i] @ Bd1
    return xout @ C.T + U @ D.T, xout

def _u2d(U, n_steps: int):
    u = np.asarray(U).astype(float)
    if u.ndim == 0:
        return np.full((n_steps, 1), float(u.item()))
    elif u.ndim == 1:
        return u.reshape((u.shape[0], 1))
    else:
        return u

def lsim(system, U, T, X0 = None, interp: bool = True):
    A, B, C, D = _ss_of(system)
    Tv = np.atleast_1d(np.asarray(T).astype(float))
    if Tv.ndim != 1:
        raise ValueError('T must be a rank-1 array.')
    n = A.shape[0]
    x0 = np.zeros(n) if X0 is None else np.atleast_1d(np.asarray(X0).astype(float))
    if U is None:
        y, x = _lsim_core(A, B, C, D, np.zeros((1, 1)), False, Tv, x0, interp)
    else:
        u2 = _u2d(U, Tv.shape[0])
        anyu = False
        for v in u2.flat:
            if v != 0.0:
                anyu = True
        y, x = _lsim_core(A, B, C, D, u2, anyu, Tv, x0, interp)
    # the state trajectory is always (n_steps, n_states) (scipy squeezes it)
    return (Tv, _squeeze_out(y), x)

def _default_response_times(A, n: int):
    vals = np.linalg.eigvals(A)
    r = math.inf
    for v in vals:
        r = min(r, abs(complex(v).real))
    if r == 0.0:
        r = 1.0
    tc = 1.0 / r
    return np.linspace(0.0, 7 * tc, n)

def impulse(system, X0 = None, T = None, N: Optional[int] = None):
    A, B, C, D = _ss_of(system)
    n = A.shape[0]
    X = np.empty(n)
    for i in range(n):
        X[i] = B[i, 0] + (0.0 if X0 is None else float(np.atleast_1d(np.asarray(X0))[i]))
    NN = 100 if N is None else N
    Tv = _default_response_times(A, NN) if T is None else np.asarray(T).astype(float)
    y, _ = _lsim_core(A, B, C, D, np.zeros((1, 1)), False, Tv, X, False)
    return (Tv, _squeeze_out(y))

def step(system, X0 = None, T = None, N: Optional[int] = None):
    A, B, C, D = _ss_of(system)
    NN = 100 if N is None else N
    Tv = _default_response_times(A, NN) if T is None else np.asarray(T).astype(float)
    n = A.shape[0]
    x0 = np.zeros(n) if X0 is None else np.atleast_1d(np.asarray(X0).astype(float))
    U = np.ones((Tv.shape[0], 1))
    y, _ = _lsim_core(A, B, C, D, U, True, Tv, x0, False)
    return (Tv, _squeeze_out(y))

def freqresp(system, w = None, n: int = 10000):
    if isinstance(system, lti):
        z, p, k = ss2zpk(system.A, system.B, system.C, system.D)
        return freqs_zpk(z, p, k, worN=(n if w is None else w))
    elif static.len(system) == 2:
        return freqs(system[0], system[1], worN=(n if w is None else w))
    elif static.len(system) == 3:
        return freqs_zpk(system[0], system[1], float(system[2]), worN=(n if w is None else w))
    else:
        A, B, C, D = _to_ss(system)
        z, p, k = ss2zpk(A, B, C, D)
        return freqs_zpk(z, p, k, worN=(n if w is None else w))

def bode(system, w = None, n: int = 100):
    w2, y = freqresp(system, w, n)
    mag = np.empty(y.size)
    ph = np.empty(y.size)
    for i in range(y.size):
        mag[i] = 20.0 * math.log10(abs(y[i]))
        ph[i] = math.atan2(y[i].imag, y[i].real)
    phase = np.unwrap(ph) * (180.0 / math.pi)
    return (w2, mag, phase)

# ---------------------------------------------------------------- discrete simulation

def _dss_of(system):
    if isinstance(system, dlti):
        return (system.A, system.B, system.C, system.D), system.dt
    elif isinstance(system, lti):
        compile_error('this function can only be used with discrete-time dlti systems.')
    elif static.len(system) == 3:
        return tf2ss(system[0], system[1]), float(system[2])
    elif static.len(system) == 4:
        return zpk2ss(system[0], system[1], float(system[2])), float(system[3])
    elif static.len(system) == 5:
        return abcd_normalize(system[0], system[1], system[2], system[3]), float(system[4])
    else:
        compile_error('discrete system must be (num, den, dt), (z, p, k, dt) or (A, B, C, D, dt)')

def _dlsim_core(A, B, C, D, dt: float, u2, tv, has_t: bool, x0):
    if has_t:
        stoptime = tv[tv.shape[0] - 1]
        out_samples = int(math.floor(stoptime / dt)) + 1
    else:
        out_samples = u2.shape[0]
        stoptime = (out_samples - 1) * dt
    n = A.shape[0]
    xout = np.zeros((out_samples, n))
    yout = np.zeros((out_samples, C.shape[0]))
    tout = np.linspace(0.0, stoptime, out_samples)
    for j in range(n):
        xout[0, j] = x0[j]
    if has_t:
        m = u2.shape[1]
        u_dt = np.empty((out_samples, m))
        for c in range(m):
            col = np.interp(tout, tv, u2[:, c].copy())
            for i in range(out_samples):
                u_dt[i, c] = col[i]
    else:
        u_dt = u2
    for i in range(out_samples - 1):
        xout[i + 1, :] = A @ xout[i, :].copy() + B @ u_dt[i, :].copy()
        yout[i, :] = C @ xout[i, :].copy() + D @ u_dt[i, :].copy()
    yout[out_samples - 1, :] = C @ xout[out_samples - 1, :].copy() + D @ u_dt[out_samples - 1, :].copy()
    return tout, yout, xout

def dlsim(system, u, t = None, x0 = None):
    ss, dt = _dss_of(system)
    A, B, C, D = ss
    u0 = np.atleast_1d(np.asarray(u).astype(float))
    u2 = u0.reshape((u0.shape[0], 1)) if u0.ndim == 1 else u0
    x0v = np.zeros(A.shape[0]) if x0 is None else np.atleast_1d(np.asarray(x0).astype(float))
    if t is None:
        tout, yout, xout = _dlsim_core(A, B, C, D, dt, u2, np.zeros(1), False, x0v)
    else:
        tout, yout, xout = _dlsim_core(A, B, C, D, dt, u2, np.asarray(t).astype(float), True, x0v)
    return (tout, yout, xout)

def dimpulse(system, x0 = None, t = None, n: Optional[int] = None):
    ss, dt = _dss_of(system)
    A, B, C, D = ss
    nn = 100 if n is None else n
    tv = np.arange(nn) * dt if t is None else np.asarray(t).astype(float)
    nin = D.shape[1]
    x0v = np.zeros(A.shape[0]) if x0 is None else np.atleast_1d(np.asarray(x0).astype(float))
    outs: List[np.ndarray[float, 2]] = []
    tout = tv
    for i in range(nin):
        u = np.zeros((tv.shape[0], nin))
        u[0, i] = 1.0
        tout, yout, _ = _dlsim_core(A, B, C, D, dt, u, tv, t is not None, x0v)
        outs.append(yout)
    return (tout, outs)

def dstep(system, x0 = None, t = None, n: Optional[int] = None):
    ss, dt = _dss_of(system)
    A, B, C, D = ss
    nn = 100 if n is None else n
    tv = np.arange(nn) * dt if t is None else np.asarray(t).astype(float)
    nin = D.shape[1]
    x0v = np.zeros(A.shape[0]) if x0 is None else np.atleast_1d(np.asarray(x0).astype(float))
    outs: List[np.ndarray[float, 2]] = []
    tout = tv
    for i in range(nin):
        u = np.zeros((tv.shape[0], nin))
        for r in range(tv.shape[0]):
            u[r, i] = 1.0
        tout, yout, _ = _dlsim_core(A, B, C, D, dt, u, tv, t is not None, x0v)
        outs.append(yout)
    return (tout, outs)

def dfreqresp(system, w = None, n: int = 10000, whole: bool = False):
    worN = n if w is None else w
    if isinstance(system, dlti):
        num, den = ss2tf(system.A, system.B, system.C, system.D)
        nl = _flist(num[0].copy())
        dl = _flist(den)
    elif static.len(system) == 4:
        return freqz_zpk(system[0], system[1], float(system[2]), worN=worN, whole=whole)
    elif static.len(system) == 3:
        nb, na = normalize(system[0], system[1])
        nl = _flist(nb)
        dl = _flist(na)
    else:
        ss, dt = _dss_of(system)
        num, den = ss2tf(ss[0], ss[1], ss[2], ss[3])
        nl = _flist(num[0].copy())
        dl = _flist(den)
    # TransferFunction._z_to_zinv: pad the shorter polynomial with leading zeros
    diff = len(nl) - len(dl)
    if diff > 0:
        dl = [0.0] * diff + dl
    elif diff < 0:
        nl = [0.0] * (-diff) + nl
    return freqz(_farr(nl), _farr(dl), worN=worN, whole=whole)

def dbode(system, w = None, n: int = 100):
    w2, y = dfreqresp(system, w, n)
    if isinstance(system, dlti):
        dt = system.dt
    else:
        dt = float(system[static.len(system) - 1])
    mag = np.empty(y.size)
    ph = np.empty(y.size)
    for i in range(y.size):
        mag[i] = 20.0 * math.log10(abs(y[i]))
        ph[i] = math.atan2(y[i].imag, y[i].real)
    phase = np.unwrap(ph) * (180.0 / math.pi)
    return (w2 / dt, mag, phase)
