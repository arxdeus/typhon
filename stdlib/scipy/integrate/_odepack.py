# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# odeint on top of the LSODA port in _lsoda.py (scipy/integrate/_odepack_py.py
# and _odepackmodule.c). `full_output=1` returns (y, infodict) where infodict
# holds hu, tcur, tolsf, tsw, nst, nfe, nje, nqu, imxer, lenrw, leniw, mused
# and message. Deviations: `printmessg` and `ixpr` print nothing; on failure
# an ODEintWarning message is printed (no warnings filtering). `ml`/`mu`
# banded Jacobians are supported (Dfun returns the packed (ml+mu+1, n) form).

import numpy as np
from _lsoda import _Lsoda, _tol_vec, _ODE_MSGS

class ODEintWarning(Exception):
    def __init__(self, message: str = ""):
        super().__init__(message)

class _OdeintInfo:
    hu: np.ndarray[float, 1]
    tcur: np.ndarray[float, 1]
    tolsf: np.ndarray[float, 1]
    tsw: np.ndarray[float, 1]
    nst: np.ndarray[int, 1]
    nfe: np.ndarray[int, 1]
    nje: np.ndarray[int, 1]
    nqu: np.ndarray[int, 1]
    imxer: int
    lenrw: int
    leniw: int
    mused: np.ndarray[int, 1]
    message: str

    def __init__(self, k: int):
        self.hu = np.zeros(k)
        self.tcur = np.zeros(k)
        self.tolsf = np.zeros(k)
        self.tsw = np.zeros(k)
        self.nst = np.zeros(k, dtype=int)
        self.nfe = np.zeros(k, dtype=int)
        self.nje = np.zeros(k, dtype=int)
        self.nqu = np.zeros(k, dtype=int)
        self.mused = np.zeros(k, dtype=int)
        self.imxer = -1
        self.lenrw = 0
        self.leniw = 0
        self.message = ''

    def __getitem__(self, key: Literal[str]):
        # dict-style access (key must be a literal string)
        if key == 'hu':
            return self.hu
        elif key == 'tcur':
            return self.tcur
        elif key == 'tolsf':
            return self.tolsf
        elif key == 'tsw':
            return self.tsw
        elif key == 'nst':
            return self.nst
        elif key == 'nfe':
            return self.nfe
        elif key == 'nje':
            return self.nje
        elif key == 'nqu':
            return self.nqu
        elif key == 'imxer':
            return self.imxer
        elif key == 'lenrw':
            return self.lenrw
        elif key == 'leniw':
            return self.leniw
        elif key == 'mused':
            return self.mused
        elif key == 'message':
            return self.message
        else:
            compile_error("unknown odeint infodict key")

def _band_to_dense(B: np.ndarray[float, 2], n: int, ml: int, mu: int) -> np.ndarray[float, 2]:
    J = np.zeros((n, n))
    for j in range(n):
        for i in range(max(0, j - mu), min(n, j + ml + 1)):
            J[i, j] = B[mu + i - j, j]
    return J

def _as_vec(r, n: int) -> np.ndarray[float, 1]:
    a = np.asarray(r).astype(float)
    if a.size != n:
        raise ValueError("The size of the array returned by func does not match the size of y0.")
    return a.reshape(n)

def _as_mat(r, rows: int, cols: int, transpose: bool) -> np.ndarray[float, 2]:
    a = np.asarray(r).astype(float)
    if a.ndim != 2 or a.size != rows * cols:
        raise ValueError("The array return by func must be a two-dimensional array.")
    if transpose:
        a2 = a.reshape(cols, rows)
        out = np.empty((rows, cols))
        for i in range(rows):
            for j in range(cols):
                out[i, j] = a2[j, i]
        return out
    return a.reshape(rows, cols).copy()

def _fcall(func, t: float, y: np.ndarray[float, 1], args, tfirst: Literal[bool], n: int):
    if tfirst:
        return _as_vec(func(t, y, *args), n)
    else:
        return _as_vec(func(y, t, *args), n)

def _jcall(Dfun, t: float, y: np.ndarray[float, 1], args, tfirst: Literal[bool], banded: bool,
           nrows: int, n: int, ml: int, mu: int, transpose: bool) -> np.ndarray[float, 2]:
    if Dfun is None:
        return np.zeros((n, n))
    else:
        r = Dfun(t, y, *args) if tfirst else Dfun(y, t, *args)
        if banded:
            return _band_to_dense(_as_mat(r, nrows, n, transpose), n, ml, mu)
        return _as_mat(r, n, n, transpose)

def odeint(func, y0, t, args=(), Dfun=None, col_deriv: int = 0, full_output: Literal[int] = 0,
           ml: Optional[int] = None, mu: Optional[int] = None, rtol=None, atol=None, tcrit=None,
           h0: float = 0.0, hmax: float = 0.0, hmin: float = 0.0, ixpr: int = 0, mxstep: int = 0,
           mxhnil: int = 0, mxordn: int = 12, mxords: int = 5, printmessg: int = 0,
           tfirst: Literal[bool] = False):
    ta = np.asarray(t).astype(float)
    if ta.ndim > 1:
        raise ValueError("Output times t must be one-dimensional.")
    nt = ta.size
    tv = ta.reshape(nt)
    inc = True
    dec = True
    for i in range(nt - 1):
        if tv[i + 1] - tv[i] < 0:
            inc = False
        if tv[i + 1] - tv[i] > 0:
            dec = False
    if not (inc or dec):
        raise ValueError("The values in t must be monotonically increasing or monotonically decreasing; repeated values are allowed.")
    ya = np.asarray(y0).astype(float)
    if ya.ndim > 1:
        raise ValueError("Initial condition y0 must be one-dimensional.")
    neq = ya.size
    y = ya.reshape(neq).copy()
    if not isinstance(args, Tuple):
        raise TypeError("Extra arguments must be in a tuple.")
    fw = lambda tt, yy: _fcall(func, tt, yy, args, tfirst, neq)
    S = _Lsoda()
    jt = 4
    if Dfun is None:
        jt = 5
    mlv = -1 if ml is None else ml
    muv = -1 if mu is None else mu
    banded = not (mlv < 0 and muv < 0)
    if not banded:
        jt -= 3
    if mlv < 0:
        mlv = 0
    if muv < 0:
        muv = 0
    S.jt = jt
    S.ml = mlv
    S.mu = muv
    transpose = col_deriv != 0
    nrows = mlv + muv + 1 if banded else neq
    jw = lambda tt, yy: _jcall(Dfun, tt, yy, args, tfirst, banded, nrows, neq, mlv, muv, transpose)
    rv, rvec = _tol_vec(rtol, neq, 1.49012e-8)
    av, avec = _tol_vec(atol, neq, 1.49012e-8)
    S.rtol = rv
    S.atol = av
    S.rtol_vec = rvec
    if h0 != 0.0 or hmax != 0.0 or hmin != 0.0 or ixpr != 0 or mxstep != 0 or mxhnil != 0 or mxordn != 0 or mxords != 0:
        S.iopt = 1
        S.opt_h0 = h0
        S.opt_hmax = hmax
        S.opt_hmin = hmin
        S.opt_ixpr = ixpr
        S.opt_mxstep = mxstep
        S.opt_mxhnil = mxhnil
        S.opt_mxordn = mxordn
        S.opt_mxords = mxords
    lmat = (neq * neq + 2) if jt <= 2 else ((2 * mlv + muv + 1) * neq + 2)
    lenrw = max(20 + neq * (mxordn + 1) + 3 * neq, 20 + neq * (mxords + 1) + 3 * neq + lmat)
    tcl: List[float] = []
    itask = 1
    if tcrit is not None:
        tca = np.asarray(tcrit).astype(float)
        tcf = tca.reshape(tca.size)
        for i in range(tca.size):
            tcl.append(tcf[i])
        itask = 4
        S.tcrit = tcl[0]
    numcrit = len(tcl)
    crit_ind = 0
    yout = np.zeros((nt, neq))
    tc = 0.0
    t0count = 0
    if nt > 0:
        tc = tv[0]
        t0count = 1
        while t0count < nt and tv[t0count] == tc:
            t0count += 1
    for k in range(t0count):
        for i in range(neq):
            yout[k, i] = y[i]
    info = _OdeintInfo(max(nt - 1, 0))
    istate = 1
    k = t0count
    while k < nt and istate > 0:
        tout = tv[k]
        if itask == 4:
            if tout > tcl[crit_ind]:
                crit_ind += 1
                if crit_ind < numcrit:
                    S.tcrit = tcl[crit_ind]
        if crit_ind >= numcrit:
            itask = 1
        tc, istate = S.run(fw, jw, y, tc, tout, itask, istate)
        if full_output:
            info.hu[k - 1] = S.hu
            info.tcur[k - 1] = S.tn
            info.tolsf[k - 1] = S.tolsf
            info.tsw[k - 1] = S.tsw
            info.nst[k - 1] = S.nst
            info.nfe[k - 1] = S.nfe
            info.nje[k - 1] = S.nje
            info.nqu[k - 1] = S.nqu
            info.imxer = S.imxer if (istate == -5 or istate == -4) else -1
            info.lenrw = lenrw
            info.leniw = 20 + neq
            info.mused[k - 1] = S.mused
        for i in range(neq):
            yout[k, i] = y[i]
        k += 1
    if istate < 0:
        print(f"ODEintWarning: {_ODE_MSGS[istate]} Run with full_output = 1 to get quantitative information.")
    elif printmessg:
        print(f"ODEintWarning: {_ODE_MSGS[istate]}")
    if full_output:
        info.message = _ODE_MSGS[istate]
        return yout, info
    else:
        return yout
