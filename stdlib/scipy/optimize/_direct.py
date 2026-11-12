# Ported from scipy 1.18.1 (BSD-3-Clause)

# DIRECT global optimizer: line-by-line translation of scipy's C sources
# (_direct/DIRect.c, DIRsubrout.c, DIRserial.c, direct_wrap.c; f2c'd
# Fortran by J. Gablonsky, cleaned by S. G. Johnson) and _direct_py.py.
# 1-based Fortran indexing is kept: f values / flags, point, anchor (offset
# by one so that anchor[-1] is A[0]), and c / length as flat row-major
# [pos][dim] arrays.
# The user function is always defined (DIRECT "hidden constraints" flag
# is 0), so the infeasible-point paths of the C code are not reachable and
# are omitted.

import math
import numpy as np
from ._common import OptimizeResult, Bounds, _call_scalar

_ERROR_MESSAGES = ['Number of function evaluations done is larger than maxfun={}',
                   'Number of iterations is larger than maxiter={}',
                   'u[i] < l[i] for some i', 'maxfun is too large',
                   'Initialization failed',
                   'There was an error in the creation of the sample points',
                   'An error occurred while the function was sampled',
                   'Maximum number of levels has been reached.',
                   'Forced stop', 'Invalid arguments', 'Out of memory']
_SUCCESS_MESSAGES = ['The best function value found is within a relative error={} of the (known) global optimum f_min',
                     'The volume of the hyperrectangle containing the lowest function value found is below vol_tol={}',
                     'The side length measure of the hyperrectangle containing the lowest function value found is below len_tol={}']

class _Direct[F, A, CB]:
    fcn: F
    args: A
    callback: CB
    n: int
    jones: int
    maxfunc: int
    maxdeep: int
    maxdiv: int
    c: List[float]        # [pos * n + (j - 1)], pos 1-based
    length: List[int]
    fv: List[float]       # f(1, pos)
    ff: List[float]       # f(2, pos)
    point: List[int]
    anchor: List[int]     # anchor[i + 1] == A(i), i = -1..maxdeep
    s1: List[int]
    s2: List[int]
    levels: List[float]
    thirds: List[float]
    w: List[float]
    list2a: List[int]
    list2b: List[int]
    arrayi: List[int]
    xs1: List[float]      # u - l
    xs2: List[float]      # l / (u - l)
    x: List[float]
    xlast: np.ndarray[float, 1]
    minf: float
    minpos: int
    fmax: float
    free: int
    maxi: int

    def __init__(self, fcn: F, args: A, callback: CB, n: int):
        self.fcn = fcn
        self.args = args
        self.callback = callback
        self.n = n
        self.xlast = np.zeros(n)
        self.jones = 0
        self.maxfunc = 0
        self.maxdeep = 0
        self.maxdiv = 0
        self.c = List[float]()
        self.length = List[int]()
        self.fv = List[float]()
        self.ff = List[float]()
        self.point = List[int]()
        self.anchor = List[int]()
        self.s1 = List[int]()
        self.s2 = List[int]()
        self.levels = List[float]()
        self.thirds = List[float]()
        self.w = List[float]()
        self.list2a = List[int]()
        self.list2b = List[int]()
        self.arrayi = List[int]()
        self.xs1 = List[float]()
        self.xs2 = List[float]()
        self.x = List[float]()
        self.minf = 0.0
        self.minpos = 0
        self.fmax = 0.0
        self.free = 0
        self.maxi = 0

    # ------------------------------------------------------------ helpers
    def L(self, pos: int, i: int) -> int:
        return self.length[pos * self.n + i - 1]

    def setL(self, pos: int, i: int, v: int):
        self.length[pos * self.n + i - 1] = v

    def C(self, pos: int, i: int) -> float:
        return self.c[pos * self.n + i - 1]

    def setC(self, pos: int, i: int, v: float):
        self.c[pos * self.n + i - 1] = v

    def A(self, i: int) -> int:
        return self.anchor[i + 1]

    def setA(self, i: int, v: int):
        self.anchor[i + 1] = v

    def getlevel(self, pos: int, jones: int) -> int:
        n = self.n
        if jones == 0:
            help = self.L(pos, 1)
            k = help
            p = 1
            for i in range(2, n + 1):
                li = self.L(pos, i)
                if li < k:
                    k = li
                if li == help:
                    p += 1
            if k == help:
                return k * n + n - p
            return k * n + p
        help = self.L(pos, 1)
        for i in range(2, n + 1):
            if self.L(pos, i) < help:
                help = self.L(pos, i)
        return help

    def getmaxdeep(self, pos: int) -> int:
        help = self.L(pos, 1)
        for i in range(2, self.n + 1):
            help = min(help, self.L(pos, i))
        return help

    def infcn(self) -> float:
        n = self.n
        xs = np.empty(n)
        for i in range(n):
            xs[i] = (self.x[i + 1] + self.xs2[i + 1]) * self.xs1[i + 1]
        self.xlast = xs
        return _call_scalar(self.fcn, xs, self.args)

    def get_i(self, pos: int):
        n = self.n
        help = self.L(pos, 1)
        for i in range(2, n + 1):
            if self.L(pos, i) < help:
                help = self.L(pos, i)
        j = 1
        for i in range(1, n + 1):
            if self.L(pos, i) == help:
                self.arrayi[j] = i
                j += 1
        self.maxi = j - 1

    # --------------------------------------------------------- DIRchoose
    def choose(self, actdeep: int, epsrel: float, epsabs: float,
               ifeasiblef: int, jones: int) -> int:
        # returns maxpos; S stored in s1 (positions), s2 (levels)
        thirds = self.levels
        fv = self.fv
        k = 1
        if ifeasiblef >= 1:
            for j in range(0, actdeep + 1):
                if self.A(j) > 0:
                    self.s1[k] = self.A(j)
                    self.s2[k] = self.getlevel(self.s1[k], jones)
                    break
            return 1
        for j in range(0, actdeep + 1):
            if self.A(j) > 0:
                self.s1[k] = self.A(j)
                self.s2[k] = self.getlevel(self.s1[k], jones)
                k += 1
        novalue = 0
        novaluedeep = 0
        if self.A(-1) > 0:
            novalue = self.A(-1)
            novaluedeep = self.getlevel(novalue, jones)
        maxpos = k - 1
        # (C: `for (j = k-1; j <= maxdeep; ++j) s[k] = 0;`)
        if k < self.maxdiv:
            self.s1[k] = 0
        minf = self.minf
        thr = min(minf - epsrel * abs(minf), minf - epsabs)
        for j in range(maxpos, 0, -1):
            helplower = math.inf
            helpgreater = 0.0
            jj = self.s1[j]
            drop = False
            for i in range(1, j):
                ii = self.s1[i]
                if ii > 0 and i != j:
                    if self.ff[ii] <= 1.0:
                        help2 = thirds[self.s2[i]] - thirds[self.s2[j]]
                        help2 = (fv[ii] - fv[jj]) / help2
                        if help2 <= 0.0:
                            drop = True
                            break
                        if help2 < helplower:
                            helplower = help2
            if not drop:
                for i in range(j + 1, maxpos + 1):
                    ii = self.s1[i]
                    if ii > 0 and i != j:
                        if self.ff[ii] <= 1.0:
                            help2 = thirds[self.s2[i]] - thirds[self.s2[j]]
                            help2 = (fv[ii] - fv[jj]) / help2
                            if help2 <= 0.0:
                                drop = True
                                break
                            if help2 > helpgreater:
                                helpgreater = help2
            if not drop:
                if helpgreater <= helplower:
                    if fv[jj] - helplower * thirds[self.s2[j]] > thr:
                        drop = True
            if drop:
                self.s1[j] = 0
        if novalue > 0:
            maxpos += 1
            self.s1[maxpos] = novalue
            self.s2[maxpos] = novaluedeep
        return maxpos

    # --------------------------------------------------- DIRDoubleInsert
    def doubleinsert(self, maxpos: int) -> Tuple[int, int]:
        oldmaxpos = maxpos
        for i in range(1, oldmaxpos + 1):
            if self.s1[i] > 0:
                actdeep = self.s2[i]
                help = self.A(actdeep)
                pos = self.point[help]
                iflag = 0
                while pos > 0 and iflag == 0:
                    if self.fv[pos] - self.fv[help] <= 1e-13:
                        if maxpos < self.maxdiv:
                            maxpos += 1
                            self.s1[maxpos] = pos
                            self.s2[maxpos] = actdeep
                            pos = self.point[pos]
                        else:
                            return maxpos, -6
                    else:
                        iflag = 1
        return maxpos, 0

    # --------------------------------------------------- DIRSamplepoints
    def samplepoints(self, delta: float, sample: int) -> int:
        # returns start (or -1 on failure)
        n = self.n
        pos = self.free
        start = self.free
        for k in range(1, 2 * self.maxi + 1):
            for j in range(1, n + 1):
                self.setL(self.free, j, self.L(sample, j))
                self.setC(self.free, j, self.C(sample, j))
            pos = self.free
            self.free = self.point[self.free]
            if self.free == 0:
                return -1
        self.point[pos] = 0
        pos = start
        for j in range(1, self.maxi + 1):
            a = self.arrayi[j]
            self.setC(pos, a, self.C(sample, a) + delta)
            pos = self.point[pos]
            self.setC(pos, a, self.C(sample, a) - delta)
            pos = self.point[pos]
        return start

    # -------------------------------------------------------- DIRSamplef
    def samplef(self, new: int):
        n = self.n
        pos = new
        for j in range(1, 2 * self.maxi + 1):
            for i in range(1, n + 1):
                self.x[i] = self.C(pos, i)
            fval = self.infcn()
            self.fv[pos] = fval
            self.ff[pos] = 0.0
            self.fmax = max(fval, self.fmax)
            pos = self.point[pos]
        pos = new
        for j in range(1, 2 * self.maxi + 1):
            if self.fv[pos] < self.minf and self.ff[pos] == 0.0:
                self.minf = self.fv[pos]
                self.minpos = pos
            pos = self.point[pos]

    # --------------------------------------------------------- DIRDivide
    def _insertlist2(self, start: int, j: int, k: int) -> int:
        w = self.w
        pos = start
        if start == 0:
            self.list2a[j] = 0
            start = j
        elif w[start] > w[j]:
            self.list2a[j] = start
            start = j
        else:
            for i in range(1, self.maxi + 1):
                if self.list2a[pos] == 0:
                    self.list2a[j] = 0
                    self.list2a[pos] = j
                    break
                else:
                    if w[j] < w[self.list2a[pos]]:
                        self.list2a[j] = self.list2a[pos]
                        self.list2a[pos] = j
                        break
                pos = self.list2a[pos]
        self.list2b[j] = k
        return start

    def divide(self, new: int, currentlength: int, sample: int):
        start = 0
        pos = new
        for i in range(1, self.maxi + 1):
            j = self.arrayi[i]
            self.w[j] = self.fv[pos]
            k = pos
            pos = self.point[pos]
            self.w[j] = min(self.fv[pos], self.w[j])
            pos = self.point[pos]
            start = self._insertlist2(start, j, k)
        for j in range(1, self.maxi + 1):
            # dirsearchmin
            k = start
            pos = self.list2b[start]
            start = self.list2a[start]
            pos2 = start
            self.setL(sample, k, currentlength + 1)
            for i in range(1, self.maxi - j + 2):
                self.setL(pos, k, currentlength + 1)
                pos = self.point[pos]
                self.setL(pos, k, currentlength + 1)
                if pos2 > 0:
                    pos = self.list2b[pos2]
                    pos2 = self.list2a[pos2]

    # ----------------------------------------------------- DIRInsertList
    def _dirinsert(self, start: int, ins: int):
        fv = self.fv
        point = self.point
        for i in range(1, self.maxfunc + 1):
            if point[start] == 0:
                point[start] = ins
                point[ins] = 0
                return
            elif fv[ins] < fv[point[start]]:
                help = point[start]
                point[start] = ins
                point[ins] = help
                return
            start = point[start]

    def insertlist(self, new: int, samp: int, jones: int):
        fv = self.fv
        point = self.point
        for j in range(1, self.maxi + 1):
            pos1 = new
            pos2 = point[pos1]
            new = point[pos2]
            deep = self.getlevel(pos1, jones)
            if self.A(deep) == 0:
                if fv[pos2] < fv[pos1]:
                    self.setA(deep, pos2)
                    point[pos2] = pos1
                    point[pos1] = 0
                else:
                    self.setA(deep, pos1)
                    point[pos2] = 0
            else:
                pos = self.A(deep)
                if fv[pos2] < fv[pos1]:
                    if fv[pos2] < fv[pos]:
                        self.setA(deep, pos2)
                        if fv[pos1] < fv[pos]:
                            point[pos2] = pos1
                            point[pos1] = pos
                        else:
                            point[pos2] = pos
                            self._dirinsert(pos, pos1)
                    else:
                        self._dirinsert(pos, pos2)
                        self._dirinsert(pos, pos1)
                else:
                    if fv[pos1] < fv[pos]:
                        self.setA(deep, pos1)
                        if fv[pos] < fv[pos2]:
                            point[pos1] = pos
                            self._dirinsert(pos, pos2)
                        else:
                            point[pos1] = pos2
                            point[pos2] = pos
                    else:
                        self._dirinsert(pos, pos1)
                        self._dirinsert(pos, pos2)
        deep = self.getlevel(samp, jones)
        pos = self.A(deep)
        if fv[samp] < fv[pos]:
            self.setA(deep, samp)
            point[samp] = pos
        else:
            self._dirinsert(pos, samp)

    # ----------------------------------------------------------- DIRInit
    def init(self, jones: int) -> int:
        n = self.n
        maxdeep = self.maxdeep
        self.minf = math.inf
        if jones == 0:
            for j in range(0, n):
                self.w[j + 1] = math.sqrt(float(n - j) + float(j) / 9.0) * 0.5
            help2 = 1.0
            for i in range(1, maxdeep // n + 1):
                for j in range(0, n):
                    self.levels[(i - 1) * n + j] = self.w[j + 1] / help2
                help2 *= 3.0
        else:
            help2 = 3.0
            for i in range(1, maxdeep + 1):
                self.levels[i] = 1.0 / help2
                help2 *= 3.0
            self.levels[0] = 1.0
        help2 = 3.0
        for i in range(1, maxdeep + 1):
            self.thirds[i] = 1.0 / help2
            help2 *= 3.0
        self.thirds[0] = 1.0
        for i in range(1, n + 1):
            self.setC(1, i, 0.5)
            self.x[i] = 0.5
            self.setL(1, i, 0)
        f1 = self.infcn()
        self.fv[1] = f1
        self.ff[1] = 0.0
        self.fmax = f1
        self.minf = f1
        self.minpos = 1
        self.point[1] = 0
        self.free = 2
        delta = self.thirds[1]
        self.get_i(1)
        new = self.samplepoints(delta, 1)
        if new < 0:
            return -4
        self.samplef(new)
        self.divide(new, 0, 1)
        self.insertlist(new, 1, jones)
        return 0

    def initlist(self):
        for i in range(0, self.maxdeep + 2):
            self.anchor[i] = 0
        for i in range(1, self.maxfunc + 1):
            self.fv[i] = 0.0
            self.ff[i] = 0.0
            self.point[i] = i + 1
        self.point[self.maxfunc] = 0
        self.free = 1

    # --------------------------------------------------------- DIRect
    def run(self, l: np.ndarray[float, 1], u: np.ndarray[float, 1],
            eps: float, maxf: int, maxt: int, algmethod: int,
            fglobal: float, fglper: float, volper: float,
            sigmaper: float) -> Tuple[np.ndarray[float, 1], float, int, int, int]:
        n = self.n
        epsabs = 0.0
        MAXDIV = 5000
        MAXMEMORY = 1073741824
        MAXFUNC = 101000 if maxf <= 0 else (maxf + 1000 + maxf // 2)
        fmd = n * (8 + 4) + (8 * 2 + 4)
        if MAXFUNC * fmd > MAXMEMORY:
            MAXFUNC = MAXMEMORY // fmd
        if maxf <= 0:
            maxf = MAXFUNC - 1000
        else:
            maxf = 2 * (MAXFUNC - 1000) // 3
        MAXDEEP = MAXFUNC // 5 if maxt <= 0 else maxt + 1000
        fmd2 = 8 * 2 + 4
        cm = 2 * (8 + 4)
        if MAXDEEP * fmd2 + cm > MAXMEMORY:
            MAXDEEP = (MAXMEMORY - cm) // fmd2
        if maxt <= 0:
            maxt = MAXDEEP
        else:
            maxt = MAXDEEP - 1000
        self.maxfunc = MAXFUNC
        self.maxdeep = MAXDEEP
        self.maxdiv = MAXDIV
        self.c = [0.0] * ((MAXFUNC + 1) * n)
        self.length = [0] * ((MAXFUNC + 1) * n)
        self.fv = [0.0] * (MAXFUNC + 1)
        self.ff = [0.0] * (MAXFUNC + 1)
        self.point = [0] * (MAXFUNC + 1)
        self.anchor = [0] * (MAXDEEP + 2)
        self.s1 = [0] * (MAXDIV + 1)
        self.s2 = [0] * (MAXDIV + 1)
        self.levels = [0.0] * (MAXDEEP + 1)
        self.thirds = [0.0] * (MAXDEEP + 1)
        self.w = [0.0] * (n + 1)
        self.list2a = [0] * (n + 1)
        self.list2b = [0] * (n + 1)
        self.arrayi = [0] * (n + 1)
        self.x = [0.0] * (n + 1)
        jones = algmethod
        # DIRheader
        iepschange = 0
        epsfix = 1e100
        if eps < 0.0:
            iepschange = 1
            epsfix = -eps
            eps = -eps
        for i in range(n):
            if u[i] <= l[i]:
                return np.zeros(n), math.inf, -1, 0, 0
        if maxf + 20 > MAXFUNC:
            return np.zeros(n), math.inf, -2, 0, 0
        divfactor = 1.0 if fglobal == 0.0 else abs(fglobal)
        oldmaxf = maxf
        increase = 0
        self.initlist()
        # DIRpreprc
        self.xs1 = [0.0] * (n + 1)
        self.xs2 = [0.0] * (n + 1)
        for i in range(1, n + 1):
            help = u[i - 1] - l[i - 1]
            self.xs2[i] = l[i - 1] / help
            self.xs1[i] = help
        ierror = self.init(jones)
        if ierror < 0:
            return np.zeros(n), math.inf, ierror, 0, 0
        numfunc = self.maxi + 1 + self.maxi
        actmaxdeep = 1
        tstart = 2
        ifeasiblef = 0
        mdeep = MAXDEEP
        t = tstart
        finished = False
        while t <= maxt - 1:
            actdeep = actmaxdeep
            maxpos = self.choose(MAXDEEP, eps, epsabs, ifeasiblef, jones)
            if algmethod == 0:
                maxpos, err = self.doubleinsert(maxpos)
                if err == -6:
                    ierror = -6
                    finished = True
                    break
            for j in range(1, maxpos + 1):
                actdeep = self.s2[j]
                if self.s1[j] > 0:
                    actdeep_div = self.getmaxdeep(self.s1[j])
                    delta = self.thirds[actdeep_div + 1]
                    actdeep = self.s2[j]
                    if actdeep + 1 >= mdeep:
                        ierror = -6
                        finished = True
                        break
                    actmaxdeep = max(actdeep, actmaxdeep)
                    help = self.s1[j]
                    if self.A(actdeep) != help:
                        pos1 = self.A(actdeep)
                        while self.point[pos1] != help:
                            pos1 = self.point[pos1]
                        self.point[pos1] = self.point[help]
                    else:
                        self.setA(actdeep, self.point[help])
                    if actdeep < 0:
                        actdeep = int(self.fv[help])
                    self.get_i(help)
                    start = self.samplepoints(delta, help)
                    if start < 0:
                        return np.zeros(n), math.inf, -4, numfunc, t
                    self.samplef(start)
                    self.divide(start, actdeep_div, help)
                    self.insertlist(start, help, jones)
                    numfunc = numfunc + self.maxi + self.maxi
            if finished:
                break
            # termination checks
            lev = self.getlevel(self.minpos, 0)
            delta = self.thirds[lev]
            if delta <= volper:
                ierror = 4
                finished = True
                break
            lev = self.getlevel(self.minpos, jones)
            delta = self.levels[lev]
            if delta <= sigmaper:
                ierror = 5
                finished = True
                break
            if (self.minf - fglobal) / divfactor <= fglper:
                ierror = 3
                finished = True
                break
            if iepschange == 1:
                eps = max(abs(self.minf) * 1e-4, epsfix)
            if increase == 1:
                maxf = numfunc + oldmaxf
                if ifeasiblef == 0:
                    increase = 0
            if numfunc > maxf:
                ierror = 1
                finished = True
                break
            if not isinstance(self.callback, NoneType):
                self.callback(self.xlast.copy())
            t += 1
        if not finished:
            ierror = 2
        xo = np.empty(n)
        for i in range(1, n + 1):
            xo[i - 1] = self.C(self.minpos, i) * self.xs1[i] + self.xs1[i] * self.xs2[i]
        return xo, self.minf, ierror, numfunc, t

def direct(func, bounds, args = (), eps: float = 0.0001,
           maxfun: Optional[int] = None, maxiter: int = 1000,
           locally_biased: bool = True, f_min: float = -math.inf,
           f_min_rtol: float = 0.0001, vol_tol: float = 1e-16,
           len_tol: float = 1e-06, callback = None):
    from ._differentialevolution import _bounds_limits
    if not isinstance(args, Tuple):
        return direct(func, bounds, (args,), eps, maxfun, maxiter,
                      locally_biased, f_min, f_min_rtol, vol_tol, len_tol,
                      callback)
    lb, ub = _bounds_limits(bounds)
    n = lb.size
    for i in range(n):
        if not lb[i] < ub[i]:
            raise ValueError('Bounds are not consistent min < max')
    for i in range(n):
        if math.isinf(lb[i]) or math.isinf(ub[i]):
            raise ValueError('Bounds must not be inf.')
    if vol_tol < 0 or vol_tol > 1:
        raise ValueError('vol_tol must be between 0 and 1.')
    if len_tol < 0 or len_tol > 1:
        raise ValueError('len_tol must be between 0 and 1.')
    if f_min_rtol < 0 or f_min_rtol > 1:
        raise ValueError('f_min_rtol must be between 0 and 1.')
    mf = 1000 * n
    if maxfun is not None:
        mf = maxfun
    if mf < 0:
        raise ValueError('maxfun must be > 0.')
    if maxiter < 0:
        raise ValueError('maxiter must be > 0.')
    # direct_wrap.c
    volper = vol_tol if vol_tol > 0 else -1.0
    sigmaper = len_tol if len_tol > 0 else -1.0
    fglper = f_min_rtol
    if f_min == -math.inf:
        fglper = 0.0
    if n < 1:
        raise ValueError(f'DIRECT requires at least one dimension, got {n}')
    d = _Direct(func, args, callback, n)
    x, fun, ret_code, nfev, nit = d.run(lb, ub, eps, mf, maxiter,
                                        1 if locally_biased else 0, f_min,
                                        fglper, volper, sigmaper)
    fv = [str(mf), str(maxiter), str(f_min_rtol), str(vol_tol), str(len_tol)]
    message = ''
    if ret_code > 2:
        message = _SUCCESS_MESSAGES[ret_code - 3].replace('{}', fv[ret_code - 1])
    elif 0 < ret_code <= 2:
        message = _ERROR_MESSAGES[ret_code - 1].replace('{}', fv[ret_code - 1])
    elif 0 > ret_code > -100:
        message = _ERROR_MESSAGES[abs(ret_code) + 1]
    else:
        message = _ERROR_MESSAGES[ret_code + 99]
    return OptimizeResult(x, fun, np.empty(0, float), nit=nit, nfev=nfev,
                          status=ret_code, success=ret_code > 2,
                          message=message)
