# Ported from scipy 1.18.1 (BSD-3-Clause)

# trust-constr (_trustregion_constr/minimize_trustregion_constr.py,
# canonical_constraint.py, equality_constrained_sqp.py, tr_interior_point.py)
# and the VectorFunction / LinearVectorFunction / IdentityVectorFunction
# constraint wrappers of _differentiable_functions.py. Hessians are applied
# through small operator records (_HOp) that keep scipy's aliasing
# semantics: quasi-Newton approximations are shared, mutable objects.

import math
import numpy as np
from ._common import OptimizeResult, Bounds, INF, _approx_jac, _vec, \
    _opt_float, _opt_int, _has_opt
from ._constraints import LinearConstraint, NonlinearConstraint
from ._hessian_update_strategy import HessianUpdateStrategy, BFGS
from ._differentiable import _ScalarFunctionH, _HESS_HUS, _HESS_CALLABLE, \
    _HESS_FD, _HESS_NONE
from ._cons import _u1, _u2, _ustr, _is_seq_of_cons
from ._callback import _cb_halt_r
from ._tc_qp import _Proj, modified_dogleg, projected_cg, box_intersections

_H_ZERO = 0
_H_MAT = 1
_H_HUS = 2

class _HOp:
    # one Hessian term: zero, a dense matrix snapshot or a shared
    # HessianUpdateStrategy (dot = symv on its current state)
    mode: int
    M: np.ndarray[float, 2]
    hus: HessianUpdateStrategy

    def __init__(self, mode: int, M: np.ndarray[float, 2], hus: HessianUpdateStrategy):
        self.mode = mode
        self.M = M
        self.hus = hus

    def dot(self, p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if self.mode == _H_MAT:
            return self.M.dot(p)
        elif self.mode == _H_HUS:
            return self.hus.dot(p)
        return np.zeros(p.size)

def _dummy_hus() -> HessianUpdateStrategy:
    return HessianUpdateStrategy(0)

def _same(a: np.ndarray[float, 1], b: np.ndarray[float, 1]) -> bool:
    if a.size != b.size:
        return False
    for i in range(a.size):
        if a[i] != b[i]:
            return False
    return True

class _NLFunTC[F]:
    f: F

    def __init__(self, f: F):
        self.f = f

    def __call__(self, x) -> np.ndarray[float, 1]:
        return _u1(self.f(x))

class _VecFunTC[F, JC, HS]:
    # VectorFunction: jac is callable or a finite-difference string; hess is
    # callable(x, v) or a HessianUpdateStrategy
    fun_: F
    jac_: JC
    hess_: HS
    n: int
    m: int
    x: np.ndarray[float, 1]
    f: np.ndarray[float, 1]
    J: np.ndarray[float, 2]
    H: np.ndarray[float, 2]
    v: np.ndarray[float, 1]
    _nfev: int
    _njev: int
    _nhev: int
    _fdnfev: int
    f_updated: bool
    J_updated: bool
    H_updated: bool
    fd: str
    rel_step: Optional[float]
    lb: Optional[np.ndarray[float, 1]]
    ub: Optional[np.ndarray[float, 1]]
    hus: HessianUpdateStrategy
    is_hus: bool
    has_prev: bool
    x_prev: np.ndarray[float, 1]
    J_prev: np.ndarray[float, 2]

    def __init__(self, fun: F, x0: np.ndarray[float, 1], jac: JC, hess: HS,
                 rel_step: Optional[float], lb: Optional[np.ndarray[float, 1]],
                 ub: Optional[np.ndarray[float, 1]], fd_override: str = ''):
        self.fun_ = fun
        self.jac_ = jac
        self.hess_ = hess
        self.x = x0.copy()
        self.n = x0.size
        self._nfev = 0
        self._njev = 0
        self._nhev = 0
        self._fdnfev = 0
        self.f_updated = False
        self.J_updated = False
        self.H_updated = False
        self.rel_step = rel_step
        self.lb = lb
        self.ub = ub
        self.fd = ''
        if isinstance(jac, str):
            if jac != '2-point' and jac != '3-point' and jac != 'cs':
                raise ValueError("`jac` must be either callable or one of ('2-point', '3-point', 'cs').")
            self.fd = jac if jac != 'cs' else '2-point'
        if fd_override != '':
            self.fd = fd_override
        self.is_hus = isinstance(hess, HessianUpdateStrategy)
        self.hus = _dummy_hus()
        if isinstance(hess, HessianUpdateStrategy):
            self.hus = hess
        self.has_prev = False
        self.x_prev = np.zeros(0)
        self.J_prev = np.zeros((0, 0))
        self.f = np.zeros(0)
        self.H = np.zeros((0, 0))
        self._update_fun()
        self.v = np.zeros(self.f.size)
        self.m = self.v.size
        self.J = np.zeros((0, self.n))
        if self.fd == '':
            self.J = self._eval_jac(self.x)
            self._njev += 1
        else:
            self.J = self._fd_jac(self.x)
        self.J_updated = True
        if self.is_hus:
            self.hus.initialize(self.n, 'hess')
            self.H_updated = True
        else:
            self.H = self._eval_hess(self.x.copy(), self.v)
            self.H_updated = True
            self._nhev += 1

    def _eval_jac(self, x) -> np.ndarray[float, 2]:
        if isinstance(self.jac_, str):
            return np.zeros((0, self.n))
        else:
            return _u2(self.jac_(x.copy()), self.n)

    def _eval_hess(self, x, v) -> np.ndarray[float, 2]:
        if isinstance(self.hess_, HessianUpdateStrategy):
            return np.zeros((0, 0))
        elif self.hess_ is None:
            return np.zeros((0, 0))
        else:
            return np.atleast_2d(np.asarray(self.hess_(x, v), float))

    def _fd_jac(self, x) -> np.ndarray[float, 2]:
        J, nf = _approx_jac(_NLFunTC(self.fun_), x, self.f, self.fd,
                            self.rel_step, None, self.lb, self.ub)
        self._fdnfev += nf
        return J

    @property
    def nfev(self) -> int:
        return self._nfev + self._fdnfev

    @property
    def njev(self) -> int:
        return self._njev

    @property
    def nhev(self) -> int:
        return self._nhev

    def _update_v(self, v: np.ndarray[float, 1]):
        if not _same(v, self.v):
            self.v = v.copy()
            self.H_updated = False

    def _update_x(self, x: np.ndarray[float, 1]):
        if not _same(x, self.x):
            if self.is_hus:
                self._update_jac()
                self.x_prev = self.x
                self.J_prev = self.J
                self.has_prev = True
                self.x = x.copy()
                self.f_updated = False
                self.J_updated = False
                self.H_updated = False
                self._update_hess()
            else:
                self.x = x.copy()
                self.f_updated = False
                self.J_updated = False
                self.H_updated = False

    def _update_fun(self):
        if not self.f_updated:
            self.f = _u1(self.fun_(self.x.copy()))
            self._nfev += 1
            self.f_updated = True

    def _update_jac(self):
        if not self.J_updated:
            if self.fd != '':
                self._update_fun()
                self.J = self._fd_jac(self.x)
            else:
                self._njev += 1
                self.J = self._eval_jac(self.x)
            self.J_updated = True

    def _update_hess(self):
        if not self.H_updated:
            if self.is_hus:
                self._update_jac()
                if self.has_prev:
                    delta_x = self.x - self.x_prev
                    delta_g = self.J.T.dot(self.v) - self.J_prev.T.dot(self.v)
                    self.hus.update(delta_x, delta_g)
            else:
                self.H = self._eval_hess(self.x.copy(), self.v)
                self._nhev += 1
            self.H_updated = True

    def fun(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        self._update_x(x)
        self._update_fun()
        return self.f.copy()

    def jac(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        self._update_x(x)
        self._update_jac()
        return self.J.copy()

    def hess(self, x: np.ndarray[float, 1], v: np.ndarray[float, 1]) -> _HOp:
        self._update_v(v)
        self._update_x(x)
        self._update_hess()
        if self.is_hus:
            return _HOp(_H_HUS, np.zeros((0, 0)), self.hus)
        return _HOp(_H_MAT, self.H.copy(), self.hus)

class _LinVF:
    # LinearVectorFunction / IdentityVectorFunction
    J: np.ndarray[float, 2]
    sparse: bool
    m: int
    n: int
    x: np.ndarray[float, 1]
    f: np.ndarray[float, 1]
    f_updated: bool
    v: np.ndarray[float, 1]

    def __init__(self, A: np.ndarray[float, 2], x0: np.ndarray[float, 1], sparse: bool):
        self.J = A
        self.sparse = sparse
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.x = x0.copy()
        self.f = A.dot(self.x)
        self.f_updated = True
        self.v = np.zeros(self.m)

    @property
    def nfev(self) -> int:
        return 0

    @property
    def njev(self) -> int:
        return 0

    @property
    def nhev(self) -> int:
        return 0

    def _update_x(self, x):
        if not _same(x, self.x):
            self.x = x.copy()
            self.f_updated = False

    def fun(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        self._update_x(x)
        if not self.f_updated:
            self.f = self.J.dot(x)
            self.f_updated = True
        return self.f

    def jac(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        self._update_x(x)
        return self.J

    def hess(self, x: np.ndarray[float, 1], v: np.ndarray[float, 1]) -> _HOp:
        self._update_x(x)
        self.v = v.copy()
        return _HOp(_H_ZERO, np.zeros((0, 0)), _dummy_hus())

class _DictFunTC[D]:
    d: D

    def __init__(self, d: D):
        self.d = d

    def __call__(self, x) -> np.ndarray[float, 1]:
        return _u1(self.d['fun'](x))

class _DictJacTC[D]:
    d: D
    n: int

    def __init__(self, d: D, n: int):
        self.d = d
        self.n = n

    def __call__(self, x) -> np.ndarray[float, 2]:
        if 'jac' in self.d:
            return _u2(self.d['jac'](x), self.n)
        return np.zeros((0, self.n))

# ------------------------------------------------------------- canonical

class _Canon:
    # CanonicalConstraint.from_PreparedConstraint for one constraint:
    # eq_k = f[eq_idx[k]] - eq_off[k]; ineq_k = sg[k] * (f[iq_idx[k]] - iq_off[k])
    m: int
    eq_idx: List[int]
    eq_off: List[float]
    iq_idx: List[int]
    iq_sg: List[float]
    iq_off: List[float]
    kf: List[bool]

    def __init__(self, lb: np.ndarray[float, 1], ub: np.ndarray[float, 1],
                 keep: np.ndarray[bool, 1]):
        m = lb.size
        self.m = m
        self.eq_idx = List[int]()
        self.eq_off = List[float]()
        self.iq_idx = List[int]()
        self.iq_sg = List[float]()
        self.iq_off = List[float]()
        self.kf = List[bool]()
        all_lninf = True
        all_uinf = True
        all_eq = True
        for i in range(m):
            if lb[i] != -INF:
                all_lninf = False
            if ub[i] != INF:
                all_uinf = False
            if lb[i] != ub[i]:
                all_eq = False
        if all_lninf and all_uinf:
            return
        if all_eq:
            for i in range(m):
                self.eq_idx.append(i)
                self.eq_off.append(lb[i])
        elif all_lninf:
            for i in range(m):
                if ub[i] < INF:
                    self._iq(i, 1.0, ub[i], keep[i])
        elif all_uinf:
            for i in range(m):
                if lb[i] > -INF:
                    self._iq(i, -1.0, lb[i], keep[i])
        else:
            for i in range(m):
                if lb[i] == ub[i]:
                    self.eq_idx.append(i)
                    self.eq_off.append(lb[i])
            for i in range(m):
                if lb[i] == -INF and ub[i] != INF:
                    self._iq(i, 1.0, ub[i], keep[i])
            for i in range(m):
                if ub[i] == INF and lb[i] != -INF:
                    self._iq(i, -1.0, lb[i], keep[i])
            for i in range(m):
                if lb[i] != ub[i] and lb[i] != -INF and ub[i] != INF:
                    self._iq(i, 1.0, ub[i], keep[i])
            for i in range(m):
                if lb[i] != ub[i] and lb[i] != -INF and ub[i] != INF:
                    self._iq(i, -1.0, lb[i], keep[i])

    def _iq(self, i: int, sg: float, off: float, kf: bool):
        self.iq_idx.append(i)
        self.iq_sg.append(sg)
        self.iq_off.append(off)
        self.kf.append(kf)

    def n_eq(self) -> int:
        return len(self.eq_idx)

    def n_ineq(self) -> int:
        return len(self.iq_idx)

    def fun(self, f: np.ndarray[float, 1]):
        ne = self.n_eq()
        ni = self.n_ineq()
        e = np.empty(ne)
        q = np.empty(ni)
        for k in range(ne):
            e[k] = f[self.eq_idx[k]] - self.eq_off[k]
        for k in range(ni):
            i = self.iq_idx[k]
            if self.iq_sg[k] > 0:
                q[k] = f[i] - self.iq_off[k]
            else:
                q[k] = self.iq_off[k] - f[i]
        return e, q

    def jac(self, J: np.ndarray[float, 2]):
        n = J.shape[1]
        ne = self.n_eq()
        ni = self.n_ineq()
        E = np.empty((ne, n))
        Q = np.empty((ni, n))
        for k in range(ne):
            for j in range(n):
                E[k, j] = J[self.eq_idx[k], j]
        for k in range(ni):
            i = self.iq_idx[k]
            for j in range(n):
                Q[k, j] = J[i, j] if self.iq_sg[k] > 0 else -J[i, j]
        return E, Q

    def v_full(self, v_eq: np.ndarray[float, 1], v_ineq: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        v = np.zeros(self.m)
        for k in range(self.n_eq()):
            v[self.eq_idx[k]] = v_eq[k]
        # interval rows receive v_il - v_ig in two steps (exact)
        for k in range(self.n_ineq()):
            i = self.iq_idx[k]
            if self.iq_sg[k] > 0:
                v[i] = v[i] + v_ineq[k]
            else:
                v[i] = v[i] - v_ineq[k]
        return v

def _cat1(a: np.ndarray[float, 1], b: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    return np.concatenate((a, b))

def _cat2(a: np.ndarray[float, 2], b: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    if a.shape[0] == 0:
        return b.copy()
    if b.shape[0] == 0:
        return a
    return np.concatenate((a, b), axis=0)

# ------------------------------------------------------- prepared constraint

class _Prep[VF]:
    # PreparedConstraint + its canonical form
    fun: VF
    lb: np.ndarray[float, 1]
    ub: np.ndarray[float, 1]
    canon: _Canon

    def __init__(self, fun: VF, lb: np.ndarray[float, 1], ub: np.ndarray[float, 1],
                 keep: np.ndarray[bool, 1]):
        self.fun = fun
        m = fun.m
        if lb.size == 1 and m != 1:
            lb = np.full(m, lb[0])
        if ub.size == 1 and m != 1:
            ub = np.full(m, ub[0])
        if keep.size == 1 and m != 1:
            keep = np.full(m, keep[0])
        self.lb = lb
        self.ub = ub
        f0 = fun.f
        for i in range(m):
            if keep[i] and lb[i] != ub[i] and (f0[i] < lb[i] or f0[i] > ub[i]):
                raise ValueError('`x0` is infeasible with respect to some inequality constraint with `keep_feasible` set to True.')
        self.canon = _Canon(lb, ub, keep)

    def c_fun(self, x: np.ndarray[float, 1]):
        return self.canon.fun(self.fun.fun(x))

    def c_jac(self, x: np.ndarray[float, 1]):
        return self.canon.jac(self.fun.jac(x))

    def c_hess(self, x: np.ndarray[float, 1], v_eq: np.ndarray[float, 1],
               v_ineq: np.ndarray[float, 1]) -> _HOp:
        return self.fun.hess(x, self.canon.v_full(v_eq, v_ineq))

    def c_initial(self):
        return self.canon.fun(self.fun.f) + self.canon.jac(self.fun.J)

def _prep_one(c, x0: np.ndarray[float, 1], fdb_lb, fdb_ub, fd: str,
              rel: Optional[float], sparse: bool):
    n = x0.size
    if isinstance(c, LinearConstraint):
        return _Prep(_LinVF(c.A, x0, sparse), c.lb, c.ub,
                     np.full(c.lb.size, c.keep_feasible))
    elif isinstance(c, NonlinearConstraint):
        hs = c.hess
        if c.hess is None:
            # NonlinearConstraint default hess=BFGS(), one per constraint
            vf = _VecFunTC(c.fun, x0, c.jac, BFGS(), rel, fdb_lb, fdb_ub)
            return _Prep(vf, c.lb, c.ub, np.full(1, c.keep_feasible))
        else:
            vf = _VecFunTC(c.fun, x0, c.jac, hs, rel, fdb_lb, fdb_ub)
            return _Prep(vf, c.lb, c.ub, np.full(1, c.keep_feasible))
    else:
        # old-style dict -> NonlinearConstraint(fun, 0, 0|inf, jac|'2-point')
        t = _ustr(c['type']).lower()
        if t != 'eq' and t != 'ineq':
            raise ValueError(f"Unknown constraint type '{t}'.")
        ub = np.zeros(1) if t == 'eq' else np.full(1, INF)
        fdo = '' if 'jac' in c else '2-point'
        vf = _VecFunTC(_DictFunTC(c), x0, _DictJacTC(c, n), BFGS(), rel, fdb_lb,
                       fdb_ub, fdo)
        return _Prep(vf, np.zeros(1), ub, np.zeros(1, bool))

def _prep_all(cs, x0, fdb_lb, fdb_ub, fd: str, rel: Optional[float], sparse: bool):
    if static.len(cs) == 0:
        return ()
    else:
        return (_prep_one(cs[0], x0, fdb_lb, fdb_ub, fd, rel, sparse),) + \
            _prep_all(cs[1:], x0, fdb_lb, fdb_ub, fd, rel, sparse)

def _as_tuple(constraints):
    if constraints is None:
        return ()
    elif isinstance(constraints, Tuple):
        return constraints
    elif isinstance(constraints, List):
        compile_error("trust-constr: pass constraints as a tuple (heterogeneous lists are not supported)")
    else:
        return (constraints,)

class _AllCons[PS]:
    # CanonicalConstraint.concatenate over all prepared constraints
    ps: PS
    n: int
    n_eq: int
    n_ineq: int
    keep: np.ndarray[bool, 1]

    def __init__(self, ps: PS, n: int):
        self.ps = ps
        self.n = n
        ne = 0
        ni = 0
        kf = List[bool]()
        for p in ps:
            ne += p.canon.n_eq()
            ni += p.canon.n_ineq()
            for b in p.canon.kf:
                kf.append(b)
        self.n_eq = ne
        self.n_ineq = ni
        self.keep = np.zeros(len(kf), bool)
        for i in range(len(kf)):
            self.keep[i] = kf[i]

    def fun(self, x: np.ndarray[float, 1]):
        e = np.zeros(0)
        q = np.zeros(0)
        for p in self.ps:
            e1, q1 = p.c_fun(x)
            e = _cat1(e, e1)
            q = _cat1(q, q1)
        return e, q

    def jac(self, x: np.ndarray[float, 1]):
        E = np.zeros((0, self.n))
        Q = np.zeros((0, self.n))
        for p in self.ps:
            E1, Q1 = p.c_jac(x)
            E = _cat2(E, E1)
            Q = _cat2(Q, Q1)
        return E, Q

    def initial(self):
        e = np.zeros(0)
        q = np.zeros(0)
        E = np.zeros((0, self.n))
        Q = np.zeros((0, self.n))
        for p in self.ps:
            e1, q1, E1, Q1 = p.c_initial()
            e = _cat1(e, e1)
            q = _cat1(q, q1)
            E = _cat2(E, E1)
            Q = _cat2(Q, Q1)
        return e, q, E, Q

    def hess(self, x: np.ndarray[float, 1], v_eq: np.ndarray[float, 1],
             v_ineq: np.ndarray[float, 1]) -> List[_HOp]:
        out = List[_HOp]()
        ie = 0
        ii = 0
        for p in self.ps:
            ne = p.canon.n_eq()
            ni = p.canon.n_ineq()
            out.append(p.c_hess(x, v_eq[ie:ie + ne].copy(), v_ineq[ii:ii + ni].copy()))
            ie += ne
            ii += ni
        return out

# ----------------------------------------------------- Lagrangian Hessian

class _ObjH[SF, HP, A]:
    # objective Hessian at x: sf.hess(x) (matrix, quasi-Newton object or
    # HessianLinearOperator(hessp))
    sf: SF
    hp: HP
    args: A
    x: np.ndarray[float, 1]
    M: np.ndarray[float, 2]
    use_hp: bool
    hus: bool

    def __init__(self, sf: SF, hp: HP, args: A, x: np.ndarray[float, 1], use_hp: bool):
        self.sf = sf
        self.hp = hp
        self.args = args
        self.x = x.copy()
        self.use_hp = use_hp
        self.M = np.zeros((0, 0))
        self.hus = False
        Hm = sf.hess(x)
        if sf.hmode == _HESS_HUS:
            self.hus = True
        elif not use_hp:
            self.M = Hm

    def dot(self, p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if self.use_hp:
            if self.hp is None:
                return np.zeros(p.size)
            else:
                return _vec(self.hp(self.x, p, *self.args))
        elif self.hus:
            return self.sf.hess_dot(self.sf.x, p)
        elif self.sf.hmode == _HESS_FD:
            return self.sf.hess_dot(self.x, p)
        return self.M.dot(p)

class _LagH[OH]:
    oh: OH
    ch: List[_HOp]
    n: int

    def __init__(self, oh: OH, ch: List[_HOp], n: int):
        self.oh = oh
        self.ch = ch
        self.n = n

    def dot(self, p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        hc = np.zeros(p.size)
        for h in self.ch:
            hc += h.dot(p)
        return self.oh.dot(p) + hc

class _BarH[LH]:
    # BarrierSubproblem.lagrangian_hessian
    Hx: LH
    shs: np.ndarray[float, 1]
    n_vars: int
    n_ineq: int

    def __init__(self, Hx: LH, shs: np.ndarray[float, 1], n_vars: int, n_ineq: int):
        self.Hx = Hx
        self.shs = shs
        self.n_vars = n_vars
        self.n_ineq = n_ineq

    def dot(self, vec: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        vx = vec[:self.n_vars].copy()
        if self.n_ineq > 0:
            vs = vec[self.n_vars:self.n_vars + self.n_ineq]
            return np.concatenate((self.Hx.dot(vx), self.shs * vs))
        return self.Hx.dot(vx)

def _norm(v: np.ndarray[float, 1]) -> float:
    if v.size == 0:
        return 0.0
    return float(np.linalg.norm(v))

def _norm_inf(v: np.ndarray[float, 1]) -> float:
    if v.size == 0:
        return 0.0
    return float(np.max(np.abs(v)))

_TC_MESSAGES = ('The maximum number of function evaluations is exceeded.',
                '`gtol` termination condition is satisfied.',
                '`xtol` termination condition is satisfied.',
                '`callback` raised `StopIteration`.',
                "Constraint violation exceeds 'gtol'")

class _TC[SF, HP, A, PS, CB]:
    sf: SF
    hp: HP
    args: A
    use_hp: bool
    cons: _AllCons[PS]
    cb: CB
    n: int
    sparse: bool
    gtol: float
    xtol: float
    barrier_tol: float
    maxiter: int
    # barrier subproblem
    barrier: bool
    n_eq: int
    n_ineq: int
    bp: float
    btol: float
    enforce: np.ndarray[bool, 1]
    terminate: bool
    # state
    nit: int
    cg_niter: int
    cg_stop_cond: int
    x: np.ndarray[float, 1]
    fun: float
    grad: np.ndarray[float, 1]
    lagrangian_grad: np.ndarray[float, 1]
    optimality: float
    constr_violation: float
    tr_radius: float
    constr_penalty: float
    barrier_parameter: float
    barrier_tolerance: float
    status: int
    method: str

    def __init__(self, sf: SF, hp: HP, args: A, use_hp: bool, cons: _AllCons[PS],
                 cb: CB, n: int, sparse: bool, gtol: float, xtol: float,
                 barrier_tol: float, maxiter: int):
        self.sf = sf
        self.hp = hp
        self.args = args
        self.use_hp = use_hp
        self.cons = cons
        self.cb = cb
        self.n = n
        self.sparse = sparse
        self.gtol = gtol
        self.xtol = xtol
        self.barrier_tol = barrier_tol
        self.maxiter = maxiter
        self.barrier = False
        self.n_eq = cons.n_eq
        self.n_ineq = cons.n_ineq
        self.bp = 0.0
        self.btol = 0.0
        self.enforce = cons.keep
        self.terminate = False
        self.nit = 0
        self.cg_niter = 0
        self.cg_stop_cond = 0
        self.x = sf.x.copy()
        self.fun = sf.f
        self.grad = sf.g.copy()
        self.lagrangian_grad = sf.g.copy()
        self.optimality = 0.0
        self.constr_violation = 0.0
        self.tr_radius = 0.0
        self.constr_penalty = 0.0
        self.barrier_parameter = 0.0
        self.barrier_tolerance = 0.0
        self.status = -1
        self.method = 'equality_constrained_sqp' if cons.n_ineq == 0 else 'tr_interior_point'

    # ------------------------------------------------ problem callbacks
    def _x(self, z: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return z[:self.n].copy()

    def _s(self, z: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return z[self.n:self.n + self.n_ineq].copy()

    def _bfun(self, f: float, c_ineq: np.ndarray[float, 1], s: np.ndarray[float, 1]) -> float:
        for i in range(s.size):
            if self.enforce[i]:
                s[i] = -c_ineq[i]
        tot = 0.0
        for si in s:
            tot += math.log(si) if si > 0 else -INF
        return f - self.bp * tot

    def _bconstr(self, c_ineq, c_eq, s) -> np.ndarray[float, 1]:
        return np.concatenate((c_eq, c_ineq + s))

    def _bgrad(self, g: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return np.concatenate((g, -self.bp * np.ones(self.n_ineq)))

    def _bjac(self, J_eq: np.ndarray[float, 2], J_ineq: np.ndarray[float, 2],
              s: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        if self.n_ineq == 0:
            return J_eq
        n = self.n
        ne = self.n_eq
        ni = self.n_ineq
        J = np.zeros((ne + ni, n + ni))
        for i in range(ne):
            for j in range(n):
                J[i, j] = J_eq[i, j]
        for i in range(ni):
            for j in range(n):
                J[ne + i, j] = J_ineq[i, j]
            J[ne + i, n + i] = s[i]
        return J

    def f_and_c(self, z: np.ndarray[float, 1]):
        if not self.barrier:
            f = self.sf.fun(z)
            c_eq, _ = self.cons.fun(z)
            return f, c_eq
        x = self._x(z)
        s = self._s(z)
        c_eq, c_ineq = self.cons.fun(x)
        f = INF
        infeas = False
        for i in range(c_ineq.size):
            if self.enforce[i] and c_ineq[i] >= 0:
                infeas = True
        if not infeas:
            f = self.sf.fun(x)
        return self._bfun(f, c_ineq, s), self._bconstr(c_ineq, c_eq, s)

    def g_and_j(self, z: np.ndarray[float, 1]):
        if not self.barrier:
            g = self.sf.grad(z)
            J_eq, _ = self.cons.jac(z)
            return g.copy(), J_eq
        x = self._x(z)
        s = self._s(z)
        g = self.sf.grad(x)
        J_eq, J_ineq = self.cons.jac(x)
        return self._bgrad(g), self._bjac(J_eq, J_ineq, s)

    def lagr_hess(self, z: np.ndarray[float, 1], v: np.ndarray[float, 1]):
        x = self._x(z)
        v_eq = v[:self.n_eq].copy()
        v_ineq = v[self.n_eq:self.n_eq + self.n_ineq].copy() if self.barrier else np.zeros(0)
        oh = _ObjH(self.sf, self.hp, self.args, x, self.use_hp)
        ch = self.cons.hess(x, v_eq, v_ineq)
        Hx = _LagH(oh, ch, self.n)
        shs = np.zeros(0)
        if self.barrier and self.n_ineq > 0:
            s = self._s(z)
            vi = v[v.size - self.n_ineq:]
            shs = np.empty(self.n_ineq)
            for i in range(self.n_ineq):
                shs[i] = vi[i] * s[i] if vi[i] > 0 else self.bp
        return _BarH(Hx, shs, self.n, self.n_ineq if self.barrier else 0)

    def scaling(self, z: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if not self.barrier:
            return np.ones(z.size)
        return np.concatenate((np.ones(self.n), self._s(z)))

    # ------------------------------------------------------- stopping
    def _update_state(self, x: np.ndarray[float, 1], failed: bool, tr_radius: float,
                      penalty: float, cg_niter: int, cg_stop: int):
        self.nit += 1
        if not failed:
            self.x = x.copy()
            self.fun = self.sf.f
            self.grad = self.sf.g.copy()
            lg = self.grad.copy()
            viol = 0.0
            for p in self.cons.ps:
                lg += p.fun.J.T.dot(p.fun.v)
                c = p.fun.f
                for i in range(c.size):
                    viol = max(viol, max(p.lb[i] - c[i], c[i] - p.ub[i]))
            self.lagrangian_grad = lg
            self.optimality = _norm_inf(lg)
            self.constr_violation = viol
        self.tr_radius = tr_radius
        self.constr_penalty = penalty
        self.cg_niter += cg_niter
        self.cg_stop_cond = cg_stop

    def result(self):
        r = OptimizeResult(self.x.copy(), self.fun, self.grad.copy(), nit=self.nit,
                           nfev=self.sf.nfev, njev=self.sf.ngev,
                           status=self.status, method=self.method)
        r.nhev = self.sf.nhev
        r.cg_niter = self.cg_niter
        r.cg_stop_cond = self.cg_stop_cond
        r.grad = self.grad.copy()
        r.lagrangian_grad = self.lagrangian_grad.copy()
        r.optimality = self.optimality
        r.constr_violation = self.constr_violation
        r.tr_radius = self.tr_radius
        r.constr_penalty = self.constr_penalty
        r.barrier_parameter = self.barrier_parameter
        r.barrier_tolerance = self.barrier_tolerance
        for p in self.cons.ps:
            r.constr.append(p.fun.f.copy())
            r.v.append(p.fun.v.copy())
            r.constr_nfev.append(p.fun.nfev)
            r.constr_njev.append(p.fun.njev)
            r.constr_nhev.append(p.fun.nhev)
        return r

    def global_stop(self, x: np.ndarray[float, 1], failed: bool, tr_radius: float,
                    penalty: float, cg_niter: int, cg_stop: int) -> bool:
        self._update_state(x, failed, tr_radius, penalty, cg_niter, cg_stop)
        if self.barrier:
            self.barrier_parameter = self.bp
            self.barrier_tolerance = self.btol
        self.status = -1
        if self.cb is not None:
            if _cb_halt_r(self.cb, self.result()):
                self.status = 3
                return True
        if self.optimality < self.gtol and self.constr_violation < self.gtol:
            self.status = 1
        elif self.barrier and self.tr_radius < self.xtol and self.bp < self.barrier_tol:
            self.status = 2
        elif (not self.barrier) and self.tr_radius < self.xtol:
            self.status = 2
        elif self.nit >= self.maxiter:
            self.status = 0
        return self.status != -1

    def stop(self, z: np.ndarray[float, 1], failed: bool, optimality: float,
             constr_violation: float, tr_radius: float, penalty: float,
             cg_niter: int, cg_stop: int) -> bool:
        if not self.barrier:
            return self.global_stop(z, failed, tr_radius, penalty, cg_niter, cg_stop)
        x = self._x(z)
        if self.global_stop(x, failed, tr_radius, penalty, cg_niter, cg_stop):
            self.terminate = True
            return True
        g_cond = optimality < self.btol and constr_violation < self.btol
        x_cond = tr_radius < self.xtol
        return g_cond or x_cond

    # ------------------------------------------------- equality SQP
    def sqp(self, x0: np.ndarray[float, 1], fun0: float, grad0: np.ndarray[float, 1],
            constr0: np.ndarray[float, 1], jac0: np.ndarray[float, 2],
            initial_penalty: float, initial_trust_radius: float,
            trust_lb: np.ndarray[float, 1], trust_ub: np.ndarray[float, 1]):
        PENALTY_FACTOR = 0.3
        LARGE_REDUCTION_RATIO = 0.9
        INTERMEDIARY_REDUCTION_RATIO = 0.3
        SUFFICIENT_REDUCTION_RATIO = 1e-08
        TRUST_ENLARGEMENT_FACTOR_L = 7.0
        TRUST_ENLARGEMENT_FACTOR_S = 2.0
        MAX_TRUST_REDUCTION = 0.5
        MIN_TRUST_REDUCTION = 0.1
        SOC_THRESHOLD = 0.1
        TR_FACTOR = 0.8
        BOX_FACTOR = 0.5
        x = x0.copy()
        trust_radius = initial_trust_radius
        penalty = initial_penalty
        f = fun0
        c = grad0
        b = constr0
        A = jac0
        S = self.scaling(x)
        P = _Proj(A, self.sparse)
        v = -P.least_squares(c)
        H = self.lagr_hess(x, v)
        optimality = _norm_inf(c + A.T.dot(v))
        constr_violation = _norm_inf(b) if b.size > 0 else 0.0
        cg_niter = 0
        cg_stop = 0
        failed = False
        while not self.stop(x, failed, optimality, constr_violation, trust_radius,
                            penalty, cg_niter, cg_stop):
            dn = modified_dogleg(A, P, b, TR_FACTOR * trust_radius,
                                 BOX_FACTOR * trust_lb, BOX_FACTOR * trust_ub)
            c_t = H.dot(dn) + c
            b_t = np.zeros_like(b)
            trust_radius_t = math.sqrt(trust_radius ** 2 - _norm(dn) ** 2)
            lb_t = trust_lb - dn
            ub_t = trust_ub - dn
            dt, cg_niter, cg_stop, _hb = projected_cg(H, c_t, P, b_t, trust_radius_t, lb_t, ub_t)
            d = dn + dt
            quadratic_model = 0.5 * float(H.dot(d).dot(d)) + float(c.dot(d))
            linearized_constr = A.dot(d) + b
            vpred = _norm(b) - _norm(linearized_constr)
            vpred = max(1e-16, vpred)
            previous_penalty = penalty
            if quadratic_model > 0:
                new_penalty = quadratic_model / ((1 - PENALTY_FACTOR) * vpred)
                penalty = max(penalty, new_penalty)
            predicted_reduction = -quadratic_model + penalty * vpred
            merit_function = f + penalty * _norm(b)
            x_next = x + S * d
            f_next, b_next = self.f_and_c(x_next)
            merit_function_next = f_next + penalty * _norm(b_next)
            actual_reduction = merit_function - merit_function_next
            reduction_ratio = actual_reduction / predicted_reduction
            if reduction_ratio < SUFFICIENT_REDUCTION_RATIO and \
                    _norm(dn) <= SOC_THRESHOLD * _norm(dt):
                y = -P.row_space(b_next)
                _, t, intersect = box_intersections(d, y, trust_lb, trust_ub)
                x_soc = x + S * (d + t * y)
                f_soc, b_soc = self.f_and_c(x_soc)
                merit_function_soc = f_soc + penalty * _norm(b_soc)
                actual_reduction_soc = merit_function - merit_function_soc
                reduction_ratio_soc = actual_reduction_soc / predicted_reduction
                if intersect and reduction_ratio_soc >= SUFFICIENT_REDUCTION_RATIO:
                    x_next = x_soc
                    f_next = f_soc
                    b_next = b_soc
                    reduction_ratio = reduction_ratio_soc
            if reduction_ratio >= LARGE_REDUCTION_RATIO:
                trust_radius = max(TRUST_ENLARGEMENT_FACTOR_L * _norm(d), trust_radius)
            elif reduction_ratio >= INTERMEDIARY_REDUCTION_RATIO:
                trust_radius = max(TRUST_ENLARGEMENT_FACTOR_S * _norm(d), trust_radius)
            elif reduction_ratio < SUFFICIENT_REDUCTION_RATIO:
                trust_reduction = (1 - SUFFICIENT_REDUCTION_RATIO) / (1 - reduction_ratio)
                new_trust_radius = trust_reduction * _norm(d)
                if new_trust_radius >= MAX_TRUST_REDUCTION * trust_radius:
                    trust_radius *= MAX_TRUST_REDUCTION
                elif new_trust_radius >= MIN_TRUST_REDUCTION * trust_radius:
                    trust_radius = new_trust_radius
                else:
                    trust_radius *= MIN_TRUST_REDUCTION
            if reduction_ratio >= SUFFICIENT_REDUCTION_RATIO:
                x = x_next
                f = f_next
                b = b_next
                c, A = self.g_and_j(x)
                S = self.scaling(x)
                P = _Proj(A, self.sparse)
                v = -P.least_squares(c)
                H = self.lagr_hess(x, v)
                failed = False
                optimality = _norm_inf(c + A.T.dot(v))
                constr_violation = _norm_inf(b) if b.size > 0 else 0.0
            else:
                penalty = previous_penalty
                failed = True
        return x

    # ---------------------------------------------- interior point
    def interior_point(self, x0: np.ndarray[float, 1], c_ineq0, J_ineq0, c_eq0, J_eq0,
                       initial_barrier_parameter: float, initial_tolerance: float,
                       initial_penalty: float, initial_trust_radius: float):
        BOUNDARY_PARAMETER = 0.995
        BARRIER_DECAY_RATIO = 0.2
        TRUST_ENLARGEMENT = 5.0
        self.barrier = True
        self.bp = initial_barrier_parameter
        self.btol = initial_tolerance
        trust_radius = initial_trust_radius
        s0 = np.maximum(-1.5 * c_ineq0, np.ones(self.n_ineq))
        f0 = self._bfun(self.sf.f, c_ineq0, s0)
        g0 = self._bgrad(self.sf.g)
        cons0 = self._bconstr(c_ineq0, c_eq0, s0)
        jac0 = self._bjac(J_eq0, J_ineq0, s0)
        z = np.concatenate((x0, s0))
        trust_lb = np.concatenate((np.full(self.n, -INF), np.full(self.n_ineq, -BOUNDARY_PARAMETER)))
        trust_ub = np.full(self.n + self.n_ineq, INF)
        while True:
            z = self.sqp(z, f0, g0, cons0, jac0, initial_penalty, trust_radius,
                         trust_lb, trust_ub)
            if self.terminate:
                break
            trust_radius = max(initial_trust_radius, TRUST_ENLARGEMENT * self.tr_radius)
            self.bp *= BARRIER_DECAY_RATIO
            self.btol *= BARRIER_DECAY_RATIO
            f0, cons0 = self.f_and_c(z)
            g0, jac0 = self.g_and_j(z)
        return self._x(z)

def _minimize_trustregion_constr(fun, x0: np.ndarray[float, 1], args, jac, hess,
                                 hessp, bnds, constraints, callback, options):
    xtol = _opt_float(options, 'xtol', 1e-8)
    gtol = _opt_float(options, 'gtol', 1e-8)
    barrier_tol = _opt_float(options, 'barrier_tol', 1e-8)
    maxiter = _opt_int(options, 'maxiter', 1000)
    initial_constr_penalty = _opt_float(options, 'initial_constr_penalty', 1.0)
    initial_tr_radius = _opt_float(options, 'initial_tr_radius', 1.0)
    initial_barrier_parameter = _opt_float(options, 'initial_barrier_parameter', 0.1)
    initial_barrier_tolerance = _opt_float(options, 'initial_barrier_tolerance', 0.1)
    rel: Optional[float] = None
    if _has_opt(options, 'finite_diff_rel_step'):
        rel = _opt_float(options, 'finite_diff_rel_step', 0.0)
    n = x0.size
    fdl: Optional[np.ndarray[float, 1]] = None
    fdu: Optional[np.ndarray[float, 1]] = None
    cs = _as_tuple(constraints)
    if hess is None:
        if hessp is not None:
            return _tc_run(fun, x0, args, jac, _HessPOp(hessp), hessp, True, bnds, cs,
                           callback, n, rel, xtol, gtol, barrier_tol, maxiter,
                           initial_constr_penalty, initial_tr_radius,
                           initial_barrier_parameter, initial_barrier_tolerance)
        return _tc_run(fun, x0, args, jac, BFGS(), None, False, bnds, cs, callback, n,
                       rel, xtol, gtol, barrier_tol, maxiter, initial_constr_penalty,
                       initial_tr_radius, initial_barrier_parameter,
                       initial_barrier_tolerance)
    return _tc_run(fun, x0, args, jac, hess, None, False, bnds, cs, callback, n, rel,
                   xtol, gtol, barrier_tol, maxiter, initial_constr_penalty,
                   initial_tr_radius, initial_barrier_parameter,
                   initial_barrier_tolerance)

def _tc_run(fun, x0: np.ndarray[float, 1], args, jac, hess, hessp, use_hp: bool, bnds,
            cs, callback, n: int, rel: Optional[float], xtol: float, gtol: float,
            barrier_tol: float, maxiter: int, initial_constr_penalty: float,
            initial_tr_radius: float, initial_barrier_parameter: float,
            initial_barrier_tolerance: float):
    gj = '2-point' if jac is None else jac
    sf = _ScalarFunctionH(fun, x0, args, gj, hess, None, rel)
    ps0 = _prep_all(cs, x0, None, None, '2-point', rel, False)
    if bnds is not None:
        lb0, ub0 = bnds
        mlb = lb0.copy()
        mub = ub0.copy()
        for i in range(n):
            if math.isfinite(lb0[i]):
                mlb[i] = float(np.nextafter(lb0[i], -INF))
            if math.isfinite(ub0[i]):
                mub[i] = float(np.nextafter(ub0[i], INF))
        # bounds become a sparse IdentityVectorFunction (sparse_jacobian
        # defaults to True when only bounds are present)
        sparse = True
        if static.len(cs) > 0:
            sparse = False
        bp = _Prep(_LinVF(np.eye(n), x0, sparse), mlb, mub, np.zeros(1, bool))
        ps = ps0 + (bp,)
        return _tc_solve(sf, hessp, use_hp, ps, callback, n, sparse, xtol, gtol,
                         barrier_tol, maxiter, initial_constr_penalty,
                         initial_tr_radius, initial_barrier_parameter,
                         initial_barrier_tolerance, x0)
    return _tc_solve(sf, hessp, use_hp, ps0, callback, n, False, xtol, gtol,
                     barrier_tol, maxiter, initial_constr_penalty, initial_tr_radius,
                     initial_barrier_parameter, initial_barrier_tolerance, x0)

def _tc_solve(sf, hessp, use_hp: bool, ps, callback, n: int, sparse: bool,
              xtol: float, gtol: float, barrier_tol: float, maxiter: int,
              initial_constr_penalty: float, initial_tr_radius: float,
              initial_barrier_parameter: float, initial_barrier_tolerance: float,
              x0: np.ndarray[float, 1]):
    cons = _AllCons(ps, n)
    tc = _TC(sf, hessp, sf.args, use_hp, cons, callback, n, sparse, gtol, xtol,
             barrier_tol, maxiter)
    c_eq0, c_ineq0, J_eq0, J_ineq0 = cons.initial()
    if cons.n_ineq == 0:
        tc.sqp(x0, sf.f, sf.g.copy(), c_eq0, J_eq0, initial_constr_penalty,
               initial_tr_radius, np.full(n, -INF), np.full(n, INF))
    else:
        tc.interior_point(x0, c_ineq0, J_ineq0, c_eq0, J_eq0,
                          initial_barrier_parameter, initial_barrier_tolerance,
                          initial_constr_penalty, initial_tr_radius)
    status = tc.status
    if (status == 1 or status == 2) and tc.constr_violation > gtol:
        status = 4
    tc.status = status
    r = tc.result()
    r.success = status == 1 or status == 2
    r.message = _TC_MESSAGES[status]
    return r
