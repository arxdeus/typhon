# Ported from scipy 1.18.1 (BSD-3-Clause)

# Constraint evaluation shared by SLSQP / COBYLA / trust-constr:
# scipy's standardize_constraints + new_constraint_to_old, evaluated
# directly. Constraints may be dicts {'type', 'fun', 'jac'}, or
# LinearConstraint / NonlinearConstraint objects (or a tuple/list of them).
# Values are reported in the "old" convention: equality rows must be 0,
# inequality rows must be >= 0.

import numpy as np
from ._constraints import LinearConstraint, NonlinearConstraint
from ._common import _approx_jac, INF

def _u1(v) -> np.ndarray[float, 1]:
    # unwrap a (possibly Union-typed) function result to a 1-D float array
    if isinstance(v, Union):
        for tag, TU in static.vars_types(v, with_index=True):
            if Union._get_tag(v) == tag:
                return _u1(Union._get_data(v, TU))
        return np.zeros(0)
    else:
        return np.atleast_1d(np.asarray(v, float)).flatten()

def _u2(v, n: int) -> np.ndarray[float, 2]:
    if isinstance(v, Union):
        for tag, TU in static.vars_types(v, with_index=True):
            if Union._get_tag(v) == tag:
                return _u2(Union._get_data(v, TU), n)
        return np.zeros((0, n))
    else:
        a = np.asarray(v, float)
        if a.ndim == 1:
            # atleast_2d: a single constraint row
            return a.reshape((1, a.size)).astype(float)
        return np.atleast_2d(a).astype(float)

def _ustr(v) -> str:
    if isinstance(v, Union):
        for tag, TU in static.vars_types(v, with_index=True):
            if Union._get_tag(v) == tag:
                return _ustr(Union._get_data(v, TU))
        return ''
    elif isinstance(v, str):
        return v
    else:
        return ''

class _DictFun[D]:
    d: D

    def __init__(self, d: D):
        self.d = d

    def __call__(self, x) -> np.ndarray[float, 1]:
        return _u1(self.d['fun'](x))

class _NLFun[C]:
    c: C

    def __init__(self, c: C):
        self.c = c

    def __call__(self, x) -> np.ndarray[float, 1]:
        return _u1(self.c.fun(x))

def _is_seq_of_cons(constraints) -> Literal[bool]:
    if isinstance(constraints, Tuple):
        return True
    elif isinstance(constraints, List):
        return True
    else:
        return False

def _is_dict(c) -> Literal[bool]:
    if isinstance(c, LinearConstraint):
        return False
    elif isinstance(c, NonlinearConstraint):
        return False
    else:
        return True

class _Masks:
    # new_constraint_to_old row selection
    i_eq: np.ndarray[bool, 1]
    i_below: np.ndarray[bool, 1]
    i_above: np.ndarray[bool, 1]
    lb: np.ndarray[float, 1]
    ub: np.ndarray[float, 1]

    def __init__(self, lb: np.ndarray[float, 1], ub: np.ndarray[float, 1]):
        k = lb.size
        self.lb = lb
        self.ub = ub
        self.i_eq = np.zeros(k, bool)
        self.i_below = np.zeros(k, bool)
        self.i_above = np.zeros(k, bool)
        for i in range(k):
            eq = lb[i] == ub[i]
            self.i_eq[i] = eq
            self.i_below[i] = (lb[i] != -INF) != eq
            self.i_above[i] = (ub[i] != INF) != eq

    def split(self, y: np.ndarray[float, 1]):
        k = self.lb.size
        ne = 0
        nb = 0
        na = 0
        for i in range(k):
            if self.i_eq[i]:
                ne += 1
            if self.i_below[i]:
                nb += 1
            if self.i_above[i]:
                na += 1
        e = np.empty(ne)
        q = np.empty(nb + na)
        a = 0
        for i in range(k):
            if self.i_eq[i]:
                e[a] = y[i] - self.lb[i]
                a += 1
        a = 0
        for i in range(k):
            if self.i_below[i]:
                q[a] = y[i] - self.lb[i]
                a += 1
        for i in range(k):
            if self.i_above[i]:
                q[a] = -(y[i] - self.ub[i])
                a += 1
        return e, q

    def split_jac(self, J: np.ndarray[float, 2]):
        k = self.lb.size
        n = J.shape[1]
        re = List[int]()
        rq = List[int]()
        sg = List[float]()
        for i in range(k):
            if self.i_eq[i]:
                re.append(i)
        for i in range(k):
            if self.i_below[i]:
                rq.append(i)
                sg.append(1.0)
        for i in range(k):
            if self.i_above[i]:
                rq.append(i)
                sg.append(-1.0)
        E = np.empty((len(re), n))
        Q = np.empty((len(rq), n))
        for a in range(len(re)):
            for j in range(n):
                E[a, j] = J[re[a], j]
        for a in range(len(rq)):
            for j in range(n):
                Q[a, j] = sg[a] * J[rq[a], j]
        return E, Q

def _bcast(v: np.ndarray[float, 1], k: int) -> np.ndarray[float, 1]:
    if v.size == 1 and k != 1:
        return np.full(k, v[0])
    return v

def _one_con(c, x: np.ndarray[float, 1], want_jac: bool, eps: Optional[float],
             fd_method: str, rel_step: Optional[float],
             lb: Optional[np.ndarray[float, 1]], ub: Optional[np.ndarray[float, 1]]):
    # -> (eq, ineq, Jeq, Jineq) for one constraint object
    n = x.size
    if isinstance(c, LinearConstraint):
        y = c.A @ x
        m = _Masks(c.lb, c.ub)
        e, q = m.split(y)
        E, Q = m.split_jac(c.A)
        return e, q, E, Q
    elif isinstance(c, NonlinearConstraint):
        y = _u1(c.fun(x))
        m = _Masks(_bcast(c.lb, y.size), _bcast(c.ub, y.size))
        e, q = m.split(y)
        if not want_jac:
            return e, q, np.zeros((0, n)), np.zeros((0, n))
        J = np.zeros((0, n))
        if c.jac is None or isinstance(c.jac, str):
            # new_constraint_to_old drops non-callable jac: the solver's
            # own finite differences are used
            J, _ = _approx_jac(_NLFun(c), x, y, fd_method, rel_step, eps, lb, ub)
        else:
            J = _u2(c.jac(x), n)
        E, Q = m.split_jac(J)
        return e, q, E, Q
    else:
        t = _ustr(c['type']).lower()
        if t != 'eq' and t != 'ineq':
            raise ValueError(f"Unknown constraint type '{t}'.")
        y = _u1(c['fun'](x))
        J = np.zeros((0, n))
        if want_jac:
            if 'jac' in c:
                J = _u2(c['jac'](x), n)
            else:
                J, _ = _approx_jac(_DictFun(c), x, y, fd_method, rel_step, eps, lb, ub)
        if t == 'eq':
            return y, np.zeros(0), J, np.zeros((0, n))
        else:
            return np.zeros(0), y, np.zeros((0, n)), J

def _cat1(a: np.ndarray[float, 1], b: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    return np.concatenate((a, b))

def _cat2(a: np.ndarray[float, 2], b: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    if a.shape[0] == 0:
        return b.copy()
    if b.shape[0] == 0:
        return a
    return np.concatenate((a, b), axis=0)

def _eval_constraints(constraints, x: np.ndarray[float, 1], want_jac: bool,
                      eps: Optional[float] = None, fd_method: str = '2-point',
                      rel_step: Optional[float] = None,
                      lb: Optional[np.ndarray[float, 1]] = None,
                      ub: Optional[np.ndarray[float, 1]] = None):
    # all constraints -> (eq values, ineq values, Jeq, Jineq)
    n = x.size
    e = np.zeros(0)
    q = np.zeros(0)
    E = np.zeros((0, n))
    Q = np.zeros((0, n))
    if constraints is None:
        pass
    elif _is_seq_of_cons(constraints):
        if isinstance(constraints, Tuple):
            if static.len(constraints) == 0:
                return e, q, E, Q
        for c in constraints:
            e1, q1, E1, Q1 = _one_con(c, x, want_jac, eps, fd_method, rel_step, lb, ub)
            e = _cat1(e, e1)
            q = _cat1(q, q1)
            if want_jac:
                E = _cat2(E, E1)
                Q = _cat2(Q, Q1)
    else:
        e, q, E, Q = _one_con(constraints, x, want_jac, eps, fd_method, rel_step, lb, ub)
    return e, q, E, Q

def _has_constraints(constraints) -> bool:
    if constraints is None:
        return False
    elif _is_seq_of_cons(constraints):
        return len(constraints) > 0
    else:
        return True
