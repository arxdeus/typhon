# Ported from scipy 1.18.1 (BSD-3-Clause)

# Constraint objects of scipy.optimize (_constraints.py).

import numpy as np

def _arr1(v) -> np.ndarray[float, 1]:
    return np.atleast_1d(np.asarray(v, float)).flatten()

class LinearConstraint:
    # lb <= A @ x <= ub
    A: np.ndarray[float, 2]
    lb: np.ndarray[float, 1]
    ub: np.ndarray[float, 1]
    keep_feasible: bool

    def __init__(self, A, lb = -float('inf'), ub = float('inf'),
                 keep_feasible: bool = False):
        a = np.asarray(A, float)
        if a.ndim == 1:
            self.A = a.reshape((1, a.size))
        else:
            self.A = np.atleast_2d(a)
        m = self.A.shape[0]
        lb1 = _arr1(lb)
        ub1 = _arr1(ub)
        if lb1.size == 1 and m != 1:
            lb1 = np.full(m, lb1[0])
        if ub1.size == 1 and m != 1:
            ub1 = np.full(m, ub1[0])
        if lb1.size != m or ub1.size != m:
            raise ValueError("`lb` and `ub` must be broadcastable to the number of rows of `A`.")
        self.lb = lb1
        self.ub = ub1
        self.keep_feasible = keep_feasible

    def residual(self, x):
        Ax = self.A @ _arr1(x)
        return Ax - self.lb, self.ub - Ax

    def __repr__(self):
        return f"LinearConstraint({self.A}, {self.lb}, {self.ub})"

class NonlinearConstraint[FN, JC, HS]:
    # lb <= fun(x) <= ub; jac is a callable or '2-point'/'3-point',
    # hess is a callable, a HessianUpdateStrategy or None.
    fun: FN
    lb: np.ndarray[float, 1]
    ub: np.ndarray[float, 1]
    jac: JC
    hess: HS
    keep_feasible: bool

    def __init__(self, fun: FN, lb, ub, jac: JC = '2-point', hess: HS = None,
                 keep_feasible: bool = False):
        self.fun = fun
        self.lb = _arr1(lb)
        self.ub = _arr1(ub)
        self.jac = jac
        self.hess = hess
        self.keep_feasible = keep_feasible

    def __repr__(self):
        return f"NonlinearConstraint({self.lb}, {self.ub})"
