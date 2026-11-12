# Generates the expected output embedded in test_linprog.py (scipy 1.18.1).
import numpy as np
def R(a):
    return [float(round(v, 9)) + 0.0 for v in a]
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
def show(r):
    if r.status == 0:
        print(r.status, R(r.x), float(round(r.fun, 9)) + 0.0)
        print(' ', R(r.slack), R(r.con))
        print(' ', R(r.ineqlin.marginals), R(r.eqlin.marginals), R(r.lower.marginals), R(r.upper.marginals))
    else:
        print(r.status, r.message)
show(linprog([-1.0, 4.0], A_ub=[[-3.0, 1.0], [1.0, 2.0]], b_ub=[6.0, 4.0], bounds=[(-np.inf, np.inf), (-3.0, np.inf)]))
show(linprog([1.0, 1.0], A_ub=[[-1.0, -1.0]], b_ub=[-5.0], bounds=[(0.0, 2.0), (0.0, 2.0)]))
show(linprog([-1.0, -1.0], A_ub=[[1.0, -1.0]], b_ub=[1.0]))
show(linprog([1.0, 2.0, 3.0], A_eq=[[1.0, 1.0, 1.0]], b_eq=[1.0]))
c = np.array([-1.0, -2.0, -3.0]); A = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.0], [0.0, 1.0, 3.0]]); b = np.array([10.0, 8.0, 12.0])
show(linprog(c, A_ub=A, b_ub=b, bounds=[(0.0, 6.0), (0.0, np.inf), (1.0, 3.0)]))
show(linprog([1.0, -1.0], A_eq=[[1.0, 1.0]], b_eq=[1.0], bounds=[(0.0, 1.0), (0.0, 0.7)]))
for seed in range(6):
    rng = np.random.default_rng(seed)
    m, n = 6 + seed, 5 + seed
    A = rng.standard_normal((m, n)); b = rng.random(m) + 0.5; c = rng.standard_normal(n)
    Ae = rng.standard_normal((2, n)); be = Ae @ (rng.random(n) * 0.2)
    show(linprog(c, A_ub=A, b_ub=b, A_eq=Ae, b_eq=be, bounds=(-1.0, 2.0)))
r = milp(-np.array([0.0, 1.0]), constraints=LinearConstraint([[-1.0, 1.0], [3.0, 2.0], [2.0, 3.0]], -np.inf, [1.0, 12.0, 12.0]), integrality=[1, 1])
print(r.status, r.x.tolist(), r.fun, r.message)
for seed in range(4):
    rng = np.random.default_rng(10 + seed)
    n = 6
    A = rng.random((5, n)) * 5; b = rng.random(5) * 10 + 5; c = -rng.random(n) * 3
    r = milp(c, constraints=LinearConstraint(A, -np.inf, b), integrality=np.ones(n), bounds=Bounds(0, 4))
    print(r.status, R(r.x), float(round(r.fun, 9)) + 0.0)
    r = linprog(c, A_ub=A, b_ub=b, bounds=(0, 4), integrality=[1, 0, 1, 0, 1, 0])
    print(r.status, R(r.x), float(round(r.fun, 9)) + 0.0)
r = milp([1.0, 1.0], constraints=LinearConstraint([[1.0, 1.0]], 1.5, 1.7), integrality=[1, 1]); print(r.status, r.message)
r = milp([-1.0, -1.0], constraints=LinearConstraint([[1.0, -1.0]], -np.inf, 1.0), integrality=[1, 1]); print(r.status, r.message)
r = linprog([1.0, 2.0], bounds=(-np.inf, 3.0)); print(r.status, r.message)
