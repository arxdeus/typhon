# Generates the expected literals for test/scipy/test_integrate_bvp.py.
# Run with the reference venv: python test/scipy/gen_integrate_bvp.py
import numpy as np
from scipy.integrate import solve_bvp

def L(v):
    return repr(np.asarray(v, float).tolist())

def fun1(x, y):
    return np.vstack((y[1], -np.exp(y[0])))
def bc1(ya, yb):
    return np.array([ya[0], yb[0]])
x = np.linspace(0, 1, 5)
y = np.zeros((2, x.size))
r = solve_bvp(fun1, bc1, x, y)
print('bratu', r.status, r.niter, r.x.size, L(r.x[::4]), L(r.y[0][::4]), L(r.sol(np.array([0.25, 0.5, 0.75]))[0]), L(r.sol(np.array([0.3]), 1)[0]), float(r.rms_residuals.max()))

# unknown parameter: y'' + k^2 y = 0, y(0)=y(1)=0, y'(0)=k
def fun2(x, y, p):
    return np.vstack((y[1], -p[0] ** 2 * y[0]))
def bc2(ya, yb, p):
    return np.array([ya[0], yb[0], ya[1] - p[0]])
x = np.linspace(0, 1, 5)
y = np.zeros((2, x.size)); y[0] = np.sin(np.pi * x)
r = solve_bvp(fun2, bc2, x, y, p=[6])
print('eig', r.status, r.niter, r.x.size, L(r.p), L(r.y[0][::4]), L(r.sol(np.array([0.3]))[0]))

# analytic jacobians
def fun_jac(x, y):
    df_dy = np.zeros((2, 2, x.size))
    df_dy[0, 1, :] = 1
    df_dy[1, 0, :] = -np.exp(y[0])
    return df_dy
def bc_jac(ya, yb):
    return np.array([[1, 0], [0, 0]]), np.array([[0, 0], [1, 0]])
bc_jac2 = lambda ya, yb: (np.array([[1., 0], [0, 0]]), np.array([[0., 0], [1, 0]]))
x = np.linspace(0, 1, 5)
r = solve_bvp(fun1, bc1, x, np.zeros((2, 5)), fun_jac=fun_jac, bc_jac=bc_jac2, tol=1e-6)
print('jac', r.status, r.niter, r.x.size, L(r.y[0][::8]))

# singular term: y'' = -2/x y' ... S
def fun3(x, y):
    return np.vstack((y[1], -np.pi ** 2 * y[0] + 0 * x - 0 * y[1]))
S = np.array([[0, 0], [0, -1]])
def bc3(ya, yb):
    return np.array([ya[1], yb[0] - 1])
x = np.linspace(0, 1, 11)
r = solve_bvp(lambda x, y: np.vstack((y[1], 2 * y[0] * 0 + 1.0 + 0 * x)), bc3, x, np.zeros((2, 11)), S=S)
print('sing', r.status, r.niter, r.x.size, L(r.y[0][::2]))

# max_nodes exceeded and tol
r = solve_bvp(fun1, bc1, np.linspace(0, 1, 5), np.zeros((2, 5)), tol=1e-10, max_nodes=20)
print('maxn', r.status, r.niter, r.x.size, r.message)
