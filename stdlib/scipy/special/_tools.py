# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3): tools.h root bracketing + Chandrupatla.
#
# ``func`` is any object callable as func(float) -> float (callable class
# instances are used instead of closures).

from . import _kernels as K

NAN = float('nan')
DBL_MIN = 2.2250738585072014e-308
DBL_MAX = 1.7976931348623157e308
EPS = 2.220446049250313e-16

def _sb(x: float) -> bool:
    return K._signbit(x)

def bracket_root(func, x0: float, xmin: float, xmax: float, step0_left: float,
                 step0_right: float, factor_left: float, factor_right: float,
                 increasing: bool, maxiter: int):
    """Returns (xl, xr, fl, fr, status) as xsf bracket_root_for_cdf_inversion."""
    y0 = func(x0)
    if y0 == 0:
        return (x0, x0, y0, y0, 0)
    y0_sgn = _sb(y0)
    search_left = False
    interior = x0
    y_interior = y0
    frontier = 0.0
    boundary = 0.0
    factor = 0.0
    if (increasing and y0 < 0) or (not increasing and y0 > 0):
        frontier = x0 + step0_right
        boundary = xmax
        factor = factor_right
    else:
        frontier = x0 + step0_left
        search_left = True
        boundary = xmin
        factor = factor_left
    y_interior_sgn = y0_sgn
    reached = False
    for i in range(maxiter):
        y_frontier = func(frontier)
        y_frontier_sgn = _sb(y_frontier)
        if y_frontier_sgn != y_interior_sgn or y_frontier == 0.0:
            if search_left:
                return (frontier, interior, y_frontier, y_interior, 0)
            return (interior, frontier, y_interior, y_frontier, 0)
        if reached:
            return (NAN, NAN, NAN, NAN, 1 if search_left else 2)
        step = (frontier - interior) * factor
        interior = frontier
        y_interior = y_frontier
        y_interior_sgn = y_frontier_sgn
        frontier += step
        if (search_left and frontier <= boundary) or (not search_left and frontier >= boundary):
            frontier = boundary
            reached = True
    return (NAN, NAN, NAN, NAN, 3)

def chandrupatla(func, x1: float, x2: float, f1: float, f2: float, rtol: float,
                 atol: float, maxiter: int):
    """Returns (root, status)."""
    if f1 == 0:
        return (x1, 0)
    if f2 == 0:
        return (x2, 0)
    t = 0.5
    x3 = 0.0
    f3 = 0.0
    for i in range(maxiter):
        x = x1 + t * (x2 - x1)
        f = func(x)
        if _sb(f) == _sb(f1):
            x3 = x1
            x1 = x
            f3 = f1
            f1 = f
        else:
            x3 = x2
            x2 = x1
            x1 = x
            f3 = f2
            f2 = f1
            f1 = f
        xm = x1
        fm = f1
        if K.fabs(f2) < K.fabs(f1):
            xm = x2
            fm = f2
        tol = 2.0 * rtol * K.fabs(xm) + 0.5 * atol
        tl = tol / K.fabs(x2 - x1)
        if tl > 0.5 or fm == 0:
            return (xm, 0)
        xi = (x1 - x2) / (x3 - x2)
        phi = (f1 - f2) / (f3 - f2)
        fl = 1.0 - K.sqrt(1.0 - xi)
        fh = K.sqrt(xi)
        if fl < phi and phi < fh:
            t = (f1 / (f2 - f1)) * (f3 / (f2 - f3)) + (f1 / (f3 - f1)) * (f2 / (f3 - f2)) * ((x3 - x1) / (x2 - x1))
        else:
            t = 0.5
        t = min(max(t, tl), 1.0 - tl)
    return (NAN, 1)

def invert_pos(func, increasing: bool) -> float:
    """Standard xsf positive-parameter CDF inversion: x0=1, bracket in
    [DBL_MIN, DBL_MAX] with power-of-two steps, then Chandrupatla.
    Status 1 -> 0, 2 -> inf, else nan."""
    r = bracket_root(func, 1.0, DBL_MIN, DBL_MAX, -0.875, 7.0, 0.125, 8.0, increasing, 342)
    st = r[4]
    if st == 1:
        return 0.0
    if st == 2:
        return K.INF
    if st >= 3:
        return NAN
    res = chandrupatla(func, r[0], r[1], r[2], r[3], EPS, 1e-100, 100)
    if res[1] != 0:
        return NAN
    return res[0]
