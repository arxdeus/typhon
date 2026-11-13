# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_integrate_odesolver.py (RK23/RK45/DOP853/Radau/BDF/LSODA classes).
import numpy as np
from scipy.integrate import RK45, RK23, DOP853, Radau, BDF, LSODA
from _gen_common import Gen
def f(t, y):
    return np.array([-0.5 * y[0] + y[1], -y[1] * 0.3 - y[0]])
ns = {'np': np}
g = Gen(ns, tol=1e-9)
for cls in [RK45, RK23, DOP853, Radau, BDF, LSODA]:
    nm = cls.__name__
    g.test('test_' + nm.lower())
    g.raw('    def f(t: float, y):\n        return np.array([-0.5 * y[0] + y[1], -y[1] * 0.3 - y[0]])')
    s = cls(f, 0.0, np.array([1.0, 0.5]), 5.0, rtol=1e-6, atol=1e-9)
    n = 0
    mid = None
    while s.status == 'running':
        s.step(); n += 1
        if n == 5:
            d5 = s.dense_output(); mid = (s.t_old, s.t, d5(0.5 * (s.t_old + s.t)))
    d = s.dense_output()
    ns['d'] = d
    g.raw(f'    s = {nm}(f, 0.0, np.array([1.0, 0.5]), 5.0, rtol=1e-6, atol=1e-9)')
    g.raw('    n = 0\n    mid = np.zeros(2)\n    tm = 0.0\n    while s.status == "running":\n        s.step()\n        n += 1\n        if n == 5:\n            d5 = s.dense_output()\n            tm = 0.5 * (s.t_old + s.t)\n            mid = d5(tm)')
    exact = nm in ('RK45', 'RK23')
    ref = np.array([float(n), s.t, s.y[0], s.y[1], float(s.nfev)])
    if exact:
        g.raw(f'    assert closec(np.array([float(n), s.t, s.y[0], s.y[1], float(s.nfev)]), {[float(v) for v in ref]!r}, 1e-9)')
    else:
        g.raw(f'    assert closec(np.array([s.t, s.y[0], s.y[1]]), {[float(v) for v in ref[1:4]]!r}, 1e-5)')
        g.raw(f'    assert abs(float(n) - {float(n)!r}) <= 0.3 * {float(n)!r} + 2.0')
    g.raw('    d = s.dense_output()')
    g.E('d(4.8)', 'd(4.8)', tol=1e-7 if exact else 2e-4)
    g.E('d(np.array([4.7, 4.9])).ravel()', 'd(np.array([4.7, 4.9])).ravel()', tol=1e-7 if exact else 2e-4)
    g.raw(f'    assert closec(mid, {[float(v) for v in mid[2]]!r}, {1e-7 if exact else 2e-4!r})')
    g.raw(f'    assert closec(np.array([tm]), {[0.5 * (mid[0] + mid[1])]!r}, {1e-12 if exact else 1e-3!r})')
g.write('test/scipy/test_integrate_odesolver.py', 'gen_integrate_odesolver.py', ['import numpy as np', 'from scipy.integrate import RK45, RK23, DOP853, Radau, BDF, LSODA'], [])
