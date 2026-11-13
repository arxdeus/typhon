# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_signal_placepoles.py (place_poles YT / KNV0).
import numpy as np
from scipy.signal import place_poles
from _gen_common import Gen, lit
ns = {'np': np}
g = Gen(ns, tol=1e-7)
g.test('test_place_poles')
A4 = np.array([[1.380, -0.2077, 6.715, -5.676], [-0.5814, -4.290, 0.0, 0.6750], [1.067, 4.273, -6.654, 5.893], [0.0480, 4.273, 1.343, -2.104]])
B42 = np.array([[0.0, 5.679], [1.136, 1.136], [0.0, 0.0], [-3.146, 0.0]])
g.raw(f'    A4 = np.array({A4.tolist()!r})\n    B42 = np.array({B42.tolist()!r})')
def case(A, B, poles, method, an, bn, ptxt):
    f = place_poles(A, B, poles, method=method)
    ns['f'] = f
    g.raw(f"    f = place_poles({an}, {bn}, {ptxt}, method='{method}')")
    g.E('f.gain_matrix.ravel()', 'f.gain_matrix.ravel()')
    g.E('np.array([f.computed_poles.real.max(), f.computed_poles.real.min(), np.abs(f.computed_poles.imag).max()])', 'np.array([f.computed_poles.real.max(), f.computed_poles.real.min(), np.abs(f.computed_poles.imag).max()])', tol=1e-6)
    g.E('f.requested_poles.real', 'f.requested_poles.real')
    g.E('f.requested_poles.imag', 'f.requested_poles.imag')
    nan = np.isnan(f.rtol)
    if not nan:
        g.raw(f'    assert abs(f.nb_iter - {float(f.nb_iter)!r}) < 1e-9')
        g.raw(f'    assert f.rtol < {10 * float(f.rtol) + 1e-12!r}')
    else:
        g.raw('    assert f.rtol != f.rtol')
case(A4, B42, np.array([-0.2, -0.5, -5.0566, -8.6659]), 'YT', 'A4', 'B42', 'np.array([-0.2, -0.5, -5.0566, -8.6659])')
case(A4, B42, np.array([-0.2, -0.5, -5.0566, -8.6659]), 'KNV0', 'A4', 'B42', 'np.array([-0.2, -0.5, -5.0566, -8.6659])')
case(A4, B42, np.array([-3.0, -1.0, -2 - 1j, -2 + 1j]) / 3., 'YT', 'A4', 'B42', 'np.array([-3.0, -1.0, -2 - 1j, -2 + 1j]) / 3.')
case(A4, B42, np.array([-1.0 + 1j, -1.0 - 1j, -2.0 + 0.5j, -2.0 - 0.5j]), 'YT', 'A4', 'B42', 'np.array([-1.0 + 1j, -1.0 - 1j, -2.0 + 0.5j, -2.0 - 0.5j])')
A2 = np.array([[0.0, 1.0], [-2.0, -3.0]]); B21 = np.array([[0.0], [1.0]])
g.raw('    A2 = np.array([[0.0, 1.0], [-2.0, -3.0]])\n    B21 = np.array([[0.0], [1.0]])')
case(A2, B21, [-4.0, -5.0], 'YT', 'A2', 'B21', '[-4.0, -5.0]')
A22 = np.array([[1.0, 2.0], [3.0, 4.0]])
g.raw('    A22 = np.array([[1.0, 2.0], [3.0, 4.0]])')
case(A22, np.eye(2), [-1.0 + 2j, -1.0 - 2j], 'YT', 'A22', 'np.eye(2)', '[-1.0 + 2j, -1.0 - 2j]')
A6 = np.diag([1.0, -2.0, 3.0, 0.5, -1.0, 2.0]) + 0.1 * np.ones((6, 6))
B63 = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0], [0, 1.0, 1.0], [1.0, 0, 1.0]])
g.raw(f'    A6 = np.array({A6.tolist()!r})\n    B63 = np.array({B63.tolist()!r})')
case(A6, B63, np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]), 'YT', 'A6', 'B63', 'np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])')
case(A6, B63, np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]), 'KNV0', 'A6', 'B63', 'np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])')
g.write('test/scipy/test_signal_placepoles.py', 'gen_signal_placepoles.py', ['import numpy as np', 'from scipy.signal import place_poles'], [])
