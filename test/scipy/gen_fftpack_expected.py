# Generates test/scipy/test_fftpack.py from the reference scipy 1.18.1.
import numpy as np, scipy.fftpack as F
from _gen_common import Gen
setup = ['x = np.cos(0.7 * np.arange(9.0)) + 0.3 * np.arange(9.0)', 'x8 = x[:8].copy()',
         'X = np.cos(np.arange(24.0) ** 1.2).reshape(4, 6)', 'p = np.sin(2 * np.pi * np.arange(16.0) / 16) + 0.5 * np.cos(6 * np.pi * np.arange(16.0) / 16)',
         'p15 = np.sin(2 * np.pi * np.arange(15.0) / 15) + 0.3 * np.cos(4 * np.pi * np.arange(15.0) / 15)']
ns = {'np': np, 'F': F}
for l in setup: exec(l, ns)
g = Gen(ns, tol=1e-11)
g.test('test_fftpack')
for v in ['x', 'x8']:
    g.E(f'F.rfft({v})'); g.E(f'F.irfft({v})'); g.E(f'F.rfft({v}, n=12)'); g.E(f'F.irfft(F.rfft({v}))')
g.E('F.rfft(X, axis=0)'); g.E('F.irfft(X)')
g.E('F.rfftfreq(9, 0.1)'); g.E('F.rfftfreq(8)')
g.raw(f'    assert F.next_fast_len(97) == {F.next_fast_len(97)}')
g.E('F.fft(x)'); g.E('F.dct(x, 3, norm="ortho")')
for t in (1, 2, 3, 4):
    g.E(f'F.idst(x, {t})'); g.E(f'F.idct(x, {t})'); g.E(f'F.idct(x, {t}, norm="ortho")')
g.E('F.idctn(X)'); g.E('F.idstn(X, 3)'); g.E('F.dctn(X, 1)')
for v in ['p', 'p15']:
    for o in [1, 2, 3]:
        g.E(f'F.diff({v}, {o})')
    g.E(f'F.diff({v}, 1, period=3.0)')
    g.E(f'F.tilbert({v}, 0.4)'); g.E(f'F.itilbert({v}, 0.4)'); g.E(f'F.hilbert({v})'); g.E(f'F.ihilbert({v})')
    g.E(f'F.cs_diff({v}, 0.3, 0.5)'); g.E(f'F.sc_diff({v}, 0.3, 0.5)'); g.E(f'F.ss_diff({v}, 0.3, 0.5)'); g.E(f'F.cc_diff({v}, 0.3, 0.5)')
    g.E(f'F.shift({v}, 0.7)'); g.E(f'F.shift({v}, 0.7, period=5.0)')
g.E('F.diff(p + 1j * p15[:15].sum(), 1)'); g.E('F.shift(p + 0.5j * p, 0.3)')
g.write('test_fftpack.py', 'gen_fftpack_expected.py', ['import numpy as np', 'import scipy.fftpack as F'], setup)
