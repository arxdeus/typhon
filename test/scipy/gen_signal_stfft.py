# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_signal_stfft.py (ShortTimeFFT, closest_STFT_dual_window).
import numpy as np
from scipy.signal import ShortTimeFFT, closest_STFT_dual_window
from _gen_common import Gen, lit
x = np.array([np.sin(0.3 * i) + 0.2 * np.cos(1.7 * i) for i in range(60)])
ns = {'np': np, 'x': x}
g = Gen(ns, tol=1e-9)
g.test('test_stfft')
g.raw('    x = np.array([np.sin(0.3 * i) + 0.2 * np.cos(1.7 * i) for i in range(60)])')
def num(v): return repr(float(v))
def case(setup, label):
    S = eval(setup, {'np': np, 'ShortTimeFFT': ShortTimeFFT})
    ns['S'] = S
    g.raw(f'    S = {setup}')
    g.raw('    Z = S.stft(x)')
    Z = S.stft(x); ns['Z'] = Z
    g.raw(f'    assert Z.shape == {Z.shape!r}')
    g.E('np.abs(Z).ravel()', 'np.abs(Z).ravel()', tol=1e-9)
    g.E('np.angle(Z[1, :]).ravel()', 'np.angle(Z[1, :]).ravel()', tol=1e-7)
    g.E('S.istft(Z)[:60]', 'S.istft(Z)[:60].real', tol=1e-9)
    g.E('S.f', 'S.f'); g.E('S.t(60)', 'S.t(60)')
    g.E('np.array([float(S.p_min), float(S.p_max(60)), float(S.k_min), float(S.k_max(60)), float(S.f_pts), float(S.lower_border_end[0]), float(S.lower_border_end[1]), float(S.upper_border_begin(60)[0]), float(S.upper_border_begin(60)[1]), S.delta_t, S.delta_f, S.fac_magnitude, S.fac_psd])',
        'np.array([float(S.p_min), float(S.p_max(60)), float(S.k_min), float(S.k_max(60)), float(S.f_pts), float(S.lower_border_end[0]), float(S.lower_border_end[1]), float(S.upper_border_begin(60)[0]), float(S.upper_border_begin(60)[1]), S.delta_t, S.delta_f, S.fac_magnitude, S.fac_psd])')
    if S.fft_mode != 'twosided':
        g.E('np.array(S.extent(60))', 'np.array(S.extent(60))')
    g.E('S.dual_win', 'S.dual_win')
for mode in ['onesided', 'twosided', 'centered']:
    case(f"ShortTimeFFT.from_window('hann', 1.0, 16, 8, fft_mode='{mode}', scale_to='psd')", mode)
case("ShortTimeFFT(np.hamming(12), 3, 2.0, mfft=20, phase_shift=None, fft_mode='onesided2X', scale_to='magnitude')", 'x2')
case("ShortTimeFFT.from_window(('gaussian', 4.0), 100.0, 20, 15, mfft=32, phase_shift=3)", 'gauss')
case("ShortTimeFFT.from_window('hamming', 1.0, 11, 5, symmetric_win=True, fft_mode='onesided2X', scale_to='psd')", 'odd')
case("ShortTimeFFT.from_win_equals_dual(np.hanning(16), 4, 1.0, scale_to='unitary', fft_mode='twosided')", 'wed')
S = ShortTimeFFT(np.hamming(12), 3, 2.0, mfft=20, phase_shift=None, fft_mode='onesided2X', scale_to='magnitude')
ns['S'] = S
g.raw("    S = ShortTimeFFT(np.hamming(12), 3, 2.0, mfft=20, phase_shift=None, fft_mode='onesided2X', scale_to='magnitude')")
g.E("np.abs(S.stft(x, padding='even')).ravel()", "np.abs(S.stft(x, padding='even')).ravel()")
g.E("np.abs(S.stft(x, padding='edge', p0=2, p1=9, k_offset=3)).ravel()", "np.abs(S.stft(x, padding='edge', p0=2, p1=9, k_offset=3)).ravel()")
g.E("np.abs(S.spectrogram(x, x ** 2, detr='linear')).ravel()", "np.abs(S.spectrogram(x, x ** 2, detr='linear')).ravel()")
g.E("S.spectrogram(x, detr='constant').ravel()", "S.spectrogram(x, detr='constant').ravel()")
g.test('test_dual')
g.raw('    w = np.hamming(12)')
for sc in [True, False]:
    wd, a = closest_STFT_dual_window(np.hamming(12), 3, scaled=sc)
    ns['wd'] = wd; ns['a'] = a
    g.raw(f'    wd, a = closest_STFT_dual_window(w, 3, scaled={sc})')
    g.E('wd', 'wd'); g.E('np.array([a])', 'np.array([float(a)])')
wd, a = closest_STFT_dual_window(np.hamming(12), 4, np.hanning(12))
ns['wd'] = wd; ns['a'] = a
g.raw('    wd, a = closest_STFT_dual_window(w, 4, np.hanning(12))')
g.E('wd', 'wd'); g.E('np.array([a])', 'np.array([float(a)])')
g.write('test/scipy/test_signal_stfft.py', 'gen_signal_stfft.py', ['import numpy as np', 'from scipy.signal import ShortTimeFFT, closest_STFT_dual_window'], [])
