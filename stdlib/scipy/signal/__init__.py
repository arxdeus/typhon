# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Native Typhon implementation of a scipy.signal subset.

from windows import get_window
import windows
from _arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext, zero_ext
from _filter_design import buttap, cheb1ap, cheb2ap, besselap, lp2lp_zpk, lp2hp_zpk, \
    lp2bp_zpk, lp2bs_zpk, bilinear_zpk, zpk2tf, tf2zpk, normalize, sos2tf, sos2zpk, \
    zpk2sos, tf2sos, iirfilter, butter, cheby1, cheby2, bessel, iirnotch, iirpeak, \
    freqz, freqz_sos, sosfreqz, freqz_zpk, ellipap, ellip, buttord, cheb1ord, ellipord, \
    group_delay
from _fir_filter_design import kaiser_beta, kaiser_atten, kaiserord, firwin, firwin2
from _signaltools import lfilter, lfilter_zi, sosfilt, sosfilt_zi, filtfilt, sosfiltfilt, \
    detrend, convolve, correlate, fftconvolve, oaconvolve, hilbert, medfilt, \
    savgol_coeffs, savgol_filter, upfirdn, resample, resample_poly, decimate
from _spectral import periodogram, welch, csd, coherence, spectrogram, stft, istft, check_COLA, check_NOLA
from _peak_finding import find_peaks, peak_prominences, peak_widths, argrelextrema, argrelmax, argrelmin
from _design_extra import lp2lp, lp2hp, lp2bp, lp2bs, bilinear, findfreqs, freqs, freqs_zpk, \
    cheb2ord, iirdesign, iircomb
from _ltisys import tf2ss, ss2tf, zpk2ss, ss2zpk, abcd_normalize, cont2discrete, lti, dlti, \
    lsim, impulse, step, bode, freqresp, dlsim, dimpulse, dstep, dfreqresp, dbode
from _signal_extra import remez, firls, minimum_phase, lfiltic, deconvolve, wiener, medfilt2d, \
    order_filter, hilbert2, correlation_lags, choose_conv_method, vectorstrength, sawtooth, \
    square, gausspulse, chirp, sweep_poly, unit_impulse, czt_points, czt, zoom_fft, CZT, ZoomFFT, \
    lombscargle
from _residue import unique_roots, residue, residuez, invres, invresz
from _splines import gauss_spline, cspline1d, qspline1d, cspline1d_eval, qspline1d_eval, \
    cspline2d, qspline2d, spline_filter, sepfir2d, symiirorder1, symiirorder2
from _signal_misc import max_len_seq, firwin_2d, gammatone, find_peaks_cwt, envelope
from _short_time_fft import ShortTimeFFT, closest_STFT_dual_window
