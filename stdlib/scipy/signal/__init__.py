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
