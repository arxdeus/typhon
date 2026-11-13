# Ported from mne 1.13.2 (BSD-3-Clause)

from .psd import psd_array_welch
from .multitaper import psd_array_multitaper, dpss_windows
from .._containers import Spectrum, EpochsSpectrum
from .tfr import (morlet, fwhm, cwt, tfr_array_morlet, tfr_array_multitaper,
                  tfr_array_stockwell, stft, istft, stftfreq)
from ._tfr_containers import (AverageTFR, AverageTFRArray, EpochsTFR, EpochsTFRArray,
                              tfr_morlet, tfr_multitaper, tfr_stockwell, compute_tfr)
from .csd import (CrossSpectralDensity, csd_array_fourier, csd_array_multitaper,
                  csd_array_morlet, csd_fourier, csd_multitaper, csd_morlet)
