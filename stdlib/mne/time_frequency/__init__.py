# Ported from mne 1.13.2 (BSD-3-Clause)

from .psd import psd_array_welch
from .multitaper import psd_array_multitaper, dpss_windows
from .._containers import Spectrum, EpochsSpectrum
