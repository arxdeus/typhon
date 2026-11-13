# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.preprocessing subset: ICA (fastica / infomax), EOG/ECG event and
# component detection, peak_finder, EOG regression and muscle annotation.

from .ica import ICA, _find_outliers
from ._infomax import infomax
from ._events import peak_finder, find_eog_events, find_ecg_events, qrs_detector
from ._regress import EOGRegression, regress_artifact, annotate_muscle_zscore
from ._epochs_ext import create_ecg_epochs, create_eog_epochs, ctps, kuiper
from .xdawn import Xdawn
from ._csd import compute_current_source_density
from ._annotate import annotate_amplitude, annotate_break
from ._lof import find_bad_channels_lof
from ._misc import fix_stim_artifact, annotate_nan, equalize_bads, compute_bridged_electrodes
from ._ssp import compute_proj_ecg, compute_proj_eog

# beamformer lane
from ._eog_regression import EOGRegression
