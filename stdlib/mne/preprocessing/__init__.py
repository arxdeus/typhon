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
from ._otp import oversampled_temporal_projection

# maxwell / ICA extras lane
from ._maxwell_filter import (maxwell_filter, compute_maxwell_basis, find_bad_channels_maxwell,
                              maxwell_filter_prepare_emptyroom)
from ._ica_ext import ica_find_ecg_events, ica_find_eog_events, corrmap, get_score_funcs
from ._movement import annotate_movement, compute_average_dev_head_t
from ._bridged import interpolate_bridged_electrodes
from ._pca_obs import apply_pca_obs
from ._fmri import GradientRemover, remove_fmri_gradient_artifact
from ._realign import realign_raw
from ._css import cortical_signal_suppression
from ._ica_io import read_ica
from ._hfc import compute_proj_hfc, read_fine_calibration, write_fine_calibration
from ._fine_cal import compute_fine_calibration
