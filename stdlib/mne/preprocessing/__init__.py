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
