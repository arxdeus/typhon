# Ported from mne 1.13.2 (BSD-3-Clause)

from .csp import CSP, SPoC
from .transformer import (Scaler, Vectorizer, PSDEstimator, FilterEstimator, TemporalFilter,
                          UnsupervisedSpatialFilter)
from .base import (LinearModel, get_coef, cross_val_multiscore, SlidingEstimator,
                   GeneralizingEstimator, EMS, compute_ems)
from .receptive_field import ReceptiveField, TimeDelayingRidge
from .ssd import SSD
from .xdawn import XdawnTransformer, SpatialFilter, get_spatial_filter_from_estimator, TimeFrequency
