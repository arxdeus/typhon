# Copyright (C) 2026 the Typhon authors
# Ported from autoreject 0.5.2 (BSD-3-Clause)

"""Automated rejection and repair of epochs in M/EEG (autoreject 0.5.2).

Native Typhon port. Differences from the Python package:

- EEG data only: MEG field interpolation (``_fast_map_meg_channels``,
  ``_compute_dots``) raises NotImplementedError. Interpolation follows the
  MNE-Python < 1.14 code path (spherical splines, ``dots`` always None).
- Plotting (``RejectLog.plot``, ``RejectLog.plot_epochs``,
  ``set_matplotlib_defaults``) and progress bars are no-ops.
- ``n_jobs`` is accepted but work runs serially (results are identical).
- ``AutoReject.save`` / ``read_auto_reject`` use JSON instead of HDF5.
  ``RejectLog.save`` writes .npz with ch_names encoded as UTF-8 byte codes.
- ``thresh_func`` arguments are ignored (compute_thresholds is used).
- ``AutoReject.cv`` must be an int (KFold). ``transform(return_log=...)``
  needs a literal bool.
"""

__version__: Literal[str] = "0.5.2"

from ._thresholds import (_GlobalAutoReject, _ChannelAutoReject, compute_thresholds,
                          validation_curve, get_rejection_threshold)
from ._local import _AutoReject, RejectLog, read_reject_log
from .autoreject import AutoReject, read_auto_reject
from .ransac import Ransac
from .utils import set_matplotlib_defaults
