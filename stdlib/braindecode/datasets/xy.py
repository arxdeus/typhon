# Ported from braindecode 2.0.0 (BSD-3-Clause)

import numpy as np
import mne
from mne.io import RawArray
from .base import BaseConcatDataset, RawDataset, Description


def create_from_X_y(X, y, drop_last_window: bool, sfreq: float, ch_names = None,
                    window_size_samples: Optional[int] = None,
                    window_stride_samples: Optional[int] = None):
    """Windows dataset from pre-cut trials X (n_trials, n_channels, n_times)
    and integer targets y."""
    from ..preprocessing.windowers import create_fixed_length_windows
    names = List[str]()
    if isinstance(ch_names, NoneType):
        for i in range(X.shape[1]):
            names.append(str(i))
    else:
        for c in ch_names:
            names.append(str(c))
    info = mne.create_info(names, sfreq)
    base = List[RawDataset]()
    lens = List[int]()
    for i in range(len(X)):
        x = np.asarray(X[i])
        lens.append(x.shape[1])
        d = Description()
        d["target"] = int(y[i])
        base.append(RawDataset(RawArray(x, info), d, target_name="target"))
    ws = window_size_samples
    st = window_stride_samples
    if ws is None and st is None:
        for l in lens:
            if l != lens[0]:
                raise ValueError("if 'window_size_samples' and 'window_stride_samples' are None, "
                                 "all trials have to have the same length")
        ws = lens[0]
        st = lens[0]
    return create_fixed_length_windows(BaseConcatDataset(base), start_offset_samples=0,
                                       stop_offset_samples=None, window_size_samples=ws,
                                       window_stride_samples=st,
                                       on_last_window="drop" if drop_last_window else "overlap")
