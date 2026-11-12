# Ported from braindecode 2.0.0 (BSD-3-Clause)

import numpy as np
import pandas as pd
from mne.io import RawArray
from .base import BaseConcatDataset, RawDataset, EEGWindowsDataset, Description, _dict_or_none


def create_from_mne_raw(raws: List[RawArray], trial_start_offset_samples: int,
                        trial_stop_offset_samples: int, window_size_samples: int,
                        window_stride_samples: int, drop_last_window: bool,
                        descriptions = None, mapping: Optional[Dict[str, int]] = None,
                        preload: bool = False, drop_bad_windows: bool = True,
                        accepted_bads_ratio: float = 0.0):
    """Windows from annotated raws (create_windows_from_events)."""
    from ..preprocessing.windowers import create_windows_from_events
    base = List[RawDataset]()
    if isinstance(descriptions, NoneType):
        for r in raws:
            base.append(RawDataset(r))
    else:
        if len(descriptions) != len(raws):
            raise ValueError(f"length of 'raws' ({len(raws)}) and 'description' ({len(descriptions)}) has to match")
        for i in range(len(raws)):
            base.append(RawDataset(raws[i], descriptions[i]))
    return create_windows_from_events(BaseConcatDataset(base),
                                      trial_start_offset_samples=trial_start_offset_samples,
                                      trial_stop_offset_samples=trial_stop_offset_samples,
                                      window_size_samples=window_size_samples,
                                      window_stride_samples=window_stride_samples,
                                      on_last_window="drop" if drop_last_window else "overlap",
                                      mapping=mapping, accepted_bads_ratio=accepted_bads_ratio)


def create_from_mne_epochs(list_of_epochs, window_size_samples: int, window_stride_samples: int,
                           drop_last_window: bool, descriptions = None,
                           mapping: Optional[Dict[str, int]] = None, preload: bool = False,
                           picks = None, drop_bad_windows: bool = True):
    """Windows from mne EpochsArray objects: each epoch becomes one recording
    (served as an EEGWindowsDataset instead of mne.Epochs)."""
    if window_size_samples <= 0 or window_stride_samples <= 0:
        raise AssertionError("window size and stride have to be larger than 0")
    if not isinstance(descriptions, NoneType):
        if len(descriptions) != len(list_of_epochs):
            raise ValueError(f"length of 'list_of_epochs' ({len(list_of_epochs)}) and 'descriptions' ({len(descriptions)}) has to match")
    out = List[EEGWindowsDataset]()
    for i in range(len(list_of_epochs)):
        ep = list_of_epochs[i]
        n_t = ep.times.shape[0]
        stop = n_t - window_size_samples
        starts = List[int]()
        s = 0
        while s < stop + 1:
            starts.append(s)
            s += window_stride_samples
        if not drop_last_window and starts[-1] < stop:
            starts.append(stop)
        for tr in range(ep._data.shape[0]):
            code = int(ep.events[tr, 2])
            target = code
            if mapping is not None:
                m = mapping.__val__()
                if str(code) in m:
                    target = m[str(code)]
            t0 = int(ep.events[tr, 0])
            cols = Dict[str, List[int]]()
            cols["i_window_in_trial"] = list(range(len(starts)))
            cols["i_start_in_trial"] = [v + t0 for v in starts]
            cols["i_stop_in_trial"] = [v + t0 + window_size_samples for v in starts]
            cols["target"] = [target for _ in range(len(starts))]
            md = pd.DataFrame(cols)
            raw = RawArray(ep._data[tr], ep.info)
            # windows are read relative to the trial array
            w = EEGWindowsDataset(raw, md, descriptions[i] if not isinstance(descriptions, NoneType) else Description())
            w._read_offset = t0
            out.append(w)
    return BaseConcatDataset(out)
