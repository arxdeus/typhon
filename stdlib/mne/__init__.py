# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Native Typhon implementation of an MNE-Python subset: Info, EpochsArray,
# RawArray, Evoked, FIR filtering, Welch PSD and decoding.CSP.

__version__: Literal[str] = "1.13.2"

from ._info import Info, create_info
from ._containers import EpochsArray, Evoked, Spectrum, EpochsSpectrum
from .annotations import Annotations, events_from_annotations
from .filter import filter_data, create_filter, IIRParams
from ._containers import RawArray
from ._raw_ext import concatenate_raws
from .epochs import (Epochs, read_epochs, write_evokeds, read_evokeds, combine_evoked,
                     EvokedArray)
from .event import (find_events, find_stim_steps, make_fixed_length_events, merge_events,
                    pick_events, read_events, write_events, concatenate_events,
                    shift_time_events)
from .io import read_raw_fif
import channels
from .channels import make_standard_montage, make_dig_montage, equalize_channels
from ._info import DigMontage
import filter
import event
import io
import time_frequency
import decoding


def set_log_level(verbose = None, return_old_level: bool = False, add_frames = None):
    """No-op in Typhon (logging is not implemented)."""
    return None


def set_config(key: str, value = None, home_dir = None, set_env: bool = True):
    return None


def pick_types(info: Info, meg = False, eeg: bool = False, stim: bool = False,
               eog: bool = False, ecg: bool = False, emg: bool = False, misc: bool = False,
               exclude = "bads"):
    from ._info import _picks_to_idx
    out = List[int]()
    excl = info.bads if isinstance(exclude, str) else List[str]()
    for i in range(info.nchan):
        t = info.ch_types[i]
        if info.ch_names[i] in excl:
            continue
        ok = ((eeg and t == "eeg") or (stim and t == "stim") or (eog and t == "eog") or
              (ecg and t == "ecg") or (emg and t == "emg") or (misc and t == "misc"))
        if isinstance(meg, bool):
            if meg and (t == "mag" or t == "grad"):
                ok = True
        else:
            if t == meg:
                ok = True
        if ok:
            out.append(i)
    import numpy as np
    return np.array(out, dtype=int)
