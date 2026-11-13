# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Native Typhon implementation of an MNE-Python subset: Info, EpochsArray,
# RawArray, Evoked, FIR filtering, Welch PSD and decoding.CSP.

__version__: Literal[str] = "1.13.2"

from ._info import Info, create_info
from ._containers import EpochsArray, Evoked, Spectrum, EpochsSpectrum
from .annotations import (Annotations, events_from_annotations, annotations_from_events,
                          count_annotations)
from .filter import filter_data, create_filter, IIRParams
from ._containers import RawArray
from ._raw_ext import concatenate_raws
from .epochs import (concatenate_epochs, make_fixed_length_epochs, equalize_epoch_counts,
                     combine_event_ids, Epochs, read_epochs, write_evokeds, read_evokeds, combine_evoked,
                     EvokedArray)
from .event import (find_events, find_stim_steps, make_fixed_length_events, merge_events,
                    pick_events, read_events, write_events, concatenate_events,
                    shift_time_events, count_events, match_event_names, define_target_events)
from .io import read_raw_fif
import channels
from .channels import make_standard_montage, make_dig_montage, equalize_channels
from ._info import DigMontage, Projection
from ._proj import (compute_proj_raw, compute_proj_epochs, compute_proj_evoked)
import filter
from ._epochs_more import grand_average
import transforms
import event
import io
import time_frequency
import decoding
from .cov import (Covariance, compute_covariance, compute_raw_covariance, make_ad_hoc_cov,
                  regularize, compute_whitener, prepare_noise_cov, whiten_evoked,
                  pick_channels_cov)
from .bem import make_sphere_model, ConductorModel
from .source_space import setup_volume_source_space, SourceSpaces
from .source_estimate import SourceEstimate, VolSourceEstimate, VolVectorSourceEstimate
from .forward import (Forward, make_forward_solution, convert_forward_solution, apply_forward,
                      apply_forward_raw, pick_channels_forward)
import forward


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


def pick_channels(ch_names, include, exclude = (), ordered: bool = True, *, verbose = None):
    """mne.pick_channels: indices of include (minus exclude) in ch_names."""
    import numpy as np
    names = [str(c) for c in ch_names]
    if len(Set[str](names)) != len(names):
        raise RuntimeError("ch_names is not a unique list, picking is unsafe")
    inc = [str(c) for c in include]
    if len(inc) == 0:
        inc = list(names)
    exc = Set[str]([str(c) for c in exclude])
    pos = Dict[str, int]()
    for i in range(len(names)):
        pos[names[i]] = i
    sel = List[int]()
    missing = List[str]()
    for n in inc:
        if n not in pos:
            missing.append(n)
        elif n not in exc:
            sel.append(pos[n])
    if len(missing) > 0 and ordered:
        raise ValueError(f"Missing channels from ch_names required by include:\n{missing}")
    if not ordered:
        sel = sorted(list(Set[int](sel)))
    return np.array(sel, dtype=int)


def pick_channels_regexp(ch_names, regexp: str):
    """mne.pick_channels_regexp (re.match semantics)."""
    import numpy as np
    import re
    r = re.compile(regexp)
    out = List[int]()
    k = 0
    for c in ch_names:
        if r.match(str(c)) is not None:
            out.append(k)
        k += 1
    return np.array(out, dtype=int)


def pick_info(info: Info, sel = None, copy: bool = True, verbose = None) -> Info:
    """mne.pick_info: restrict Info to the channel indices sel."""
    if isinstance(sel, NoneType):
        return info.copy() if copy else info
    idx = [int(i) for i in sel]
    return info._pick(idx)


def channel_type(info: Info, idx: int) -> str:
    return info.ch_types[idx]


def channel_indices_by_type(info: Info, picks = None) -> Dict[str, List[int]]:
    """mne.channel_indices_by_type (types present in info only)."""
    from ._info import _picks_to_idx
    pk = _picks_to_idx(info, picks, "all", ())
    out = Dict[str, List[int]]()
    for i in pk:
        t = info.ch_types[i]
        if t not in out:
            out[t] = List[int]()
        out[t].append(i)
    return out

# beamformer / simulation lane
import beamformer
import simulation
