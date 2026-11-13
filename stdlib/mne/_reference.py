# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne._fiff.reference: set_eeg_reference, set_bipolar_reference,
# add_reference_channels; mne.rank.compute_rank.

import numpy as np
from ._info import Info, create_info, _picks_by_type
from ._containers import RawArray, EpochsArray, Evoked
from .cov import _check_rank, _make_scalings, _compute_rank_data, _compute_rank_cov, Covariance
from ._proj import _has_eeg_average_ref_proj

_COIL_EEG_BIPOLAR: int = 5


def _chan_data(inst):
    """The (channels, times) or (epochs, channels, times) data array of an instance."""
    if isinstance(inst, EpochsArray):
        return inst._data
    elif isinstance(inst, RawArray):
        return inst._data
    else:
        return inst.data


def _set_chan_data(inst, d):
    if isinstance(inst, EpochsArray):
        inst._data = d
    elif isinstance(inst, RawArray):
        inst._data = d
    else:
        inst.data = d


def _with_rows(inst, rows):
    """Append channel rows (n_new, T) / (E, n_new, T) to the instance data."""
    d = _chan_data(inst)
    if isinstance(inst, EpochsArray):
        _set_chan_data(inst, np.concatenate([d, rows], axis=1))
    else:
        _set_chan_data(inst, np.concatenate([d, rows], axis=0))


def set_eeg_reference(inst, ref_channels = "average", copy: bool = True, projection: bool = False,
                      ch_type = "auto", forward = None, *, joint: bool = False, verbose = None):
    """mne.set_eeg_reference: returns (inst, ref_data)."""
    out = inst.copy() if copy else inst
    info = out.info
    eeg = [i for i in range(info.nchan) if info.ch_types[i] == "eeg" and info.ch_names[i] not in info.bads]
    refs = List[int]()
    if not projection:
        if isinstance(ref_channels, str):
            if ref_channels == "average":
                refs = eeg
            elif ref_channels in info.ch_names:
                refs = [info.ch_names.index(ref_channels)]
            else:
                raise ValueError(f"Invalid ref_channels {ref_channels}")
        else:
            for c in ref_channels:
                refs.append(info.ch_names.index(str(c)))
    d = _chan_data(out)
    if len(refs) > 0:
        ra = np.array(refs, dtype=int)
        if isinstance(out, EpochsArray):
            ref_data = d[:, ra, :].mean(axis=1)
        else:
            ref_data = d[ra, :].mean(axis=0)
    else:
        if isinstance(out, EpochsArray):
            ref_data = np.zeros((d.shape[0], d.shape[2]))
        else:
            ref_data = np.zeros(d.shape[1])
    out.set_eeg_reference(ref_channels, projection=projection)
    return out, ref_data


def add_reference_channels(inst, ref_channels, copy: bool = True):
    """mne.add_reference_channels: add all-zero EEG channels and reference to them."""
    names = List[str]()
    if isinstance(ref_channels, str):
        names.append(ref_channels)
    else:
        for c in ref_channels:
            names.append(str(c))
    for c in names:
        if c in inst.info.ch_names:
            raise ValueError(f"Channel {c} already specified in inst.")
    if _has_eeg_average_ref_proj(inst.info):
        for p in inst.info.projs:
            if p.active and p.kind == 10:
                raise RuntimeError("Average reference already applied to data.")
    out = inst.copy() if copy else inst
    d = _chan_data(out)
    if isinstance(out, EpochsArray):
        _with_rows(out, np.zeros((d.shape[0], len(names), d.shape[2])))
    else:
        _with_rows(out, np.zeros((len(names), d.shape[1])))
    ref_info = create_info(names, out.info.sfreq, "eeg")
    for i in range(len(names)):
        out.info._append_ch(ref_info, i)
    out.set_eeg_reference(names, projection=False)
    return out


def set_bipolar_reference(inst, anode, cathode, ch_name = None, ch_info = None, drop_refs: bool = True,
                          copy: bool = True, on_bad: str = "warn", verbose = None):
    """mne.set_bipolar_reference: virtual bipolar channels anode - cathode."""
    an = [anode] if isinstance(anode, str) else [str(a) for a in anode]
    ca = [cathode] if isinstance(cathode, str) else [str(a) for a in cathode]
    if len(an) != len(ca):
        raise ValueError(f"Number of anodes (got {len(an)}) must equal the number of cathodes (got {len(ca)}).")
    names = List[str]()
    if isinstance(ch_name, NoneType):
        for k in range(len(an)):
            names.append(f"{an[k]}-{ca[k]}")
    elif isinstance(ch_name, str):
        names.append(ch_name)
    else:
        for c in ch_name:
            names.append(str(c))
    if len(names) != len(an):
        raise ValueError(f"Number of channel names must equal the number of anodes/cathodes (got {len(names)}).")
    for k in range(len(an)):
        if names[k] != an[k] and names[k] != ca[k] and names[k] in inst.info.ch_names:
            raise ValueError(f'There is already a channel named "{names[k]}", please specify a different name for the bipolar channel using the ch_name parameter.')
    out = inst.copy() if copy else inst
    info = out.info
    mult = np.zeros((len(an), info.nchan))
    for k in range(len(an)):
        mult[k, info.ch_names.index(an[k])] = 1.0
        mult[k, info.ch_names.index(ca[k])] = -1.0
    d = _chan_data(out)
    if isinstance(out, EpochsArray):
        rows = np.zeros((d.shape[0], len(an), d.shape[2]))
        for e in range(d.shape[0]):
            rows[e] = np.dot(mult, d[e])
    else:
        rows = np.dot(mult, d)
    # new channel info copied from the anode channels
    for k in range(len(an)):
        ai = info.ch_names.index(an[k])
        info.ch_names.append(names[k])
        info.ch_types.append(info.ch_types[ai])
        info.cals.append(info.cals[ai])
        info.ranges.append(info.ranges[ai])
        info.locs.append(list(info.locs[ai]))
        info.coil_types.append(_COIL_EEG_BIPOLAR)
        info.units.append(info.units[ai])
    _with_rows(out, rows)
    bad = List[str]()
    for k in range(len(an)):
        if an[k] in info.bads or ca[k] in info.bads:
            bad.append(names[k])
    for b in bad:
        info.bads.append(b)
    if drop_refs:
        drop = List[str]()
        for a in an:
            if a not in drop:
                drop.append(a)
        for c in ca:
            if c not in drop:
                drop.append(c)
        drop = [c for c in drop if c in info.ch_names]
        out.drop_channels(drop)
    return out


def compute_rank(inst, rank = None, scalings = None, info: Optional[Info] = None, tol = "auto", *,
                 proj: bool = True, tol_kind: str = "absolute", on_rank_mismatch: str = "ignore",
                 verbose = None) -> Dict[str, int]:
    """mne.compute_rank for Raw / Epochs / Evoked (data) and Covariance instances."""
    rk = _check_rank(rank)
    if isinstance(inst, Covariance):
        if info is None:
            raise ValueError("info cannot be None when inst is a Covariance")
        inf = info.__val__()
        sc = _make_scalings(scalings, "cov_rank")
        return _compute_rank_cov(inst._C, inf, rk, sc, inst.diag)
    else:
        inf2 = inst.info
        if info is not None:
            inf2 = info.__val__()
        sc2 = _make_scalings(scalings, "cov_rank")
        d = _chan_data(inst)
        if isinstance(inst, EpochsArray):
            n = d.shape[1]
            m = d.shape[0] * d.shape[2]
            flat = np.zeros((n, m))
            for e in range(d.shape[0]):
                flat[:, e * d.shape[2]:(e + 1) * d.shape[2]] = d[e]
            return _compute_rank_data(flat, inf2, rk, sc2)
        return _compute_rank_data(d, inf2, rk, sc2)
