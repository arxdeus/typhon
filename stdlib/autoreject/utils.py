# Copyright (C) 2026 the Typhon authors
# Ported from autoreject 0.5.2 (BSD-3-Clause)

"""Utility functions for autoreject.

Typhon notes: interpolation follows the MNE-Python < 1.14 code path of
autoreject (spherical splines for EEG, ``dots`` is always None). MEG field
interpolation is not available and raises NotImplementedError. Progress
bars (``_pbar``) return the iterable unchanged, and
``set_matplotlib_defaults`` is a no-op.
"""

import numpy as np
import warnings
from mne import pick_info, channel_type
from mne._info import Info, _picks_to_idx
from mne.channels import _make_interpolation_matrix, _fit_sphere

_SUPPORTED = ("mag", "grad", "eeg", "hbo", "hbr", "ecog", "seeg")


def _check_ch_locs(info: Info) -> bool:
    """True if valid 3D channel locations exist for the channels of info."""
    n = info.nchan
    if n == 0:
        return False
    all_zero = True
    for i in range(n):
        for k in range(3):
            v = info.locs[i][k]
            if not np.isfinite(v):
                return False
            if v != 0.0:
                all_zero = False
    if all_zero:
        return False
    close0 = True
    for i in range(n):
        for k in range(3):
            if abs(info.locs[i][k]) > 1e-8:
                close0 = False
    return not close0


def _ch_types_of(info: Info, picks) -> Set[str]:
    out = Set[str]()
    for p in picks:
        out.add(info.ch_types[int(p)])
    return out


def _check_data(epochs, picks, ch_constraint: str = 'data_channels',
                check_loc: bool = True, verbose = True):
    if not epochs.preload:
        raise ValueError('Data must be preloaded.')
    n_bads = len(epochs.info.bads)
    pk = [int(p) for p in picks]
    picked_info = pick_info(epochs.info, pk)
    types = _ch_types_of(epochs.info, pk)
    if check_loc and not _check_ch_locs(picked_info):
        raise RuntimeError('Valid channel positions are needed '
                           'for autoreject to work')
    message = 'AutoReject only supports ' + ', '.join(list(_SUPPORTED)) + ' currently.'
    if ch_constraint == 'data_channels':
        for t in types:
            if t not in _SUPPORTED:
                raise ValueError(message)
    elif ch_constraint == 'single_channel_type':
        c = 0
        for t in _SUPPORTED:
            if t in types:
                c += 1
        if c > 1:
            raise ValueError(message)
    else:
        raise ValueError('bad value for ch_constraint.')
    if n_bads > 0 and _verbose_bool(verbose):
        warnings.warn(f'{n_bads} channels are marked as bad. These will be ignored. '
                      'If you want them to be considered by autoreject please '
                      'remove them from epochs.info["bads"].')


def _verbose_bool(verbose) -> bool:
    if isinstance(verbose, NoneType):
        return False
    elif isinstance(verbose, str):
        return len(verbose) > 0
    else:
        return bool(verbose)


def _is_data_type(t: str) -> bool:
    """mne.pick_types(meg=True, eeg=True, ref_meg=False, fnirs=True,
    ecog=True, seeg=True)."""
    return (t == "mag" or t == "grad" or t == "eeg" or t == "ecog" or t == "seeg"
            or t == "hbo" or t == "hbr" or t.startswith("fnirs"))


def _handle_picks(info: Info, picks) -> np.ndarray[int, 1]:
    """Pick the data channels or return picks."""
    out = List[int]()
    if isinstance(picks, NoneType):
        for i in range(info.nchan):
            if _is_data_type(info.ch_types[i]) and info.ch_names[i] not in info.bads:
                out.append(i)
    else:
        out = _picks_to_idx(info, picks, "all", "bads")
    return np.array(out, dtype=int)


def _get_picks_by_type(info: Info, picks) -> List[Tuple[str, List[int]]]:
    """Get the picks grouped by channel type (in order of first appearance)."""
    keys = List[str]()
    sub = Dict[str, List[int]]()
    for pp in picks:
        p = int(pp)
        key = channel_type(info, p)
        if key not in sub:
            sub[key] = List[int]()
            keys.append(key)
        sub[key].append(p)
    return [(k, sub[k]) for k in keys]


def set_matplotlib_defaults(plt = None, style: str = 'ggplot'):
    """No-op in Typhon (no matplotlib)."""
    pass


def _pprint(params: List[Tuple[str, str]], offset: int = 0) -> str:
    """Pretty print 'name=repr' pairs (sorted by name), sklearn style."""
    items = sorted(params, key=lambda kv: kv[0])
    parts = List[str]()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i in range(len(items)):
        this_repr = items[i][0] + '=' + items[i][1]
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                parts.append(line_sep)
                this_line_length = len(line_sep)
            else:
                parts.append(', ')
                this_line_length += 2
        parts.append(this_repr)
        this_line_length += len(this_repr)
    lines = ''.join(parts)
    return '\n'.join([li.rstrip(' ') for li in lines.split('\n')])


def _pbar(iterable, desc: str = '', verbose = True):
    """Progress bars are not rendered in Typhon: returns the iterable."""
    if isinstance(verbose, str):
        if verbose not in ("tqdm", "tqdm_notebook", "progressbar"):
            raise ValueError(f"verbose must be a boolean value. Got {repr(verbose)}")
    return iterable


def _check_eeg_only(info: Info, picks):
    for p in picks:
        t = info.ch_types[int(p)]
        if t == "mag" or t == "grad":
            raise NotImplementedError("MEG interpolation is not available in Typhon autoreject")


def _interp_rows(info: Info, picks: List[int], bads_local: List[int]) -> np.ndarray[float, 2]:
    """Rows (n_bads, n_picks) interpolating the bads (local indices into
    picks) from the other EEG picks, as autoreject's _interpolate_bads_eeg on
    an identity Evoked. Rows of non-EEG bads leave the data as is."""
    n = len(picks)
    out = np.zeros((len(bads_local), n))
    for k in range(len(bads_local)):
        out[k, bads_local[k]] = 1.0
    eeg_local = [i for i in range(n) if info.ch_types[picks[i]] == "eeg"]
    bad_set = Set[int](bads_local)
    eeg_bads = [i for i in eeg_local if i in bad_set]
    eeg_goods = [i for i in eeg_local if i not in bad_set]
    if len(eeg_local) == 0 or len(eeg_bads) == 0:
        return out
    pos_good = np.array([[info.locs[picks[i]][0], info.locs[picks[i]][1], info.locs[picks[i]][2]] for i in eeg_goods])
    pos_bad = np.array([[info.locs[picks[i]][0], info.locs[picks[i]][1], info.locs[picks[i]][2]] for i in eeg_bads])
    radius, center = _fit_sphere(pos_good)
    dist = 0.0
    for i in range(pos_good.shape[0]):
        d = pos_good[i] - center
        dist += np.sqrt(np.sum(d * d)) / radius
    dist /= pos_good.shape[0]
    if abs(1. - dist) > 0.1:
        warnings.warn('Your spherical fit is poor, interpolation results are '
                      'likely to be inaccurate.')
    interp = _make_interpolation_matrix(pos_good, pos_bad)
    row_of = Dict[int, int]()
    for k in range(len(bads_local)):
        row_of[bads_local[k]] = k
    for b in range(len(eeg_bads)):
        r = row_of[eeg_bads[b]]
        out[r, bads_local[r]] = 0.0
        for g in range(len(eeg_goods)):
            out[r, eeg_goods[g]] = interp[b, g]
    return out


def _interp_matrices(info: Info, picks, dots, bad_sets: List[List[str]], mode: str = 'fast',
                     desc: str = '', verbose = True) -> List[np.ndarray[float, 2]]:
    """For each set of bad channel names, the (n_bads, n_picks) matrix that
    interpolates them from the picks."""
    pk = [int(p) for p in picks]
    _check_eeg_only(info, pk)
    idx = Dict[str, int]()
    for i in range(len(pk)):
        idx[info.ch_names[pk[i]]] = i
    out = List[np.ndarray[float, 2]]()
    for bads in bad_sets:
        out.append(_interp_rows(info, pk, [idx[ch] for ch in bads]))
    return out


def _has_interp_data() -> bool:
    return False


def _precompute_interp(info: Info, picks, mode: str = 'fast', n_jobs: int = 1):
    """No precomputation is needed for EEG (MEG unsupported)."""
    _check_eeg_only(info, picks)
    return None


def _get_interp_data(info: Info, picks, dots = None, mode: str = 'fast', n_jobs: int = 1):
    return dots


def _restore_interp_data(dots):
    return dots


def _inst_data(inst):
    if hasattr(inst, "_data"):
        return inst._data
    else:
        return inst.data


def _clean_by_interp(inst, picks = None, dots = None, verbose = True):
    inst_interp = inst.copy()
    pk = _handle_picks(inst_interp.info, picks)
    pl = [int(p) for p in pk]
    ch_names = [inst.ch_names[p] for p in pl]
    mats = _interp_matrices(inst.info, pl, dots, [[ch] for ch in ch_names], 'fast',
                            'Creating augmented epochs', verbose)
    loo = np.concatenate(mats)
    src = _inst_data(inst)
    dst = _inst_data(inst_interp)
    if src.ndim == 3:
        for e in range(src.shape[0]):
            res = loo @ src[e][pk, :]
            for i in range(len(pl)):
                dst[e, pl[i], :] = res[i]
    else:
        res = loo @ src[pk, :]
        for i in range(len(pl)):
            dst[pl[i], :] = res[i]
    return inst_interp


def clean_by_interp(inst, picks = None, verbose = True):
    """Clean epochs/evoked by LOOCV: each channel replaced by its
    interpolation from all the other picked channels."""
    return _clean_by_interp(inst, picks=picks, verbose=verbose)


def interpolate_bads(inst, picks, dots = None, reset_bads: bool = True, mode: str = 'accurate'):
    """Interpolate bad EEG channels of inst (in place) from the good picks."""
    pl = [int(p) for p in picks]
    _check_eeg_only(inst.info, pl)
    eeg = [p for p in pl if inst.info.ch_types[p] == "eeg"]
    bads_local = [i for i in range(len(eeg)) if inst.info.ch_names[eeg[i]] in inst.info.bads]
    if len(eeg) > 0 and len(bads_local) > 0:
        rows = _interp_rows(inst.info, eeg, bads_local)
        ea = np.array(eeg, dtype=int)
        d = _inst_data(inst)
        if d.ndim == 3:
            for e in range(d.shape[0]):
                res = rows @ d[e][ea, :]
                for k in range(len(bads_local)):
                    d[e, eeg[bads_local[k]], :] = res[k]
        else:
            res = rows @ d[ea, :]
            for k in range(len(bads_local)):
                d[eeg[bads_local[k]], :] = res[k]
    if reset_bads:
        inst.info.bads = List[str]()
    return inst


def _get_channel_type(epochs, picks) -> str:
    """Return whether a set of picks are all meg or all eeg channels."""
    types = _ch_types_of(epochs.info, picks)
    has_meg = "mag" in types or "grad" in types
    if "eeg" in types and has_meg:
        raise ValueError('Got mixed channel types. Pick either eeg or meg'
                         ' but not both')
    if "eeg" in types:
        return 'eeg'
    elif has_meg:
        return 'meg'
    else:
        raise ValueError('Oh no! Your channel type is not known.')
