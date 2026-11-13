# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.simulation: simulate_sparse_stc, simulate_evoked, add_noise, simulate_raw
# (static head, forward given). RNG streams follow mne: a numpy Generator
# (default_rng) unless a legacy `random_state` is given (RandomState).
#
# Not ported: label based simulation (simulate_stc, SourceSimulator,
# labels= of simulate_sparse_stc), head_pos movement, add_eog/ecg/chpi.

import numpy as np
from .._info import Info, _picks_to_idx
from .._containers import Evoked, EpochsArray, RawArray
from ..cov import Covariance, compute_whitener
from ..forward import Forward, apply_forward, pick_channels_forward, convert_forward_solution
from ..source_estimate import _STC
from ..source_space import SourceSpaces
from ..epochs import EvokedArray


def _get_rng(rng, random_state):
    """mne._legacy_rng: random_state -> RandomState, rng -> Generator."""
    if isinstance(random_state, NoneType):
        if isinstance(rng, NoneType):
            return np.random.default_rng()
        elif isinstance(rng, int):
            return np.random.default_rng(rng)
        else:
            return rng
    elif isinstance(random_state, int):
        return np.random.RandomState(random_state)
    else:
        return random_state


def simulate_sparse_stc(src: SourceSpaces, n_dipoles: int, times, data_fun = None, labels = None,
                        location: str = "random", subject = None, subjects_dir = None,
                        surf: str = "sphere", *, rng = None, random_state = None):
    """mne.simulation.simulate_sparse_stc (volume / discrete source spaces).

    data_fun maps the times array to a 1-D waveform (default
    1e-7 * sin(20 * pi * t))."""
    if not isinstance(labels, NoneType):
        compile_error("label based simulation is not supported")
    r = _get_rng(rng, random_state)
    tt = np.asarray(times, dtype=float)
    data = np.zeros((n_dipoles, tt.shape[0]))
    for i in range(n_dipoles):
        if isinstance(data_fun, NoneType):
            data[i, :] = 1e-7 * np.sin(20 * np.pi * tt)
        else:
            data[i, :] = np.asarray(data_fun(tt), dtype=float)
    ns = len(src)
    offsets = np.linspace(0, n_dipoles, ns + 1).astype(int)
    vs = List[np.ndarray[int, 1]]()
    for si in range(ns):
        n = int(offsets[si + 1] - offsets[si])
        s = src.spaces[si]
        perm = r.permutation(np.arange(s.nuse))
        sel = np.sort(perm[:n])
        vs.append(s.vertno[sel].copy())
    tstep = float(tt[1] - tt[0])
    kind = "volume" if src.kind == "discrete" else src.kind
    return _STC[np.ndarray[float, 2]](data, vs, float(tt[0]), tstep, subject, kind)


def _generate_noise(info: Info, cov: Covariance, iir_filter, r, n_samples: int, picks):
    """mne.simulation.evoked._generate_noise: colored noise (n_ch, n_samples)."""
    _, _, colorer = compute_whitener(cov, info, picks, pca=True, return_colorer=True)
    z = r.standard_normal((colorer.shape[1], n_samples))
    noise = np.dot(colorer, z)
    if not isinstance(iir_filter, NoneType):
        a = [float(v) for v in iir_filter]
        zi = np.zeros((colorer.shape[0], len(a) - 1))
        y, zf = lfilter_(a, noise, zi)
        return y
    return noise


def lfilter_(a: List[float], x, zi):
    from scipy.signal import lfilter
    return lfilter([1.0], a, x, -1, zi)


def _noise_picks(info: Info, cov: Covariance) -> List[int]:
    cs = Set[str](cov.names)
    return [i for i in range(info.nchan) if info.ch_names[i] in cs]


def _add_noise_2d(data, info: Info, cov: Covariance, iir_filter, r, subselect: bool):
    """Add noise to a (n_ch, n_times) array in place (mne._add_noise body)."""
    if subselect:
        picks = _noise_picks(info, cov)
        pinfo = info._pick(picks)
        gen = list(range(pinfo.nchan))
        noise = _generate_noise(pinfo, cov, iir_filter, r, data.shape[1], gen)
        for k in range(len(picks)):
            data[picks[k], :] += noise[k]
    else:
        allp = list(range(info.nchan))
        noise = _generate_noise(info, cov, iir_filter, r, data.shape[1], allp)
        data += noise


def add_noise(inst, cov: Covariance, iir_filter = None, *, verbose = None, rng = None,
              random_state = None):
    """mne.simulation.add_noise for Raw / Epochs / Evoked (in place, returns inst)."""
    if isinstance(random_state, int):
        # legacy integer seed: a fresh RandomState per epoch
        if isinstance(inst, EpochsArray):
            for e in range(inst._data.shape[0]):
                r = np.random.RandomState(random_state)
                d = inst._data[e].copy()
                _add_noise_2d(d, inst.info, cov, iir_filter, r, True)
                inst._data[e] = d
            return inst
    r = _get_rng(rng, random_state)
    if isinstance(inst, RawArray):
        _add_noise_2d(inst._data, inst.info, cov, iir_filter, r, True)
    elif isinstance(inst, Evoked):
        _add_noise_2d(inst.data, inst.info, cov, iir_filter, r, True)
    elif isinstance(inst, EpochsArray):
        for e in range(inst._data.shape[0]):
            d = inst._data[e].copy()
            _add_noise_2d(d, inst.info, cov, iir_filter, r, True)
            inst._data[e] = d
    else:
        compile_error("inst must be Raw, Epochs or Evoked")
    return inst


def simulate_evoked(fwd: Forward, stc, info: Info, cov = None, nave: float = 30.0,
                    iir_filter = None, use_cps: bool = True, verbose = None, *, rng = None,
                    random_state = None):
    """mne.simulation.simulate_evoked."""
    evoked = apply_forward(fwd, stc, info, use_cps=use_cps)
    if isinstance(cov, NoneType):
        return evoked
    r = _get_rng(rng, random_state)
    if nave < np.inf:
        noise = EvokedArray(np.zeros(evoked.data.shape), evoked.info, float(evoked.times[0]))
        _add_noise_2d(noise.data, noise.info, cov, iir_filter, r, False)
        evoked.data = evoked.data + noise.data / np.sqrt(nave)
        evoked.nave = int(nave)
    if len(cov.projs) > 0:
        evoked.add_proj(cov.projs)
        evoked.apply_proj()
    return evoked


def simulate_raw(info: Info, stc = None, trans = None, src = None, bem = None, head_pos = None,
                 mindist: float = 1.0, interp: str = "cos2", n_jobs = None, use_cps: bool = True,
                 forward = None, first_samp: int = 0, max_iter: int = 10000, verbose = None):
    """mne.simulation.simulate_raw for a single STC, a fixed head position and a
    precomputed discrete-source forward solution. The first sample at
    time 0 of the STC carries event code 1 on the first stim channel."""
    if isinstance(forward, NoneType):
        compile_error("simulate_raw requires forward=")
    if not isinstance(head_pos, NoneType):
        compile_error("head_pos is not supported")
    fw = convert_forward_solution(forward, surf_ori=True, force_fixed=True, copy=True)
    from ..forward import _stc_src_sel
    src_sel, stc_sel = _stc_src_sel(fw.src, stc.vertices)
    sa = np.array(stc_sel, dtype=int)
    stc_data = stc.data[sa]
    gain = fw.sol[:, np.array(src_sel, dtype=int)]
    nt = stc_data.shape[1]
    if nt <= 2:
        raise ValueError(f"stc must have at least three time points, got {nt}")
    if abs(info.sfreq - 1.0 / stc.tstep) > 1e-8 + 1e-5 * abs(1.0 / stc.tstep):
        raise ValueError(f"stc and info must have same sample rate, got {1.0 / stc.tstep} and {info.sfreq}")
    out = np.zeros((info.nchan, nt))
    meeg = np.dot(gain, stc_data)
    for k in range(len(fw.ch_names)):
        out[info.ch_names.index(fw.ch_names[k])] = meeg[k]
    for i in range(info.nchan):
        if info.ch_types[i] == "stim":
            amin = 0
            best = np.inf
            tms = stc.times
            for j in range(nt):
                if abs(tms[j]) < best:
                    best = abs(tms[j])
                    amin = j
            out[i, amin] = 1.0
            break
    return RawArray(out, info, first_samp=first_samp)


from ._labels import Label, simulate_stc, SourceSimulator, add_eog, add_ecg
