# Ported from mne 1.13.2 (BSD-3-Clause)
#
# maxwell_filter (SSS and tSSS), compute_maxwell_basis,
# find_bad_channels_maxwell and maxwell_filter_prepare_emptyroom.
#
# Typhon Info carries no dev_head_t, cross-talk or fine-calibration data:
# the device and head frames coincide, head_pos / destination / cross_talk /
# calibration are not supported (they raise).

import math
import numpy as np
from scipy.signal import get_window
from .._info import Info
from ..annotations import Annotations
from ._annotate import _mask_to_onsets_offsets
from ._maxwell import (_get_n_moments, _MFCoils, _prep_mf_coils, _basis, _col_norm_pinv,
                       _regularize_in, _regularize_out)
from ..channels import _auto_origin


def _coil_base(ct: int) -> float:
    if ct == 3012 or ct == 3013 or ct == 3014:
        return 0.0168
    return 0.0


class _Decomp:
    S_decomp: np.ndarray[float, 2]
    S_full: np.ndarray[float, 2]
    pS: np.ndarray[float, 2]
    reg_moments: List[int]
    n_use_in: int

    def __init__(self, S_decomp, S_full, pS, reg_moments: List[int], n_use_in: int):
        self.S_decomp = S_decomp
        self.S_full = S_full
        self.pS = pS
        self.reg_moments = reg_moments
        self.n_use_in = n_use_in


class _MFPrep:
    meg_picks: List[int]
    good_mask: List[bool]
    mag_mask: List[bool]
    coil_scale: np.ndarray[float, 1]
    mag_scale: float
    origin: np.ndarray[float, 1]
    coils: _MFCoils
    int_order: int
    ext_order: int
    regularize: str
    bad_condition: str
    S_recon: np.ndarray[float, 2]

    def __init__(self, meg_picks: List[int], good_mask: List[bool], mag_mask: List[bool],
                 coil_scale: np.ndarray[float, 1], mag_scale: float, origin: np.ndarray[float, 1],
                 coils: _MFCoils, int_order: int, ext_order: int, regularize: str,
                 bad_condition: str, S_recon: np.ndarray[float, 2]):
        self.meg_picks = meg_picks
        self.good_mask = good_mask
        self.mag_mask = mag_mask
        self.coil_scale = coil_scale
        self.mag_scale = mag_scale
        self.origin = origin
        self.coils = coils
        self.int_order = int_order
        self.ext_order = ext_order
        self.regularize = regularize
        self.bad_condition = bad_condition
        self.S_recon = S_recon

    def n_good(self) -> int:
        n = 0
        for g in self.good_mask:
            if g:
                n += 1
        return n


def _prep_maxwell(info: Info, origin, int_order: int, ext_order: int, regularize, bad_condition: str,
                  mag_scale, calibration, cross_talk, head_pos, destination, coord_frame: str) -> _MFPrep:
    if not isinstance(calibration, NoneType) or not isinstance(cross_talk, NoneType):
        raise NotImplementedError("calibration and cross_talk are not supported")
    if not isinstance(head_pos, NoneType) or not isinstance(destination, NoneType):
        raise NotImplementedError("head_pos and destination are not supported")
    if coord_frame != "head" and coord_frame != "meg":
        raise ValueError(f"coord_frame must be one of ['head', 'meg'], got {coord_frame}")
    reg = ""
    if isinstance(regularize, str):
        if regularize != "in" and regularize != "in_argmax":
            raise ValueError(f"regularize must be one of ['in', 'in_argmax'] when str, got {regularize}")
        reg = regularize
    if bad_condition not in ("error", "warning", "ignore", "info"):
        raise ValueError(f"bad_condition must be one of ['error', 'warning', 'ignore', 'info'], got {bad_condition}")
    meg_picks = List[int]()
    mag_mask = List[bool]()
    good_mask = List[bool]()
    for i in range(info.nchan):
        t = info.ch_types[i]
        if t == "mag" or t == "grad":
            meg_picks.append(i)
            mag_mask.append(t == "mag")
            good_mask.append(info.ch_names[i] not in info.bads)
    ng = 0
    for g in good_mask:
        if g:
            ng += 1
    n_bases = _get_n_moments(int_order) + _get_n_moments(ext_order)
    if n_bases > ng:
        raise ValueError(f"Number of requested bases ({n_bases}) exceeds number of good sensors ({ng})")
    n_mag = 0
    for m in mag_mask:
        if m:
            n_mag += 1
    ms = 100.0
    if isinstance(mag_scale, str):
        if mag_scale != "auto":
            raise ValueError(f'mag_scale must be a float or "auto", got "{mag_scale}"')
        if n_mag == 0 or n_mag == len(meg_picks):
            ms = 100.0
        else:
            base = -1.0
            ok = True
            for k in range(len(meg_picks)):
                if not mag_mask[k]:
                    b = _coil_base(info.coil_types[meg_picks[k]])
                    if base < 0.0:
                        base = b
                    elif b != base:
                        ok = False
            if (not ok) or base <= 0.0:
                raise RuntimeError("Could not automatically determine mag_scale, could not find one proper gradiometer distance")
            ms = 1.0 / base
    else:
        ms = float(mag_scale)
    cs = np.ones(len(meg_picks))
    for k in range(len(meg_picks)):
        if mag_mask[k]:
            cs[k] = ms
    org = np.zeros(3)
    if isinstance(origin, str):
        if origin != "auto":
            raise ValueError(f'origin must be a numerical array, or "auto", not {origin}')
        if coord_frame == "head":
            org = _auto_origin(info)
    else:
        o = np.array(origin, dtype=float)
        if o.shape[0] != 3:
            raise ValueError("origin must be a 3-element array")
        org = o
    info = info.copy()
    for k in range(len(meg_picks)):
        if mag_mask[k] and (info.coil_types[meg_picks[k]] == 3022 or info.coil_types[meg_picks[k]] == 3023):
            info.coil_types[meg_picks[k]] = 3024
    coils = _prep_mf_coils(info, meg_picks)
    S_recon = _basis(coils, org, int_order, 0)
    return _MFPrep(meg_picks, good_mask, mag_mask, cs, ms, org, coils, int_order, ext_order, reg,
                   bad_condition, S_recon)


def _take_rows(A: np.ndarray[float, 2], idx: List[int]) -> np.ndarray[float, 2]:
    out = np.empty((len(idx), A.shape[1]))
    for a in range(len(idx)):
        out[a, :] = A[idx[a], :]
    return out


def _take_cols(A: np.ndarray[float, 2], idx: List[int]) -> np.ndarray[float, 2]:
    out = np.empty((A.shape[0], len(idx)))
    for a in range(len(idx)):
        out[:, a] = A[:, idx[a]]
    return out


def _decomp(p: _MFPrep, good_mask: List[bool]) -> _Decomp:
    n_in = _get_n_moments(p.int_order)
    S = _basis(p.coils, p.origin, p.int_order, p.ext_order)
    nm = S.shape[0]
    for i in range(nm):
        S[i, :] *= p.coil_scale[i]
    good = [i for i in range(nm) if good_mask[i]]
    S_decomp = _take_rows(S, good)
    any_mag = False
    for m in p.mag_mask:
        if m:
            any_mag = True
    in_removes = List[int]()
    out_removes = List[int]()
    if p.regularize != "":
        in_removes, out_removes = _regularize_in(p.int_order, p.ext_order, S_decomp, any_mag,
                                                 p.regularize == "in_argmax")
    else:
        out_removes = _regularize_out(p.int_order, p.ext_order, any_mag)
    reg_moments = List[int]()
    n_use_in = 0
    for k in range(n_in):
        if k not in in_removes:
            reg_moments.append(k)
            n_use_in += 1
    for k in range(n_in, S.shape[1]):
        if k not in out_removes:
            reg_moments.append(k)
    S_decomp = _take_cols(S_decomp, reg_moments)
    S_full = _take_cols(S, reg_moments)
    pS, sing = _col_norm_pinv(S_decomp.copy())
    cond = float(sing[0]) / float(sing[sing.shape[0] - 1])
    if p.bad_condition == "error" and cond >= 1000.0:
        raise RuntimeError(f"Matrix is badly conditioned: {int(round(cond))} >= 1000")
    for j in range(len(good)):
        pS[:, j] *= p.coil_scale[good[j]]
    for j in range(len(good)):
        S_decomp[j, :] /= p.coil_scale[good[j]]
    for i in range(nm):
        S_full[i, :] /= p.coil_scale[i]
    return _Decomp(S_decomp, S_full, pS, reg_moments, n_use_in)


def _ops(p: _MFPrep, d: _Decomp, reconstruct: str):
    """op_sss, op_in, op_resid (mne _MoveComp.get_decomp_by_offset, no movement)."""
    nu = d.n_use_in
    op_sss = np.zeros((1, 1))
    if reconstruct == "orig":
        op_sss = np.dot(d.S_full, d.pS)
    else:
        cols = [d.reg_moments[k] for k in range(nu)]
        op_sss = np.dot(_take_cols(p.S_recon, cols), d.pS[:nu, :])
    S_in = d.S_decomp[:, :nu]
    op_in = np.dot(S_in, d.pS[:nu, :])
    ng = d.S_decomp.shape[0]
    op_resid = np.eye(ng) - op_in
    op_resid = op_resid - np.dot(d.S_decomp[:, nu:], d.pS[nu:, :])
    return op_sss, op_in, op_resid


def _orth_overwrite(A: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    M = A.shape[0]
    N = A.shape[1]
    eps = 2.220446049250313e-16
    tol = max(M, N) * float(np.max(s)) * eps
    num = 0
    for v in s:
        if v > tol:
            num += 1
    return u[:, :num].copy()


def _overlap_projector(data_int: np.ndarray[float, 2], data_res: np.ndarray[float, 2],
                       corr: float) -> np.ndarray[float, 2]:
    n = float(np.sqrt((data_int * data_int).sum()))
    if n == 0.0:
        n = 1.0
    di = _orth_overwrite((data_int / n).T.copy())
    n = float(np.sqrt((data_res * data_res).sum()))
    if n == 0.0:
        n = 1.0
    dr = _orth_overwrite((data_res / n).T.copy())
    if di.shape[1] == 0 or dr.shape[1] == 0:
        return np.zeros((data_int.shape[1], 0))
    qi, ri = np.linalg.qr(di)
    qr_, rr = np.linalg.qr(dr)
    C = np.dot(qi.T, qr_)
    u, S, Vh = np.linalg.svd(C, full_matrices=False)
    keep = [k for k in range(S.shape[0]) if S[k] >= corr]
    Vk = np.empty((len(keep), Vh.shape[1]))
    for a in range(len(keep)):
        Vk[a, :] = Vh[keep[a], :]
    return np.dot(qr_, Vk.T)


def _tsss_apply(clean: np.ndarray[float, 2], op_in: np.ndarray[float, 2], op_resid: np.ndarray[float, 2],
                orig: np.ndarray[float, 2], corr: float, valid: bool) -> np.ndarray[float, 2]:
    t_proj = np.zeros((clean.shape[1], 0))
    if valid:
        orig_in = np.dot(op_in, orig)
        resid = np.dot(op_resid, orig)
        t_proj = _overlap_projector(orig_in, resid, corr)
    return clean - np.dot(np.dot(clean, t_proj), t_proj.T)


def _cola_window(name: str, n: int, step: int) -> np.ndarray[float, 1]:
    w = get_window(name, n, fftbins = ((n - 1) % 2) != 0)
    nb = step
    binsums = np.zeros(nb)
    for ii in range(n // step):
        for k in range(step):
            binsums[k] += w[ii * step + k]
    if n % step != 0:
        r = n % step
        for k in range(r):
            binsums[k] += w[n - r + k]
    srt = np.sort(binsums)
    const = 0.0
    if nb % 2 == 1:
        const = float(srt[nb // 2])
    else:
        const = 0.5 * (float(srt[nb // 2 - 1]) + float(srt[nb // 2]))
    dev = float(np.max(np.abs(binsums - const)))
    if dev > 1e-10:
        raise ValueError(f"segment length {n} with step {step} for {name} window type does not provide a constant output")
    return w / const


def _segments(raw, skip_by_annotation) -> Tuple[List[int], List[int]]:
    kinds = List[str]()
    if isinstance(skip_by_annotation, str):
        kinds.append(skip_by_annotation)
    else:
        for k in skip_by_annotation:
            kinds.append(str(k))
    n = raw._data.shape[1]
    ft = raw.first_time
    sf = raw.info.sfreq
    ann = raw.annotations
    mask = [True for _ in range(n)]
    extra_on = List[int]()
    for a in range(len(ann)):
        hit = False
        for kd in kinds:
            if ann.description[a].upper().startswith(kd.upper()):
                hit = True
        if not hit:
            continue
        s = int(round((ann.onset[a] - ft) * sf))
        e = int(round((ann.onset[a] - ft + ann.duration[a]) * sf))
        for k in range(max(s, 0), min(e, n)):
            mask[k] = False
        if s == e:
            extra_on.append(s)
    on, off = _mask_to_onsets_offsets(mask)
    for x in extra_on:
        on.append(x)
        off.append(x)
    on = sorted(on)
    off = sorted(off)
    return on, off


def _reset_meg_bads(info: Info):
    nb = List[str]()
    for b in info.bads:
        i = info.ch_names.index(b)
        if not (info.ch_types[i] == "mag" or info.ch_types[i] == "grad"):
            nb.append(b)
    info.bads = nb


def maxwell_filter(raw, origin = "auto", int_order: int = 8, ext_order: int = 3, calibration = None,
                   cross_talk = None, st_duration = None, st_correlation: float = 0.98,
                   coord_frame: str = "head", destination = None, regularize = "in",
                   ignore_ref: bool = False, bad_condition: str = "error", head_pos = None,
                   st_fixed: bool = True, st_only: bool = False, mag_scale = 100.0,
                   skip_by_annotation = ("edge", "bad_acq_skip"), extended_proj = (),
                   st_overlap: bool = True, mc_interp: str = "hann", verbose = None):
    """Maxwell filter (SSS / tSSS) raw data (mne 1.13.2)."""
    if st_correlation <= 0.0 or st_correlation > 1.0:
        raise ValueError(f"Need 0 < st_correlation <= 1., got {st_correlation}")
    has_st = not isinstance(st_duration, NoneType)
    if st_only and not has_st:
        raise ValueError("st_duration must not be None if st_only is True")
    if len(extended_proj) > 0:
        raise NotImplementedError("extended_proj is not supported")
    p = _prep_maxwell(raw.info, origin, int_order, ext_order, regularize, bad_condition, mag_scale,
                      calibration, cross_talk, head_pos, destination, coord_frame)
    sf = raw.info.sfreq
    n_all = raw._data.shape[1]
    st_dur = 0
    st_dur = min(max(int(round(10.0 * sf)), 1), n_all)
    if isinstance(st_duration, NoneType):
        pass
    else:
        st_dur = int(round(float(st_duration) * sf))
    out = raw.copy()
    onsets, ends = _segments(raw, skip_by_annotation)
    max_samps = 0
    for k in range(len(onsets)):
        max_samps = max(max_samps, ends[k] - onsets[k])
    if not (0.0 < float(st_dur) and float(st_dur) <= float(max_samps) + 1.0):
        raise ValueError(f"st_duration ({st_dur / sf:0.1f}s) must be between 0 and the longest contiguous duration of the data ({max_samps / sf:0.1f}s).")
    good = [i for i in range(len(p.meg_picks)) if p.good_mask[i]]
    d = _decomp(p, p.good_mask)
    op_sss, op_in, op_resid = _ops(p, d, "in")
    meg = p.meg_picks
    good_rows = [meg[i] for i in good]
    for si in range(len(onsets)):
        onset = onsets[si]
        end = ends[si]
        n = end - onset
        if n <= 0:
            continue
        seg_all = _gather(raw._data, meg, onset, end)
        seg_good = _gather(raw._data, good_rows, onset, end)
        cleaned = seg_good
        if has_st:
            valid = n >= st_dur
            n_overlap = 0
            wname = "boxcar"
            if st_overlap and valid:
                n_overlap = st_dur // 2
                wname = "hann"
            ns = min(st_dur, n)
            step = ns - n_overlap
            win = _cola_window(wname, ns, step)
            nst = (n - ns) // step + 1
            acc = np.zeros(seg_all.shape if st_only else seg_good.shape)
            base = seg_all if st_only else seg_good
            for w in range(nst):
                s0 = w * step
                s1 = s0 + ns
                if w == nst - 1:
                    s1 = n
                this_len = s1 - s0
                tw = np.zeros(this_len)
                for k in range(ns):
                    tw[k] = win[k]
                if w == nst - 1:
                    for shift in range(step, this_len, step):
                        for k in range(this_len - shift):
                            tw[shift + k] += win[k]
                if w == 0:
                    nu = ns - step
                    while nu > 0:
                        for k in range(nu):
                            tw[k] += win[ns - nu + k]
                        nu -= step
                chunk = np.ascontiguousarray(base[:, s0:s1])
                orig = np.ascontiguousarray(seg_good[:, s0:s1])
                res = _tsss_apply(chunk, op_in, op_resid, orig, st_correlation, valid)
                for r in range(res.shape[0]):
                    for k in range(this_len):
                        acc[r, s0 + k] += res[r, k] * tw[k]
            if st_only:
                for r in range(len(meg)):
                    for k in range(n):
                        out._data[meg[r], onset + k] = acc[r, k]
                continue
            cleaned = acc
        res2 = np.dot(op_sss, cleaned)
        for r in range(len(meg)):
            for k in range(n):
                out._data[meg[r], onset + k] = res2[r, k]
    if not st_only:
        _reset_meg_bads(out.info)
    return out


def _gather(data: np.ndarray[float, 2], rows: List[int], s: int, e: int) -> np.ndarray[float, 2]:
    out = np.empty((len(rows), e - s))
    for a in range(len(rows)):
        out[a, :] = data[rows[a], s:e]
    return out


def compute_maxwell_basis(info: Info, origin = "auto", int_order: int = 8, ext_order: int = 3,
                          calibration = None, coord_frame: str = "head", regularize = "in",
                          ignore_ref: bool = True, bad_condition: str = "error",
                          mag_scale = 100.0, extended_proj = (), verbose = None):
    """SSS basis S, pseudoinverse pS, kept moments and n_use_in (mne 1.13.2)."""
    if len(extended_proj) > 0:
        raise NotImplementedError("extended_proj is not supported")
    p = _prep_maxwell(info, origin, int_order, ext_order, regularize, bad_condition, mag_scale,
                      calibration, None, None, None, coord_frame)
    d = _decomp(p, p.good_mask)
    rm = np.empty(len(d.reg_moments), dtype = int)
    for k in range(len(d.reg_moments)):
        rm[k] = d.reg_moments[k]
    return d.S_full, d.pS, rm, d.n_use_in


class _BadScores:
    ch_names: List[str]
    ch_types: List[str]
    bins: np.ndarray[float, 2]
    scores_flat: np.ndarray[float, 2]
    limits_flat: np.ndarray[float, 2]
    scores_noisy: np.ndarray[float, 2]
    limits_noisy: np.ndarray[float, 2]

    def __init__(self, ch_names: List[str], ch_types: List[str], bins, scores_flat, limits_flat,
                 scores_noisy, limits_noisy):
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.bins = bins
        self.scores_flat = scores_flat
        self.limits_flat = limits_flat
        self.scores_noisy = scores_noisy
        self.limits_noisy = limits_noisy

    def __getitem__(self, key: Literal[str]):
        if key == "ch_names":
            return self.ch_names
        elif key == "ch_types":
            return self.ch_types
        elif key == "bins":
            return self.bins
        elif key == "scores_flat":
            return self.scores_flat
        elif key == "limits_flat":
            return self.limits_flat
        elif key == "scores_noisy":
            return self.scores_noisy
        elif key == "limits_noisy":
            return self.limits_noisy
        else:
            compile_error("unsupported key")


def _fbm(raw, limit: float, duration: float, min_count: int, origin, int_order: int, ext_order: int,
         calibration, cross_talk, coord_frame: str, regularize, ignore_ref: bool, bad_condition: str,
         head_pos, mag_scale, skip_by_annotation, h_freq):
    r = raw
    if not isinstance(h_freq, NoneType):
        hf = float(h_freq)
        if not (r.info.lowpass > 0.0 and r.info.lowpass <= hf):
            r = raw.copy()
            r.filter(None, hf)
    sf = r.info.sfreq
    onsets, ends = _segments(r, skip_by_annotation)
    step = int(round(sf * duration))
    starts = List[int]()
    stops = List[int]()
    for k in range(len(onsets)):
        if ends[k] - onsets[k] >= step:
            s = onsets[k]
            while s <= ends[k] - step:
                starts.append(s)
                stops.append(s + step)
                s += step
            stops[len(stops) - 1] = ends[k]
    mc = min(min_count, len(starts))
    p = _prep_maxwell(raw.info, origin, int_order, ext_order, regularize, bad_condition, mag_scale,
                      calibration, cross_talk, head_pos, None, coord_frame)
    meg = p.meg_picks
    nm = len(meg)
    names = raw.info.ch_names
    noisy_cnt = Dict[str, int]()
    flat_cnt = Dict[str, int]()
    all_flats = List[str]()
    flat_step = max(20, int(30.0 * sf / 1000.0))
    ns = len(starts)
    bins = np.empty((ns, 2))
    sc_flat = np.full((nm, ns), float("nan"))
    sc_noisy = np.full((nm, ns), float("nan"))
    th_flat = np.full((nm, 1), float("nan"))
    th_noisy = np.full((nm, 1), float("nan"))
    base_good = [p.good_mask[i] for i in range(nm)]
    for si in range(ns):
        start = starts[si]
        stop = stops[si]
        orig = _gather(r._data, meg, start, stop)
        n = stop - start
        flat_stop = n - (n % flat_step)
        bins[si, 0] = (r.first_samp + start) / sf
        bins[si, 1] = (r.first_samp + start + n - 1) / sf
        nseg = flat_stop // flat_step
        chunk_flats = List[str]()
        for a in range(nm):
            if not base_good[a]:
                continue
            lim = 0.01e-15
            if not p.mag_mask[a]:
                lim = 0.01e-13
            th_flat[a, 0] = lim
            dmin = float("inf")
            for sg in range(nseg):
                mu = 0.0
                for k in range(flat_step):
                    mu += orig[a, sg * flat_step + k]
                mu /= flat_step
                var = 0.0
                for k in range(flat_step):
                    dd = orig[a, sg * flat_step + k] - mu
                    var += dd * dd
                sd = math.sqrt(var / flat_step)
                if sd < dmin:
                    dmin = sd
            sc_flat[a, si] = dmin
            if dmin < lim:
                nmn = names[meg[a]]
                chunk_flats.append(nmn)
                flat_cnt[nmn] = flat_cnt.get(nmn, 0) + 1
                if nmn not in all_flats:
                    all_flats.append(nmn)
        these = List[int]()
        for a in range(nm):
            if base_good[a] and names[meg[a]] not in all_flats:
                these.append(a)
        if len(these) == 0:
            break
        chunk_noisy = List[str]()
        for n_iter in range(1, 101):
            gm = [(names[meg[a]] not in raw.info.bads) and (names[meg[a]] not in chunk_noisy)
                  and (names[meg[a]] not in all_flats) for a in range(nm)]
            d = _decomp(p, gm)
            op_sss, op_in, op_resid = _ops(p, d, "orig")
            gidx = [meg[a] for a in range(nm) if gm[a]]
            gdat = _gather(r._data, gidx, start, stop)
            filt = np.dot(op_sss, gdat)
            rng = np.empty(len(these))
            for q in range(len(these)):
                a = these[q]
                lo = float("inf")
                hi = -float("inf")
                for k in range(n):
                    v = orig[a, k] - filt[a, k]
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
                rng[q] = (hi - lo) * p.coil_scale[a]
            mean = float(np.mean(rng))
            std = float(np.std(rng))
            zbest = -float("inf")
            idx = 0
            for q in range(len(these)):
                z = (rng[q] - mean) / std
                sc_noisy[these[q], si] = z
                th_noisy[these[q], 0] = limit
                if z > zbest:
                    zbest = z
                    idx = q
            if zbest < limit:
                break
            chunk_noisy.append(names[meg[these[idx]]])
            del these[idx]
        for nmn in chunk_noisy:
            noisy_cnt[nmn] = noisy_cnt.get(nmn, 0) + 1
    noisy = List[str]()
    flat = List[str]()
    for a in range(nm):
        nmn = names[meg[a]]
        if noisy_cnt.get(nmn, 0) >= mc and nmn in noisy_cnt:
            noisy.append(nmn)
        if flat_cnt.get(nmn, 0) >= mc and nmn in flat_cnt:
            flat.append(nmn)
    return noisy, flat, bins, sc_flat, th_flat, sc_noisy, th_noisy


def find_bad_channels_maxwell(raw, limit: float = 7.0, duration: float = 5.0, min_count: int = 5,
                              return_scores: Literal[bool] = False, origin = "auto", int_order: int = 8,
                              ext_order: int = 3, calibration = None, cross_talk = None,
                              coord_frame: str = "head", regularize = "in", ignore_ref: bool = False,
                              bad_condition: str = "error", head_pos = None, mag_scale = 100.0,
                              skip_by_annotation = ("edge", "bad_acq_skip"), h_freq = 40.0,
                              extended_proj = (), mc_interp: str = "hann", verbose = None):
    """Find bad (noisy / flat) MEG channels by Maxwell filtering (mne 1.13.2)."""
    if len(extended_proj) > 0:
        raise NotImplementedError("extended_proj is not supported")
    noisy, flat, bins, sf_, tf_, sn_, tn_ = _fbm(raw, float(limit), duration, min_count, origin, int_order,
                                                 ext_order, calibration, cross_talk, coord_frame,
                                                 regularize, ignore_ref, bad_condition, head_pos,
                                                 mag_scale, skip_by_annotation, h_freq)
    if return_scores:
        meg = List[int]()
        for i in range(raw.info.nchan):
            if raw.info.ch_types[i] == "mag" or raw.info.ch_types[i] == "grad":
                meg.append(i)
        sc = _BadScores([raw.info.ch_names[i] for i in meg], [raw.info.ch_types[i] for i in meg],
                        bins, sf_, tf_, sn_, tn_)
        return noisy, flat, sc
    else:
        return noisy, flat


def maxwell_filter_prepare_emptyroom(raw_er, *, raw, bads: str = "from_raw", annotations: str = "from_raw",
                                     meas_date: str = "keep", emit_warning: bool = False, verbose = None):
    """Prepare an empty-room recording for Maxwell filtering (mne 1.13.2)."""
    if bads not in ("from_raw", "union", "keep"):
        raise ValueError(f"bads must be one of ['from_raw', 'union', 'keep'], got {bads}")
    if annotations not in ("from_raw", "union", "keep"):
        raise ValueError(f"annotations must be one of ['from_raw', 'union', 'keep'], got {annotations}")
    if meas_date not in ("keep", "from_raw"):
        raise ValueError(f"meas_date must be one of ['keep', 'from_raw'], got {meas_date}")
    er = raw_er.copy()
    nb = List[str]()
    if bads == "from_raw":
        nb = list(raw.info.bads)
    elif bads == "union":
        u = Set[str]()
        for b in raw.info.bads:
            u.add(b)
        for b in er.info.bads:
            u.add(b)
        nb = sorted(list(u))
    else:
        nb = list(er.info.bads)
    meg_names = List[str]()
    for i in range(er.info.nchan):
        if er.info.ch_types[i] == "mag" or er.info.ch_types[i] == "grad":
            meg_names.append(er.info.ch_names[i])
    er.info.bads = [b for b in nb if b in meg_names]
    shift = raw.first_time - er.first_time
    ann = er.annotations.copy()
    ann.onset = [o + shift for o in ann.onset]
    er.annotations = ann
    er.first_samp = int(round(raw.first_time * er.info.sfreq))
    if annotations != "keep":
        a2 = Annotations(List[float](), List[float](), List[str](), er.info.meas_date)
        if annotations == "union":
            a2 = er.annotations.copy()
        ra = raw.annotations
        for k in range(len(ra)):
            a2.onset.append(ra.onset[k])
            a2.duration.append(ra.duration[k])
            a2.description.append(ra.description[k])
        if er.info.meas_date is None:
            a2.onset = [o - er.first_time for o in a2.onset]
        er.set_annotations(a2, emit_warning)
    if meas_date == "from_raw":
        er.info.meas_date = raw.info.meas_date
    return er
