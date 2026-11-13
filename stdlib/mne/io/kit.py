# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# KIT / Yokogawa .sqd/.con reader (mne/io/kit/kit.py). Narrower: continuous
# data only, no digitization (mrk/elp/hsp), no dig/HPI directory parsing.

import numpy as np
from math import sin, cos, sqrt
from ._bytes import _read_file, _ascii, _i16_le, _i32_le, _f64_le, _u8
from .._info import Info
from .._containers import RawArray

_GAINS = [1, 2, 5, 10, 20, 50, 100, 200]
_HPFS1 = [0.0, 1.0, 3.0, 3.0]
_HPFS2 = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
_HPFS4 = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0, 500.0]
_LPFS1 = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
_LPFS3 = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 10000.0]
_LPFS4 = [10.0, 30.0, 100.0, 300.0, 1000.0, 2000.0, 5000.0, 10000.0]


def _hpf(opt: int, i: int) -> float:
    if opt == 1:
        return _HPFS1[i]
    if opt == 4:
        return _HPFS4[i]
    return _HPFS2[i]


def _lpf(opt: int, i: int) -> float:
    if opt == 3:
        return _LPFS3[i]
    if opt == 4:
        return _LPFS4[i]
    return _LPFS1[i]


def _fll(t: int) -> Tuple[int, int, int]:
    if t == 0 or t == 10 or t == 20:
        return (1, 1, 1)
    if t == 50 or t == 60:
        return (2, 1, 1)
    if t == 100 or t == 120:
        return (3, 3, 3)
    if t == 101:
        return (1, 3, 2)
    if t == 200:
        return (4, 4, 3)
    return (2, 2, 2)


def _is_meg(t: int) -> bool:
    return (t >= 1 and t <= 4) or (t >= 0x101 and t <= 0x104)


def _name_n(t: int) -> int:
    if t == 1 or t == 2:
        return 6
    if t == -2:
        return 8
    return 32


def _cs(b: bytes, off: int, n: int) -> str:
    s = _ascii(b, off, n)
    k = s.find("\0")
    if k >= 0:
        s = s[:k]
    return s


def read_raw_kit(input_fname, mrk = None, elp = None, hsp = None, stim = ">", slope: str = "-",
                 stimthresh = 1, preload = False, stim_code: str = "binary",
                 allow_unknown_format: bool = False, standardize_names = None,
                 verbose = None) -> RawArray:
    fn = str(input_fname)
    b = _read_file(fn)
    # directories
    d_off = List[int]()
    d_size = List[int]()
    d_cnt = List[int]()
    p = 0
    ndir = 1
    i = 0
    while i < ndir:
        d_off.append(_i32_le(b, p) & 0xFFFFFFFF)
        d_size.append(_i32_le(b, p + 4))
        d_cnt.append(_i32_le(b, p + 12))
        if i == 0:
            ndir = d_cnt[0]
        p += 16
        i += 1
    o = d_off[1]
    version = _i32_le(b, o)
    revision = _i32_le(b, o + 4)
    if version < 2 or (version == 2 and revision < 3):
        if not allow_unknown_format:
            raise ValueError(f"SQD file format V{version}R{revision:03d} is not officially supported. Set allow_unknown_format=True to load it anyways.")
        raise OSError("Legacy KIT formats are not supported")
    sysid = _i32_le(b, o + 8)
    system_name = _cs(b, o + 12, 128)
    model_name = _cs(b, o + 140, 128)
    nchan = _i32_le(b, o + 268)
    q = o + 272 + 256
    create_time = _i32_le(b, q)
    q += 8 + 12
    q += 4 + 12
    fll_type = _i32_le(b, q)
    q += 4 + 12
    q += 4 + 12
    q += 4 + 116
    adc_range = 0.0
    if version < 2 or (version == 2 and revision <= 3):
        adc_range = float(_i32_le(b, q))
        q += 4
    else:
        adc_range = _f64_le(b, q)
        q += 8
    adc_alloc = _i32_le(b, q + 4)
    adc_stored = _i32_le(b, q + 8)
    system_name = system_name.strip().replace("\n", "/")
    model_name = model_name.strip().replace("\n", "/")
    description = f"{system_name} ({sysid}) V{version}R{revision:03d} {model_name}"
    isz = adc_alloc // 8
    if isz != 2 and isz != 4:
        raise OSError("Unsupported KIT ADC word size")
    if fll_type not in (0, 10, 20, 50, 60, 100, 101, 120, 200, 300):
        use = 0
        for c in (0, 10, 20, 50, 60, 100, 101, 120, 200, 300):
            if c < fll_type:
                use = c
        print("RuntimeWarning: Unknown site filter settings (FLL)")
        fll_type = use
    # channels
    co = d_off[4]
    cs = d_size[4]
    ctype = List[int]()
    cname = List[str]()
    cloc = List[List[float]]()
    exg = List[float]()
    for i in range(nchan):
        r = co + cs * i
        t = _i32_le(b, r)
        if sysid == 52 and i < 160 and t == 0:
            t = 0x101
        loc5 = [0.0, 0.0, 0.0, 0.0, 0.0]
        nm = ""
        if _is_meg(t):
            if t != 0x101 and t != 2:
                raise NotImplementedError(f"KIT channel type {t} can not be read. Please contact the mne-python developers.")
            for k in range(5):
                loc5[k] = _f64_le(b, r + 4 + k * 8)
            if t == 1 or t == 2:
                nm = _cs(b, r + 4 + 40 + 16, 6)
        elif t == -1 or t == -2 or t == -3 or t == -4:
            nm = _cs(b, r + 4 + 8, _name_n(t))
            if t == -2 or t == -3:
                off2 = 6 if t == -2 else 8
                exg.append(_f64_le(b, r + 4 + 8 + _name_n(t) + off2))
        elif t == 0:
            pass
        else:
            raise OSError(f"Unknown KIT channel type: {t}")
        ctype.append(t)
        cname.append(nm)
        cloc.append(loc5)
    # calibration
    cb = d_off[5]
    gain = List[float]()
    for i in range(nchan):
        gain.append(_f64_le(b, cb + (i * 2 + 1) * 8))
    # amp gain
    amp = _i32_le(b, d_off[7])
    amp_gain = 1.0
    hpf = 0
    lpf = 0
    bef = 0
    if fll_type >= 100:
        g1 = (amp & 0x00007000) >> 12
        g2 = (amp & 0x70000000) >> 28
        g3 = (amp & 0x07000000) >> 24
        amp_gain = float(_GAINS[g1] * _GAINS[g2] * _GAINS[g3])
        hpf = (amp & 0x00000700) >> 8
        lpf = (amp & 0x00070000) >> 16
        bef = amp & 3
    else:
        ig = (amp & 0x1800) >> 11
        og = amp & 0x0007
        amp_gain = float(_GAINS[ig] * _GAINS[og])
        hpf = (amp & 0x007) >> 4
        lpf = (amp & 0x0700) >> 8
        bef = (amp & 0xC000) >> 14
    fl = _fll(fll_type)
    highpass = _hpf(fl[0], hpf)
    lowpass = _lpf(fl[1], lpf)
    # acquisition
    ao = d_off[8]
    acq = _i32_le(b, ao)
    sfreq = _f64_le(b, ao + 4)
    if acq != 1:
        raise TypeError("SQD file contains epochs, not raw data. Wrong reader.")
    n_samples = _i32_le(b, ao + 16)
    # conversion factors
    ad_to_volt = adc_range / (2.0 ** adc_stored)
    conv = List[float]()
    ei = 0
    for i in range(nchan):
        t = ctype[i]
        if _is_meg(t):
            conv.append(ad_to_volt / amp_gain * gain[i])
        elif t == -2 or t == -3:
            g = exg[ei] / (2.0 ** (adc_stored - 14))
            ei += 1
            conv.append(ad_to_volt if g == 0.0 else g)
        else:
            conv.append(ad_to_volt)
    # channel info
    names = List[str]()
    types = List[str]()
    cnt = Dict[str, int]()
    for i in range(nchan):
        t = ctype[i]
        nm = cname[i]
        if _is_meg(t):
            if nm == "" or (standardize_names is not None and bool(standardize_names)):
                nm = f"MEG {i + 1:03d}"
            names.append(nm)
            types.append("mag" if t == 2 else "ref_meg")
        else:
            lab = "MISC"
            ty = "misc"
            if t == -1:
                lab = "TRIGGER"
            elif t == -2:
                lab = "EEG"
                ty = "eeg"
            elif t == -3:
                lab = "ECG"
                ty = "ecg"
            cnt[lab] = (cnt[lab] if lab in cnt else 0) + 1
            if nm == "" or nm == "EEG" or (standardize_names is not None and bool(standardize_names)):
                nm = f"{lab} {cnt[lab]:03d}"
            names.append(nm)
            types.append(ty)
    # stim channel synthesis
    stim_idx = List[int]()
    misc_idx = List[int]()
    for i in range(nchan):
        if types[i] == "misc":
            misc_idx.append(i)
    misc_idx = misc_idx[:8]
    has_stim = stim is not None
    if has_stim:
        if isinstance(stim, str):
            if stim == "<":
                stim_idx = misc_idx[::-1]
            elif stim == ">":
                stim_idx = misc_idx
            else:
                raise ValueError(f"stim needs to be list of int, '>' or '<', not {stim!r}")
        else:
            for x in stim:
                stim_idx.append(int(x))
                if int(x) >= nchan:
                    raise ValueError(f"Got stim, but sqd file only has {nchan} channels.")
        names.append("STI 014")
        types.append("stim")
    nout = len(names)
    info = Info(names, sfreq, types)
    for i in range(nchan):
        t = ctype[i]
        if _is_meg(t):
            info.coil_types[i] = 6001 if t == 2 else 6002
            info.units[i] = 112
            th = cloc[i][3] * 3.141592653589793 / 180.0
            ph = cloc[i][4] * 3.141592653589793 / 180.0
            vz = [sin(th) * cos(ph), sin(th) * sin(ph), cos(th)]
            nz = sqrt(vz[0] * vz[0] + vz[1] * vz[1] + vz[2] * vz[2])
            vz = [vz[0] / nz, vz[1] / nz, vz[2] / nz]
            vx = [0.0, 0.0, 0.0]
            if vz[1] < vz[2]:
                if vz[0] < vz[1]:
                    vx[0] = 1.0
                else:
                    vx[1] = 1.0
            elif vz[0] < vz[2]:
                vx[0] = 1.0
            else:
                vx[2] = 1.0
            dp = vx[0] * vz[0] + vx[1] * vz[1] + vx[2] * vz[2]
            vx = [vx[0] - dp * vz[0], vx[1] - dp * vz[1], vx[2] - dp * vz[2]]
            nx = sqrt(vx[0] * vx[0] + vx[1] * vx[1] + vx[2] * vx[2])
            vx = [vx[0] / nx, vx[1] / nx, vx[2] / nx]
            vy = [vz[1] * vx[2] - vz[2] * vx[1], vz[2] * vx[0] - vz[0] * vx[2], vz[0] * vx[1] - vz[1] * vx[0]]
            pos = [cloc[i][0], cloc[i][1], cloc[i][2]]
            loc = List[float]()
            for v in (pos, vx, vy, vz):
                # ALS -> RAS: (x, y, z) -> (-y, x, z)
                loc.append(-v[1])
                loc.append(v[0])
                loc.append(v[2])
            info.locs[i] = loc
        else:
            info.units[i] = 107
            info.coil_types[i] = 1 if t == -2 else 0
            info.locs[i] = [0.0 for _ in range(12)]
    if has_stim:
        info.units[nchan] = -1
        info.coil_types[nchan] = 0
    info.highpass = highpass
    info.lowpass = lowpass
    info.meas_date = float(create_time)
    info.description = description
    data = np.zeros((nout, n_samples))
    base = d_off[9]
    for s in range(n_samples):
        for c in range(nchan):
            off = base + (s * nchan + c) * isz
            v = _i32_le(b, off) if isz == 4 else _i16_le(b, off)
            data[c, s] = float(v) * conv[c]
        if has_stim:
            acc = 0
            for k in range(len(stim_idx)):
                x = data[stim_idx[k], s]
                hit = (x > float(stimthresh)) if slope == "+" else (x < float(stimthresh))
                if slope != "+" and slope != "-":
                    raise ValueError("slope needs to be '+' or '-'")
                if hit:
                    if stim_code == "binary":
                        acc += 1 << k
                    else:
                        acc += stim_idx[k]
            data[nchan, s] = float(acc)
    raw = RawArray(data, info)
    raw._raw_kind = "RawKIT"
    return raw
