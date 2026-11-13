# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# Blackrock NSx reader (mne/io/nsx/nsx.py): NEURALCD / BRSMPGRP (spec >= 2.2).

import numpy as np
from ._bytes import _read_file, _ascii, _u8, _uint_le, _i16_le
from ._fmtutil import _utc_secs
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations


def _cs(b: bytes, off: int, n: int) -> str:
    s = _ascii(b, off, n)
    k = s.find("\0")
    if k >= 0:
        s = s[:k]
    return s


def _unit_scale(u: str) -> float:
    # scaling from SI (V) to the stated unit
    if u.startswith("u"):
        return 1e6
    if u.startswith("m"):
        return 1e3
    if u.startswith("n"):
        return 1e9
    return 1.0


def read_raw_nsx(input_fname, stim_channel = "auto", eog = None, misc = None,
                 preload = False, verbose = None) -> RawArray:
    fn = str(input_fname)
    low = fn.lower()
    j = low.rfind(".")
    if j < 0 or not low[j:].startswith(".ns"):
        raise NotImplementedError(f"Only NSx files are supported, got {fn[j:]}.")
    b = _read_file(fn)
    fid = _cs(b, 0, 8)
    if fid == "NEURALSG":
        raise NotImplementedError("NSx file id (= NEURALSG), i.e., file version 2.1 is currently not supported.")
    if fid != "NEURALCD" and fid != "BRSMPGRP":
        raise ValueError(f"NSx file id (={fid}) does not match with supported file ids: ('NEURALCD', 'BRSMPGRP')")
    ver_major = _u8(b, 8)
    ver_minor = _u8(b, 9)
    bih = _uint_le(b, 10, 4)
    period = _uint_le(b, 314, 4)
    tres = _uint_le(b, 318, 4)
    yr = _uint_le(b, 322, 2)
    mo = _uint_le(b, 324, 2)
    dy = _uint_le(b, 328, 2)
    hr = _uint_le(b, 330, 2)
    mi = _uint_le(b, 332, 2)
    sc = _uint_le(b, 334, 2)
    nch = _uint_le(b, 338, 4)
    names = List[str]()
    types = List[str]()
    cals = List[float]()
    rngs = List[float]()
    his = List[float]()
    los = List[float]()
    base = 342
    for i in range(nch):
        o = base + i * 66
        names.append(_cs(b, o + 4, 16))
        mind = float(_i16_le(b, o + 22))
        maxd = float(_i16_le(b, o + 24))
        mina = float(_i16_le(b, o + 26))
        maxa = float(_i16_le(b, o + 28))
        units = _cs(b, o + 30, 16)
        cals.append((maxa - mina) / (maxd - mind))
        rngs.append(1.0 / _unit_scale(units))
        his.append(float(_uint_le(b, o + 46, 4)) * 1e-3)
        los.append(float(_uint_le(b, o + 56, 4)) * 1e-3)
        types.append(_cs(b, o, 2))
    # stim channel selection
    stim_idx = List[int]()
    if isinstance(stim_channel, str):
        want = List[str]()
        if stim_channel == "auto":
            want = ["status", "trigger"]
        else:
            want = [stim_channel.lower()]
        for i in range(nch):
            if names[i].lower() in want:
                stim_idx.append(i)
    elif isinstance(stim_channel, int):
        stim_idx.append(int(stim_channel))
    ek = List[str]()
    ei = List[int]()
    if eog is not None:
        for x in eog:
            if isinstance(x, str):
                ek.append(x)
            else:
                ei.append(int(x))
    mk = List[str]()
    mi_ = List[int]()
    if misc is not None:
        for x in misc:
            if isinstance(x, str):
                mk.append(x)
            else:
                mi_.append(int(x))
    kinds = List[str]()
    for i in range(nch):
        k = "seeg"
        if names[i] in ek or i in ei or (i - nch) in ei:
            k = "eog"
        elif names[i] in mk or i in mi_ or (i - nch) in mi_:
            k = "misc"
        elif i in stim_idx:
            k = "stim"
        kinds.append(k)
    sfreq = float(tres) / float(period)
    info = Info(names, sfreq, kinds)
    for i in range(nch):
        info.cals[i] = cals[i]
        info.ranges[i] = rngs[i]
        if kinds[i] == "seeg":
            info.coil_types[i] = 1
        if kinds[i] == "stim":
            info.units[i] = -1
    info.meas_date = _utc_secs(yr, mo, dy, hr, mi, sc) + float(_uint_le(b, 336, 2)) / 1000.0
    # online filters
    hp = his[:128]
    lp = los[:128]
    same_h = True
    for x in hp:
        if x != hp[0]:
            same_h = False
    if same_h:
        info.highpass = hp[0]
    else:
        info.highpass = max(hp)
        print("RuntimeWarning: Channels contain different highpass filters. Highest filter setting will be stored.")
    same_l = True
    for x in lp:
        if x != lp[0]:
            same_l = False
    if same_l:
        if lp[0] != 0.0:
            info.lowpass = lp[0]
    else:
        info.lowpass = min(lp)
        print("RuntimeWarning: Channels contain different lowpass filters. Lowest filter setting will be stored.")
    # data packets
    ts = List[int]()
    npts = List[int]()
    offs = List[int]()
    off = bih
    while off < len(b):
        hdr = _u8(b, off)
        if ver_major + 0.1 * ver_minor < 3.0 and not (ver_major >= 3):
            t = _uint_le(b, off + 1, 4)
            n = _uint_le(b, off + 5, 4)
            hsz = 9
        else:
            t = _uint_le(b, off + 1, 8)
            n = _uint_le(b, off + 9, 4)
            hsz = 13
        if hdr == 1:
            ts.append(t // period)
            npts.append(n)
            offs.append(off + hsz)
        off = off + hsz + n * nch * 2
    if len(ts) == 0:
        raise RuntimeError("NSx file appears to be broken")
    first = ts[0]
    n_samples = ts[len(ts) - 1] + npts[len(npts) - 1] - first
    data = np.zeros((nch, n_samples))
    for p in range(len(ts)):
        s0 = ts[p] - first
        for s in range(npts[p]):
            if s0 + s >= n_samples:
                break
            for c in range(nch):
                data[c, s0 + s] = float(_i16_le(b, offs[p] + (s * nch + c) * 2)) * cals[c] * rngs[c]
    raw = RawArray(data, info)
    raw._raw_kind = "RawNSX"
    raw.first_samp = first
    if len(ts) > 1:
        on = List[float]()
        du = List[float]()
        ds = List[str]()
        for p in range(len(ts) - 1):
            st = ts[p] + npts[p] + 1
            sp = ts[p + 1] - 1
            on.append(float(st) / sfreq)
            du.append(float(sp - st + 1) / sfreq)
            ds.append("BAD_ACQ_SKIP")
        raw.set_annotations(Annotations(on, du, ds, info.meas_date), False)
    return raw
