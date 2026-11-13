# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# GDF 1.x / 2.x reader (mne/io/edf/edf.py: _read_gdf_header, RawGDF).
# Channels with differing sampling rates are not supported.

import numpy as np
from ._bytes import _read_file, _ascii, _u8, _uint_le, _sint, _f32_le, _f64_le
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations
from .edf import _prefilter_value, _prefilter_float

_GDF_BYTES = [0, 1, 1, 2, 2, 4, 4, 8, 8, 0, 0, 0, 0, 0, 0, 0, 4, 8]


def _gstr(b: bytes, off: int, n: int) -> str:
    s = _ascii(b, off, n)
    z = s.find("\0")
    if z >= 0:
        s = s[:z]
    return s


def _i64(b: bytes, off: int) -> int:
    u = _uint_le(b, off, 4) | (_uint_le(b, off + 4, 4) << 32)
    return u


def _gval(b: bytes, off: int, t: int) -> float:
    if t == 1:
        return float(_sint(_u8(b, off), 1))
    if t == 2:
        return float(_u8(b, off))
    if t == 3:
        return float(_sint(_uint_le(b, off, 2), 2))
    if t == 4:
        return float(_uint_le(b, off, 2))
    if t == 5:
        return float(_sint(_uint_le(b, off, 4), 4))
    if t == 6:
        return float(_uint_le(b, off, 4))
    if t == 7:
        return float(_i64(b, off))
    if t == 8:
        return float(_uint_le(b, off, 8))
    if t == 16:
        return _f32_le(b, off)
    if t == 17:
        return _f64_le(b, off)
    raise ValueError(f"Unsupported GDF data type {t}")


def _days_from_civil(y: int, m: int, d: int) -> int:
    yy = y - 1 if m <= 2 else y
    era = (yy if yy >= 0 else yy - 399) // 400
    yoe = yy - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def read_raw_gdf(input_fname, eog = None, misc = None, stim_channel = "auto", exclude = (),
                 preload = False, include = None, verbose = None) -> RawArray:
    """mne.io.read_raw_gdf (GDF 1.x and 2.x)."""
    fn = str(input_fname)
    if not fn.lower().endswith(".gdf"):
        raise NotImplementedError(f"Only GDF files are supported, got {fn.split('.')[-1]}.")
    b = _read_file(fn)
    ver = _ascii(b, 0, 8)
    if not ver.startswith("GDF"):
        raise ValueError("Bad GDF file provided.")
    number = float(ver[4:8])
    meas_date: Optional[float] = None
    n_records = 0
    rl0 = 1
    rl1 = 1
    nchan = 0
    header_nbytes = 0
    p = 0
    is1 = number < 1.9
    if is1:
        tm = _gstr(b, 168, 16).strip()
        if len(tm) >= 16:
            if tm[14:16] == "  ":
                tm = tm[:14] + "00" + tm[16:]
            if tm[:14].isdigit() and tm[14:16].isdigit():
                dd = _days_from_civil(int(tm[0:4]), int(tm[4:6]), int(tm[6:8]))
                meas_date = float(dd * 86400 + int(tm[8:10]) * 3600 + int(tm[10:12]) * 60 + int(tm[12:14])) + int(tm[14:16]) * 0.01
        header_nbytes = _uint_le(b, 184, 8)
        n_records = _uint_le(b, 236, 8)
        rl0 = _uint_le(b, 244, 4)
        rl1 = _uint_le(b, 248, 4)
        nchan = _uint_le(b, 252, 4)
        p = 256
    else:
        mdu = _uint_le(b, 168, 8)
        if mdu != 0:
            meas_date = -62135596800.0 + (float(mdu) * (2.0 ** -32) - 367.0) * 86400.0
        header_nbytes = _uint_le(b, 184, 2) * 256
        n_records = _uint_le(b, 236, 8)
        rl0 = _uint_le(b, 244, 4)
        rl1 = _uint_le(b, 248, 4)
        nchan = _uint_le(b, 252, 2)
        p = 256
    if rl0 == 0:
        rl0 = 1
        print("RuntimeWarning: Header information is incorrect for record length. Default record length set to 1.")
    labels = [_gstr(b, p + 16 * i, 16).strip() for i in range(nchan)]
    p += 16 * nchan
    units = List[float]()
    pmin = List[float]()
    pmax = List[float]()
    dmin = List[float]()
    dmax = List[float]()
    hp_s = List[str]()
    lp_s = List[str]()
    hp_f = List[float]()
    lp_f = List[float]()
    if is1:
        p += 80 * nchan
        for i in range(nchan):
            u = _gstr(b, p + 8 * i, 8).strip()
            units.append(1e-6 if u[:2] == "uV" else 1.0)
        p += 8 * nchan
        for i in range(nchan):
            pmin.append(_f64_le(b, p + 8 * i))
        p += 8 * nchan
        for i in range(nchan):
            pmax.append(_f64_le(b, p + 8 * i))
        p += 8 * nchan
        for i in range(nchan):
            dmin.append(float(_i64(b, p + 8 * i)))
        p += 8 * nchan
        for i in range(nchan):
            dmax.append(float(_i64(b, p + 8 * i)))
        p += 8 * nchan
        for i in range(nchan):
            pf = _gstr(b, p + 80 * i, 80)
            hp_s.append(_prefilter_value(pf, "HP"))
            lp_s.append(_prefilter_value(pf, "LP"))
        p += 80 * nchan
    else:
        p += 80 * nchan + 6 * nchan
        for i in range(nchan):
            u = _uint_le(b, p + 2 * i, 2)
            if u == 4275:
                units.append(1e-6)
            elif u == 4274:
                units.append(1e-3)
            elif u == 512 or u == 0:
                units.append(1.0)
            else:
                print(f"RuntimeWarning: Unsupported physical dimension for channel {i} (assuming dimensionless). Please contact the MNE-Python developers for support.")
                units.append(1.0)
        p += 2 * nchan
        for i in range(nchan):
            pmin.append(_f64_le(b, p + 8 * i))
        p += 8 * nchan
        for i in range(nchan):
            pmax.append(_f64_le(b, p + 8 * i))
        p += 8 * nchan
        for i in range(nchan):
            dmin.append(_f64_le(b, p + 8 * i))
        p += 8 * nchan
        for i in range(nchan):
            dmax.append(_f64_le(b, p + 8 * i))
        p += 8 * nchan + 68 * nchan
        for i in range(nchan):
            lp_f.append(_f32_le(b, p + 4 * i))
        p += 4 * nchan
        for i in range(nchan):
            hp_f.append(_f32_le(b, p + 4 * i))
        p += 4 * nchan + 4 * nchan
    n_samps = [_sint(_uint_le(b, p + 4 * i, 4), 4) for i in range(nchan)]
    p += 4 * nchan
    dtypes = [_sint(_uint_le(b, p + 4 * i, 4), 4) for i in range(nchan)]
    bytes_tot = 0
    for i in range(nchan):
        bytes_tot += _GDF_BYTES[dtypes[i]] * n_samps[i]
    for i in range(nchan):
        if _GDF_BYTES[dtypes[i]] != _GDF_BYTES[dtypes[0]]:
            raise RuntimeError("Reading multiple data types not supported")
    # events
    ev_pos = List[int]()
    ev_typ = List[int]()
    ev_dur = List[int]()
    etp = header_nbytes + n_records * bytes_tot
    if etp < len(b):
        etmode = _u8(b, etp)
        q = etp + 1
        n_events = 0
        have = False
        if is1:
            if etmode == 1 or etmode == 3:
                q += 3
                n_events = _uint_le(b, q, 4)
                q += 4
                have = True
        else:
            if number < 1.94:
                q += 3
                n_events = _uint_le(b, q, 4)
                q += 4
            else:
                n_events = _uint_le(b, q, 3)
                q += 3 + 4
            have = True
        if have:
            for i in range(n_events):
                ev_pos.append(_uint_le(b, q + 4 * i, 4) - 1)
            q += 4 * n_events
            for i in range(n_events):
                ev_typ.append(_uint_le(b, q + 2 * i, 2))
            q += 2 * n_events
            if etmode == 3:
                q += 2 * n_events
                for i in range(n_events):
                    ev_dur.append(max(_uint_le(b, q + 4 * i, 4), 1))
            else:
                for i in range(n_events):
                    ev_dur.append(1)
    # channel selection
    excl = List[int]()
    if not isinstance(include, NoneType):
        inc = [str(x) for x in include]
        excl = [i for i in range(nchan) if labels[i] not in inc]
    else:
        ex = [str(x) for x in exclude]
        excl = [i for i in range(nchan) if labels[i] in ex]
    sel = [i for i in range(nchan) if i not in excl]
    names = [labels[i] for i in sel]
    valid = List[str]()
    if isinstance(stim_channel, bool):
        if stim_channel:
            valid = ["status", "trigger"]
    elif isinstance(stim_channel, str):
        valid = ["status", "trigger"] if stim_channel == "auto" else [stim_channel.lower()]
    elif isinstance(stim_channel, NoneType):
        pass
    else:
        valid = [str(s).lower() for s in stim_channel]
    low = [n.lower() for n in names]
    stim_idx = List[int]()
    for v in sorted(set(valid)):
        if v in low:
            stim_idx.append(low.index(v))
    eogl = List[str]() if isinstance(eog, NoneType) else [str(x) for x in eog]
    miscl = List[str]() if isinstance(misc, NoneType) else [str(x) for x in misc]
    types = List[str]()
    for k in range(len(names)):
        t = "eeg"
        if names[k] in eogl:
            t = "eog"
        elif names[k] in miscl:
            t = "misc"
        elif k in stim_idx:
            t = "stim"
        types.append(t)
    not_stim = [k for k in range(len(sel)) if k not in stim_idx]
    if len(not_stim) == 0:
        not_stim = list(range(len(sel)))
    max_samp = max([n_samps[sel[k]] for k in not_stim])
    for i in sel:
        if n_samps[i] != max_samp:
            raise NotImplementedError("Typhon mne: GDF channels with different sampling rates are not supported")
    sfreq = float(max_samp) * float(rl1) / float(rl0)
    info = Info(names, sfreq, types)
    info.meas_date = meas_date
    for k in range(len(names)):
        if types[k] == "stim":
            info.units[k] = -1
            info.coil_types[k] = 0
        elif types[k] != "eeg":
            info.coil_types[k] = 0
            info.units[k] = 107
    filt = [k for k in range(len(sel)) if k not in stim_idx]
    if len(filt) > 0:
        for key in ("HP", "LP"):
            vals = List[float]()
            for k in filt:
                if is1:
                    s = hp_s[sel[k]] if key == "HP" else lp_s[sel[k]]
                    vals.append(_prefilter_float(s) if s != "" else np.nan)
                else:
                    vals.append(hp_f[sel[k]] if key == "HP" else lp_f[sel[k]])
            v = 0.0
            if len(set(vals)) > 1:
                fv = [x for x in vals if not np.isnan(x)]
                if len(fv) == 0:
                    v = np.nan
                else:
                    v = max(fv) if key == "HP" else min(fv)
            else:
                v = vals[0]
            if not np.isnan(v) and v != 0:
                if key == "HP":
                    info.highpass = v
                else:
                    info.lowpass = v
    if np.isnan(info.lowpass) or info.lowpass <= 0:
        info.lowpass = sfreq / 2.0
    if info.highpass > info.lowpass:
        print(f"RuntimeWarning: Highpass cutoff frequency {info.highpass} is greater than lowpass cutoff frequency {info.lowpass}, setting values to 0 and Nyquist.")
        info.highpass = 0.0
        info.lowpass = sfreq / 2.0
    # data
    nsamp = n_records * max_samp
    data = np.zeros((len(sel), nsamp))
    offs = [0]
    for s in n_samps:
        offs.append(offs[-1] + s)
    bpd = _GDF_BYTES[dtypes[0]]
    rec_total = offs[-1]
    for k in range(len(sel)):
        i = sel[k]
        t = dtypes[i]
        dr = dmax[i] - dmin[i]
        if not np.isfinite(dr) or dr == 0:
            dr = 1.0
        pr = pmax[i] - pmin[i]
        if pr == 0:
            pr = 1.0
        cal = pr / dr
        off = pmin[i] - dmin[i] * cal
        gain = units[i]
        if k in stim_idx:
            cal = 1.0
            off = 0.0
            gain = 1.0
        for r in range(n_records):
            base = header_nbytes + r * rec_total * bpd + offs[i] * bpd
            for q in range(max_samp):
                data[k, r * max_samp + q] = (_gval(b, base + q * bpd, t) * cal + off) * gain
    raw = RawArray(data, info)
    raw._raw_kind = "RawGDF"
    if len(ev_pos) > 0:
        on = [float(v) / sfreq for v in ev_pos]
        du = [float(v) / sfreq for v in ev_dur]
        de = [str(v) for v in ev_typ]
        raw.set_annotations(Annotations(on, du, de), False)
    return raw
