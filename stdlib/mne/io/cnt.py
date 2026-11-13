# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# Neuroscan CNT reader (mne/io/cnt/cnt.py + _utils.py). Data is always
# loaded. meas_date is interpreted as UTC.

import numpy as np
from ._bytes import _read_file, _ascii, _u8, _uint_le, _i16_le, _i32_le, _f32_le
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations

_NCHANNELS_OFFSET = 370
_NSAMPLES_OFFSET = 864
_RATE_OFFSET = 376
_EVENTTABLEPOS_OFFSET = 886
_DATA_OFFSET = 900
_CH_SIZE = 75


def _read_str(b: bytes, off: int, n: int) -> str:
    # mne.io.utils.read_str: strip trailing NULs
    s = _ascii(b, off, n)
    k = s.find("\0")
    if k >= 0:
        s = s[:k]
    return s


def _find_channels(ch_names: List[str], ch_type: str) -> List[int]:
    out = List[int]()
    for i in range(len(ch_names)):
        if ch_names[i].upper().startswith(ch_type):
            out.append(i)
    return out


def _to_idx(sel, names: List[str], n: int) -> List[int]:
    out = List[int]()
    for x in sel:
        if isinstance(x, str):
            for i in range(n):
                if names[i] == x:
                    out.append(i)
        else:
            out.append(int(x))
    return out


def _parse_date(s: str, ddmm: bool) -> Optional[float]:
    # "MM/DD/YY HH:MM:SS"
    try:
        p = s.strip().split()
        d = p[0].split("/")
        t = p[1].split(":")
        a = int(d[0])
        c = int(d[1])
        yy = int(d[2])
        mo = c if ddmm else a
        da = a if ddmm else c
        y = 1900 + yy if yy >= 69 else 2000 + yy
        yy2 = y - 1 if mo <= 2 else y
        era = yy2 // 400
        yoe = yy2 - era * 400
        mp = (mo + 9) % 12
        doy = (153 * mp + 2) // 5 + da - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        days = era * 146097 + doe - 719468
        return float(days) * 86400.0 + float(int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2]))
    except:
        return None


def read_raw_cnt(input_fname, eog = (), misc = (), ecg = (), emg = (), *,
                 data_format: str = "auto", date_format: str = "mm/dd/yy",
                 recompute_n_samples = None, header: str = "auto",
                 preload = False, verbose = None) -> RawArray:
    fn = str(input_fname)
    b = _read_file(fn)
    fsize = len(b)
    if date_format != "mm/dd/yy" and date_format != "dd/mm/yy":
        raise ValueError(f"Invalid value for the 'date_format' parameter. Allowed values are 'mm/dd/yy' and 'dd/mm/yy', but got '{date_format}' instead.")
    if header != "auto" and header != "new" and header != "old":
        raise ValueError(f"Invalid value for the 'header' parameter. Allowed values are 'auto', 'new', and 'old', but got '{header}' instead.")
    if data_format != "auto" and data_format != "int16" and data_format != "int32":
        raise ValueError(f"Invalid value for the 'data_format' parameter. Allowed values are 'auto', 'int16', and 'int32', but got '{data_format}' instead.")
    sfreq = float(_uint_le(b, _RATE_OFFSET, 2))
    nch = _uint_le(b, _NCHANNELS_OFFSET, 2)
    n_samples = _i32_le(b, _NSAMPLES_OFFSET)
    samples_offset = _DATA_OFFSET + _CH_SIZE * nch
    n_bytes = 2
    event_offset = 0
    if fsize < 2000000000:
        event_offset = _i32_le(b, _EVENTTABLEPOS_OFFSET)
        if event_offset > fsize:
            if data_format == "auto":
                raise RuntimeError(f"Event table offset from header ({event_offset}) is larger than file size ({fsize}), cannot automatically compute data format, pass data_format='int16' or 'int32' explicitly")
            print(f"RuntimeWarning: Event table offset from header ({event_offset}) is larger than file size ({fsize}), recomputing event table offset.")
            nb = 2 if data_format == "int16" else 4
            event_offset = samples_offset + n_samples * nch * nb
        ndb = event_offset - samples_offset
        if data_format == "auto":
            why = ""
            bpc = ndb // nch
            if ndb % nch != 0:
                why = f"number of data bytes {ndb} is not evenly divisible by n_channels={nch}"
            elif n_samples == 0:
                why = "number of samples (according to header) is 0"
            else:
                n_bytes = bpc // n_samples
                if bpc % n_samples != 0 or (n_bytes != 2 and n_bytes != 4):
                    why = f"number of bytes per channel {bpc} is not evenly divisible by n_samples={n_samples} or does not result in 2 or 4 bytes per sample (n_bytes={n_bytes})"
            if why != "":
                raise RuntimeError(f"Could not automatically compute number of bytes per sample as the {why}.  set data_format manually.")
        else:
            n_bytes = 2 if data_format == "int16" else 4
        do_rec = False
        if recompute_n_samples is None:
            do_rec = n_samples <= 0
        else:
            do_rec = bool(recompute_n_samples)
        if do_rec:
            n_samples = ndb // (nch * n_bytes)
    else:
        if data_format == "auto":
            raise RuntimeError("Using `data_format='auto' for a CNT file larger than 2Gb is not supported, explicitly pass data_format as 'int16' or 'int32'")
        n_bytes = 2 if data_format == "int16" else 4
        event_offset = samples_offset + n_bytes * nch * n_samples
    # header fields
    patient_id_s = _read_str(b, 21, 20)
    patient_id = int(patient_id_s) if patient_id_s.isdigit() else 0
    pname = _read_str(b, 121, 20).split()
    last_name = pname[0] if len(pname) > 0 else ""
    first_name = pname[len(pname) - 1] if len(pname) > 0 else ""
    session_label = _read_str(b, 205, 20)
    session_date = _read_str(b, 225, 10) + " " + _read_str(b, 235, 12)
    meas_date = _parse_date(session_date, date_format == "dd/mm/yy")
    lowpass_toggle = _u8(b, 438) != 0
    highpass_toggle = _u8(b, 439) != 0
    lowcutoff = _f32_le(b, 869)
    highcutoff = _f32_le(b, 875)
    ch_off = _i32_le(b, 894)
    if ch_off > 1:
        ch_off = ch_off // n_bytes
    else:
        ch_off = 1
    names = List[str]()
    cals = List[float]()
    baselines = List[int]()
    bads = List[str]()
    xs = List[float]()
    ys = List[float]()
    for c in range(nch):
        o = _DATA_OFFSET + _CH_SIZE * c
        nm = _read_str(b, o, 10)
        names.append(nm)
        if header == "new" or header == "auto":
            if _u8(b, o + 14) != 0:
                bads.append(nm)
        if header == "old" or header == "auto":
            if _u8(b, o + 4) != 0:
                bads.append(nm)
        xs.append(_f32_le(b, o + 19))
        ys.append(-_f32_le(b, o + 23))
        baselines.append(_i16_le(b, o + 47))
        sens = _f32_le(b, o + 59)
        cal = _f32_le(b, o + 71)
        cals.append(cal * sens * 1e-6 / 204.8)
    eogl = List[int]()
    if isinstance(eog, str):
        if eog == "header":
            for k in range(2):
                v = _i16_le(b, 402 + 2 * k)
                if v >= 0:
                    eogl.append(v)
        elif eog == "auto":
            eogl = _find_channels(names, "EOG")
    else:
        eogl = _to_idx(eog, names, nch)
    ecgl = _find_channels(names, "ECG") if (isinstance(ecg, str) and ecg == "auto") else _to_idx(ecg, names, nch) if not isinstance(ecg, str) else List[int]()
    emgl = _find_channels(names, "EMG") if (isinstance(emg, str) and emg == "auto") else _to_idx(emg, names, nch) if not isinstance(emg, str) else List[int]()
    miscl = _to_idx(misc, names, nch)
    types = List[str]()
    for i in range(nch):
        t = "eeg"
        if i in eogl:
            t = "eog"
        elif i in ecgl:
            t = "ecg"
        elif i in emgl:
            t = "emg"
        elif i in miscl:
            t = "misc"
        types.append(t)
    info = Info(names, sfreq, types)
    if lowpass_toggle and highcutoff > 0:
        info.lowpass = highcutoff
    if highpass_toggle and lowcutoff > 0:
        info.highpass = lowcutoff
    info.meas_date = meas_date
    info.description = None if session_label == "" else session_label
    # topo -> sphere
    eegs = [i for i in range(nch) if types[i] == "eeg"]
    sq = 0.0
    for i in eegs:
        sq = max(sq, np.sqrt(xs[i] * xs[i] + ys[i] * ys[i]))
    for i in range(nch):
        xs[i] /= sq
        ys[i] /= sq
    mx = 0.0
    my = 0.0
    for i in eegs:
        mx += xs[i]
        my += ys[i]
    mx /= float(len(eegs))
    my /= float(len(eegs))
    for i in range(nch):
        xs[i] = (xs[i] + 0.5 - mx) * 2.0 - 1.0
        ys[i] = (ys[i] + 0.5 - my) * 2.0 - 1.0
        r = min(max(np.sqrt(xs[i] * xs[i] + ys[i] * ys[i]), 0.0), 1.0)
        info.locs[i][0] = xs[i]
        info.locs[i][1] = ys[i]
        info.locs[i][2] = np.sin(np.arccos(r))
    for i in range(nch):
        info.cals[i] = cals[i]
        if types[i] != "eeg":
            info.coil_types[i] = 0
    for bd in bads:
        if bd not in info.bads:
            info.bads.append(bd)
    # data
    data = np.zeros((nch, n_samples))
    nchunks = (n_samples + ch_off - 1) // ch_off
    for ci in range(nchunks):
        for c in range(nch):
            for q in range(ch_off):
                s = ci * ch_off + q
                if s >= n_samples:
                    continue
                off = samples_offset + ((ci * nch + c) * ch_off + q) * n_bytes
                v = 0
                if n_bytes == 4:
                    v = _i32_le(b, off)
                else:
                    v = _i16_le(b, off)
                data[c, s] = (float(v) - float(baselines[c])) * cals[c]
    raw = RawArray(data, info)
    raw._raw_kind = "RawCNT"
    # annotations
    et = _u8(b, event_offset)
    total = _i32_le(b, event_offset + 1)
    esz = 8 if et == 1 else 19
    if et < 1 or et > 3:
        raise ValueError(f"unknown CNT even type {et}")
    nev = total // esz
    base = event_offset + 9
    onset = List[float]()
    desc = List[str]()
    acc = List[str]()
    for k in range(nev):
        o = base + k * esz
        stim = _uint_le(b, o, 2)
        kb = _u8(b, o + 2)
        kp = _u8(b, o + 3)
        offv = _i32_le(b, o + 4)
        if et == 3:
            offv *= n_bytes * nch
        t = offv - _DATA_OFFSET - _CH_SIZE * nch
        t = t // (nch * n_bytes)
        t = t - 1
        if t < 0:
            t = 0
        onset.append(float(t) / sfreq)
        if kp == 0xD0:
            acc.append("good")
        elif kp == 0xC0:
            acc.append("bad")
        else:
            acc.append("NA")
        kpad = kp & 0x0F
        if kpad != 0:
            desc.append(f"KeyPad Response {kpad}")
        elif kb != 0:
            desc.append(f"Keyboard Response {kb}")
        else:
            desc.append(str(stim))
    if nev > 0:
        dur = [0.0 for _ in range(nev)]
        fo = List[float]()
        fd = List[float]()
        fs = List[str]()
        if "bad" in acc:
            gb_on = List[float]()
            gb_kind = List[str]()
            for k in range(nev):
                if acc[k] == "bad" or acc[k] == "good":
                    gb_on.append(onset[k])
                    gb_kind.append(acc[k])
            first_bad = gb_kind.index("bad")
            dl = List[float]()
            i = first_bad
            while i < len(gb_kind):
                dl.append(gb_on[i + 1] - gb_on[i])
                i += 2
            di = 0
            for k in range(nev):
                if acc[k] == "bad":
                    dur[k] = dl[di]
                    desc[k] = "BAD_" + desc[k]
                    di += 1
            for k in range(nev):
                if acc[k] != "good":
                    fo.append(onset[k])
                    fd.append(dur[k])
                    fs.append(desc[k])
        else:
            fo = onset
            fd = dur
            fs = desc
        raw.set_annotations(Annotations(fo, fd, fs), False)
    return raw
