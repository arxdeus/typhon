# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# EDF/EDF+/BDF reader (mne/io/edf/edf.py): header parsing, channel type
# inference, calibration, stim channel handling and EDF+ TAL annotations.
# Channels with differing sampling rates (other than the TAL channel) are
# not supported.

import numpy as np
from ._bytes import _read_file, _ascii, _arr_le, _u8, _uint_le, _sint
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations

_CH_TYPE_MAPPING = ("EEG", "SEEG", "ECOG", "DBS", "EOG", "ECG", "EMG", "BIO", "RESP",
                    "TEMP", "MISC", "SAO2", "STIM")


def _map_type(t: str) -> str:
    if t == "TEMP":
        return "temperature"
    if t == "SAO2":
        return "bio"
    return t.lower()


def _edf_str(b: bytes, off: int, n: int) -> str:
    s = _ascii(b, off, n)
    z = s.find("\0")
    if z >= 0:
        s = s[:z]
    return s


def _prefilter_value(s: str, key: str) -> str:
    # re.findall(rf"{t}:\s*([a-zA-Z0-9,.]+)(Hz)?", filt), last non-empty match
    value = ""
    i = 0
    while True:
        j = s.find(key + ":", i)
        if j < 0:
            break
        k = j + len(key) + 1
        while k < len(s) and s[k] in " \t\n\r\f\v":
            k += 1
        st = k
        while k < len(s) and (s[k].isalnum() or s[k] == "," or s[k] == "."):
            k += 1
        if k > st:
            value = s[st:k].replace("Hz", "").replace(",", ".")
        i = j + 1
    return value


def _prefilter_float(s: str) -> float:
    if s == "DC":
        return 0.0
    if s.replace(".", "", 1).isdigit():
        return float(s)
    return np.nan


def _parse_tal(tal: List[int]) -> Tuple[List[float], List[float], List[str]]:
    """re.findall('([+-]\\d+\\.?\\d*)(\\x15(\\d+\\.?\\d*))?(\\x14.*?)\\x14\\x00') + mne logic."""
    n = len(tal)
    on = List[float]()
    du = List[float]()
    de = List[str]()
    offset = 0.0
    k = 0
    i = 0

    while i < n:
        c = tal[i]
        if c != 43 and c != 45:
            i += 1
            continue
        j = i + 1
        d0 = j
        while j < n and 48 <= tal[j] <= 57:
            j += 1
        if j == d0:
            i += 1
            continue
        if j < n and tal[j] == 46:
            j += 1
            while j < n and 48 <= tal[j] <= 57:
                j += 1
        onset_s = "".join([chr(tal[q]) for q in range(i, j)])
        dur_s = ""
        if j < n and tal[j] == 21:
            q = j + 1
            d1 = q
            while q < n and 48 <= tal[q] <= 57:
                q += 1
            if q > d1:
                if q < n and tal[q] == 46:
                    q += 1
                    while q < n and 48 <= tal[q] <= 57:
                        q += 1
                dur_s = "".join([chr(tal[r]) for r in range(d1, q)])
                j = q
        if j >= n or tal[j] != 20:
            i += 1
            continue
        # lazily find "\x14\x00" after j (pattern group 4 starts with \x14)
        q = j + 1
        end = -1
        while q + 1 < n:
            if tal[q] == 20 and tal[q + 1] == 0:
                end = q
                break
            if tal[q] == 10:
                break
            q += 1
        if end < 0:
            i += 1
            continue
        body = "".join([chr(tal[r]) for r in range(j, end)])
        onset = float(onset_s) + offset
        dur = float(dur_s) if dur_s != "" else 0.0
        descs = body.split("\x14")[1:]
        for d in descs:
            if d != "":
                on.append(onset)
                du.append(dur)
                de.append(d)
            elif k == 0:
                offset = -onset
        k += 1
        i = end + 2
    return on, du, de


def _read_raw_edf_bdf(fname: str, is_bdf: bool, eog, misc, stim_channel, exclude,
                      infer_types: bool, include, preload) -> RawArray:
    b = _read_file(fname)
    header_nbytes = int(_edf_str(b, 184, 8).strip())
    n_records = int(_edf_str(b, 236, 8).strip())
    rec_len = float(_edf_str(b, 244, 8).strip())
    if rec_len == 0:
        rec_len = 1.0
    nchan = int(_edf_str(b, 252, 4).strip())
    p = 256
    labels = [_ascii(b, p + 16 * i, 16).strip() for i in range(nchan)]
    p += 16 * nchan
    p += 80 * nchan  # transducer
    units_s = [_ascii(b, p + 8 * i, 8).strip() for i in range(nchan)]
    p += 8 * nchan
    pmin = [float(_edf_str(b, p + 8 * i, 8).replace(",", ".")) for i in range(nchan)]
    p += 8 * nchan
    pmax = [float(_edf_str(b, p + 8 * i, 8).replace(",", ".")) for i in range(nchan)]
    p += 8 * nchan
    dmin = [float(_edf_str(b, p + 8 * i, 8).replace(",", ".")) for i in range(nchan)]
    p += 8 * nchan
    dmax = [float(_edf_str(b, p + 8 * i, 8).replace(",", ".")) for i in range(nchan)]
    p += 8 * nchan
    prefilt = [_edf_str(b, p + 80 * i, 80).strip() for i in range(nchan)]
    p += 80 * nchan
    n_samps = [int(_edf_str(b, p + 8 * i, 8).strip()) for i in range(nchan)]
    ch_types = List[str]()
    ch_names = List[str]()
    for lab in labels:
        t = "EEG"
        nm = lab
        if infer_types:
            parts = lab.split(" ")
            if len(parts) > 1 and parts[0].upper() in _CH_TYPE_MAPPING:
                t = parts[0].upper()
                nm = " ".join(parts[1:])
        ch_types.append(t)
        ch_names.append(nm)
    tal_idx = [i for i in range(nchan) if ch_names[i] in ("EDF Annotations", "BDF Annotations")]
    excl = List[int]()
    if not isinstance(include, NoneType):
        inc = [str(x) for x in include]
        excl = [i for i in range(nchan) if ch_names[i] not in inc]
    else:
        ex = [str(x) for x in exclude]
        excl = [i for i in range(nchan) if ch_names[i] in ex]
    sel = [i for i in range(nchan) if i not in excl and i not in tal_idx]
    gains = List[float]()
    for i in sel:
        u = units_s[i]
        if u in ("\u03bcV", "\u00b5V", "uV") or (len(u) == 3 and ord(u[0]) == 0x83):
            gains.append(1e-6)
        elif u == "mV":
            gains.append(1e-3)
        else:
            gains.append(1.0)
    # _unique_channel_names
    names = [ch_names[i] for i in sel]
    types = [ch_types[i] for i in sel]
    for i in range(len(names)):
        cnt = names.count(names[i])
        if cnt > 1:
            dups = [k for k in range(len(names)) if names[k] == names[i]]
            for q in range(len(dups)):
                names[dups[q]] = f"{names[dups[q]]}-{q}"
    # stim channel(s)
    stim_idx = List[int]()
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
    for v in sorted(set(valid)):
        if v in low:
            stim_idx.append(low.index(v))
    eogl = List[str]() if isinstance(eog, NoneType) else [str(x) for x in eog]
    miscl = List[str]() if isinstance(misc, NoneType) else [str(x) for x in misc]
    out_types = List[str]()
    for idx in range(len(names)):
        t = types[idx]
        mt = _map_type(t)
        if names[idx] in eogl:
            mt = "eog"
        elif names[idx] in miscl:
            mt = "misc"
        elif idx in stim_idx:
            mt = "stim"
            gains[idx] = 1.0
        out_types.append(mt)
    ns_sel = [n_samps[i] for i in sel]
    not_stim = [x for x in range(len(sel)) if x not in stim_idx]
    if len(not_stim) == 0:
        not_stim = list(range(len(sel)))
    max_samp = max([ns_sel[x] for x in not_stim])
    sfreq = max_samp / rec_len
    for i in sel:
        if n_samps[i] != max_samp:
            raise NotImplementedError("Typhon mne: EDF channels with different sampling rates are not supported")
    info = Info(names, sfreq, out_types)
    for idx in range(len(names)):
        if out_types[idx] == "stim":
            info.units[idx] = -1
            info.coil_types[idx] = 0
        elif out_types[idx] not in ("eeg", "ecog", "seeg", "dbs"):
            info.coil_types[idx] = 0
            info.units[idx] = 107
    # prefilters (exclude stim)
    filt = [k for k in range(len(sel)) if k not in stim_idx]
    for key in ("HP", "LP"):
        vals = [_prefilter_value(prefilt[sel[k]], key) for k in filt]
        if len(vals) == 0:
            continue
        v = 0.0
        if len(set(vals)) > 1:
            fv = [_prefilter_float(x) for x in vals]
            fv2 = [x for x in fv if not np.isnan(x)]
            if len(fv2) == 0:
                v = np.nan
            else:
                v = max(fv2) if key == "HP" else min(fv2)
        else:
            v = _prefilter_float(vals[0]) if vals[0] != "" else np.nan
        if not np.isnan(v) and v != 0:
            if key == "HP":
                info.highpass = v
            else:
                info.lowpass = v
    if np.isnan(info.lowpass) or info.lowpass <= 0:
        info.lowpass = sfreq / 2.0
    if info.highpass > info.lowpass:
        info.highpass = 0.0
        info.lowpass = sfreq / 2.0
    bpd = 3 if is_bdf else 2
    rec_total = sum(n_samps)
    n_data = len(b) - header_nbytes
    read_records = (n_data // bpd) // rec_total
    if read_records != n_records:
        n_records = read_records
    nsamp = n_records * max_samp
    data = np.zeros((len(sel), nsamp))
    offs = [0]
    for s in n_samps:
        offs.append(offs[-1] + s)
    tal = List[int]()
    cal = List[float]()
    off = List[float]()
    for k in range(len(sel)):
        i = sel[k]
        dr = dmax[i] - dmin[i]
        if not np.isfinite(dr) or dr == 0:
            dr = 1.0
        pr = pmax[i] - pmin[i]
        if pr == 0:
            pr = 1.0
        c = pr / dr
        o = pmin[i] - dmin[i] * c
        if is_bdf and k in stim_idx:
            c = 1.0
            o = 0.0
        cal.append(c)
        off.append(o)
    code = "i3" if is_bdf else "i2"
    for r in range(n_records):
        base = header_nbytes + r * rec_total * bpd
        for k in range(len(sel)):
            i = sel[k]
            vals = _arr_le(b, base + offs[i] * bpd, n_samps[i], code)
            row = (vals * cal[k] + off[k]) * gains[k]
            if k in stim_idx:
                for q in range(row.shape[0]):
                    row[q] = float(int(row[q]) & (2 ** 17 - 1))
            data[k, r * max_samp:(r + 1) * max_samp] = row
        for i in tal_idx:
            st = base + offs[i] * bpd
            if is_bdf:
                for q in range(n_samps[i] * 3):
                    tal.append(_u8(b, st + q))
            else:
                for q in range(n_samps[i] * 2):
                    tal.append(_u8(b, st + q))
    raw = RawArray(data, info)
    raw._raw_kind = "RawBDF" if is_bdf else "RawEDF"
    if len(tal_idx) > 0:
        on, du, de = _parse_tal(tal)
        order = sorted(range(len(on)), key=lambda q: on[q])
        raw.set_annotations(Annotations([on[q] for q in order], [du[q] for q in order],
                                        [de[q] for q in order]), False)
    return raw


def read_raw_edf(input_fname, eog = None, misc = None, stim_channel = "auto", exclude = (),
                 infer_types: bool = False, include = None, preload = False, units = None,
                 encoding: str = "utf8", exclude_after_unique: bool = False, *, verbose = None):
    fn = str(input_fname)
    if not fn.lower().endswith(".edf"):
        raise NotImplementedError(f"Only EDF files are supported, got {fn.split('.')[-1]}.")
    return _read_raw_edf_bdf(fn, False, eog, misc, stim_channel, exclude, infer_types,
                             include, preload)


def read_raw_bdf(input_fname, eog = None, misc = None, stim_channel = "auto", exclude = (),
                 infer_types: bool = False, include = None, preload = False, units = None,
                 encoding: str = "utf8", exclude_after_unique: bool = False, *, verbose = None):
    fn = str(input_fname)
    if not fn.lower().endswith(".bdf"):
        raise NotImplementedError(f"Only BDF files are supported, got {fn.split('.')[-1]}.")
    return _read_raw_edf_bdf(fn, True, eog, misc, stim_channel, exclude, infer_types,
                             include, preload)
