# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# EGI simple binary (.raw) reader (mne/io/egi/egi.py). MFF is not supported.
# Only continuous (unsegmented) files, events become annotations.

import numpy as np
from ._bytes import _read_file, _i16_be, _i32_be, _i32_le, _arr_be, _ascii
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations


def _days_from_civil(y: int, m: int, d: int) -> int:
    yy = y - 1 if m <= 2 else y
    era = (yy if yy >= 0 else yy - 399) // 400
    yoe = yy - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def read_raw_egi(input_fname, eog = None, misc = None, include = None, exclude = None,
                 preload = False, channel_naming: str = "E%d", *,
                 events_as_annotations: bool = True, event_key = None, verbose = None) -> RawArray:
    fn = str(input_fname)
    if fn.rstrip("/").endswith(".mff"):
        raise NotImplementedError("Typhon mne: EGI MFF files are not supported")
    if not events_as_annotations:
        raise NotImplementedError("Typhon mne: read_raw_egi supports events_as_annotations=True only")
    b = _read_file(fn)
    le = _i32_le(b, 0)
    if not (le > (6 & ~(le & 6))):
        raise ValueError("Watchout. This does not seem to be a simple binary EGI file.")
    version = _i32_be(b, 0)
    year = _i16_be(b, 4)
    month = _i16_be(b, 6)
    day = _i16_be(b, 8)
    hour = _i16_be(b, 10)
    minute = _i16_be(b, 12)
    second = _i16_be(b, 14)
    srate = _i16_be(b, 20)
    nch = _i16_be(b, 22)
    bits = _i16_be(b, 26)
    vrange = _i16_be(b, 28)
    unsegmented = (version & 1) == 0
    precision = version & 6
    if precision == 0:
        raise RuntimeError("Floating point precision is undefined.")
    if not unsegmented:
        raise NotImplementedError("Only continuous files are supported")
    nsamp = _i32_be(b, 30)
    nev = _i16_be(b, 34)
    codes = List[str]()
    for e in range(nev):
        codes.append(_ascii(b, 36 + 4 * e, 4))
    off = 36 + nev * 4
    nrows = nch + nev
    code = "i2"
    if precision == 4:
        code = "f4"
    elif precision == 6:
        code = "f8"
    cal = 1e-6
    if vrange != 0 and bits != 0:
        cal = float(vrange) / (2.0 ** bits)
    flat = _arr_be(b, off, nsamp * nrows, code)
    mm = flat.reshape((nsamp, nrows))
    names = List[str]()
    for i in range(nch):
        names.append(channel_naming.replace("%d", str(i + 1)))
    for c in codes:
        names.append(c)
    eogl = [str(x) for x in eog] if eog is not None else List[str]()
    miscl = [str(x) for x in misc] if misc is not None else List[str]()
    types = List[str]()
    for k in range(len(names)):
        t = "eeg"
        if names[k] in eogl or str(k) in eogl:
            t = "eog"
        elif names[k] in miscl:
            t = "misc"
        if k >= nch:
            t = "stim"
        types.append(t)
    info = Info(names, float(srate), types)
    data = np.zeros((len(names), nsamp))
    for k in range(len(names)):
        c = cal if k < nch else 1.0
        info.cals[k] = c
        if types[k] == "stim":
            info.units[k] = -1
            info.coil_types[k] = 0
        elif types[k] != "eeg":
            info.coil_types[k] = 0
        data[k, :] = mm[:, k] * c
    info.meas_date = float(_days_from_civil(year, month, day)) * 86400.0 + float(hour * 3600 + minute * 60 + second)
    raw = RawArray(data, info)
    raw._raw_kind = "RawEGI"
    # event inclusion (mne.io.egi.events._triage_include_exclude)
    inc = List[str]()
    if include is not None:
        for x in include:
            if str(x) not in codes:
                raise ValueError(f"Could not find event channel named include[0]={str(x)!r}")
            inc.append(str(x))
    else:
        exc = List[str]()
        if exclude is not None:
            for x in exclude:
                if str(x) not in codes:
                    raise ValueError(f"Could not find event channel named exclude[0]={str(x)!r}")
                exc.append(str(x))
        else:
            for d in ("sync", "TREV"):
                if d in codes:
                    exc.append(d)
            for e in range(nev):
                s = 0.0
                for t in range(nsamp):
                    s += mm[t, nch + e]
                if s < 1 and codes[e] != "":
                    exc.append(codes[e])
        for c in codes:
            if c not in exc:
                inc.append(c)
    on = List[float]()
    du = List[float]()
    de = List[str]()
    for e in range(nev):
        if codes[e] not in inc:
            continue
        for t in range(nsamp):
            if mm[t, nch + e] != 0:
                on.append(float(t) / float(srate))
                du.append(0.0)
                de.append(codes[e])
    if len(on) > 0:
        raw.set_annotations(Annotations(on, du, de), False)
    return raw
