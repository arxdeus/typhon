# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# Nihon Kohden .EEG reader (mne/io/nihon/nihon.py) with .21E channel names,
# .PNT meas_date and .LOG annotations. One control block only, as in mne.

import numpy as np
from ._bytes import _read_file, _ascii, _u8, _uint_le
from ._fmtutil import _utc_secs
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations

_VALID = ["EEG-1100A V01.00", "EEG-1100B V01.00", "EEG-1100C V01.00", "QI-403A   V01.00",
          "QI-403A   V02.00", "EEG-2100  V01.00", "EEG-2100  V02.00", "DAE-2100D V01.30",
          "DAE-2100D V02.00", "EEG-1200A V01.00"]


def _labels() -> List[str]:
    out = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4",
           "T5", "T6", "FZ", "CZ", "PZ", "E", "PG1", "PG2", "A1", "A2", "T1", "T2"]
    for i in range(1, 12):
        out.append(f"X{i}")
    for i in range(1, 6):
        out.append(f"NA{i}")
    for i in range(1, 33):
        out.append(f"DC{i:02d}")
    out.append("BN1")
    out.append("BN2")
    out.append("Mark1")
    out.append("Mark2")
    for i in range(6, 28):
        out.append(f"NA{i}")
    out.append("X12/BP1")
    out.append("X13/BP2")
    out.append("X14/BP3")
    out.append("X15/BP4")
    for i in range(16, 166):
        out.append(f"X{i}")
    out.append("NA28")
    out.append("Z")
    return out


def _cstr(b: bytes, off: int, n: int) -> str:
    s = _ascii(b, off, n)
    k = s.find("\0")
    if k >= 0:
        s = s[:k]
    return s


def _stem(fn: str) -> str:
    j = fn.rfind(".")
    return fn[:j] if j >= 0 else fn


def _try_read(stem: str, exts: List[str]) -> Optional[bytes]:
    for e in exts:
        try:
            return _read_file(stem + e)
        except:
            pass
    return None


def _chan_labels(stem: str) -> List[str]:
    lab = _labels()
    eb = _try_read(stem, [".21E", ".21e"])
    if eb is None:
        print("RuntimeWarning: Could not find .21E containing channel definitions; Default channel names are chosen.")
        return lab
    txt = _ascii(eb, 0, len(eb))
    keep = False
    for line in txt.split("\n"):
        if line.startswith("["):
            keep = ("ELECTRODE" in line) or ("REFERENCE" in line)
        elif keep and "=" in line:
            k = line.find("=")
            idx = int(line[:k])
            while idx >= len(lab):
                lab.append("UNK")
            lab[idx] = line[k + 1:].strip()
    return lab


def _map_type(nm: str) -> str:
    if "Mark" in nm:
        return "stim"
    for k in ("DC", "NA", "Z", "$"):
        if k in nm:
            return "misc"
    if "X" in nm:
        return "bio"
    return "eeg"


def read_raw_nihon(fname, preload = False, *, encoding: str = "utf-8", verbose = None) -> RawArray:
    fn = str(fname)
    stem = _stem(fn)
    labels = _chan_labels(stem)
    b = _read_file(fn)
    version = _cstr(b, 0, 16)
    if version not in _VALID:
        raise ValueError(f"Not a valid Nihon Kohden EEG file ({version})")
    if _cstr(b, 0x81, 16) not in _VALID:
        raise ValueError(f"Not a valid Nihon Kohden EEG file (control block {version})")
    if _u8(b, 0x17FE) != 1:
        raise ValueError("Not a valid Nihon Kohden EEG file (waveform block)")
    n_ctl = _u8(b, 0x91)
    if n_ctl != 1:
        raise NotImplementedError("I dont know how to read more than one control block for this type of file :(")
    ca = _uint_le(b, 0x92, 4)
    n_db = _u8(b, ca + 17)
    d_addr = List[int]()
    d_nch = List[int]()
    d_ns = List[int]()
    d_sf = List[int]()
    chans = List[str]()
    for i in range(n_db):
        da = _uint_le(b, ca + i * 20 + 18, 4)
        nch = _u8(b, da + 0x26)
        names = List[str]()
        for k in range(nch):
            names.append(labels[_u8(b, da + 0x27 + k * 10)])
        dur = _uint_le(b, da + 0x1C, 4)
        sf = _uint_le(b, da + 0x1A, 2) & 0x3FFF
        if i == 0:
            chans = names
        else:
            if names != chans:
                raise ValueError("Channel names in datablocks do not match")
            if sf != d_sf[0]:
                raise ValueError("Sample frequency in datablocks do not match")
        d_addr.append(da)
        d_nch.append(nch)
        d_sf.append(sf)
        d_ns.append(dur * sf // 10)
    n = len(chans)
    sfreq = float(d_sf[0])
    types = List[str]()
    for c in chans:
        types.append(_map_type(c))
    info = Info(chans, sfreq, types)
    up = [x.upper() for x in labels]
    cal = List[float]()
    off = List[float]()
    gain = List[float]()
    for i in range(n):
        unit = 1e-3
        pmin = -12002.9
        pmax = 12002.56
        nm = chans[i].upper()
        if nm in up:
            idx = up.index(nm)
            if (idx < 42 or idx > 73) and idx != 76 and idx != 77:
                unit = 1e-6
                pmin = -3200.0
                pmax = 3199.902
        rng = pmax - pmin
        c = rng / 65535.0
        cal.append(c)
        off.append(pmin - (-32768.0 * c))
        gain.append(unit)
        info.ranges[i] = rng
        info.cals[i] = 1.0 / rng
    # PNT meas_date
    pb = _try_read(stem, [".PNT", ".pnt"])
    if pb is None:
        print("RuntimeWarning: No PNT file exists. Metadata will be blank")
    else:
        pv = _cstr(pb, 0, 16)
        if pv not in _VALID:
            raise ValueError(f"Not a valid Nihon Kohden PNT file ({pv})")
        ms = _ascii(pb, 0x40, 14)
        info.meas_date = _utc_secs(int(ms[0:4]), int(ms[4:6]), int(ms[6:8]), int(ms[8:10]), int(ms[10:12]), int(ms[12:14]))
    total = 0
    for x in d_ns:
        total += x
    data = np.zeros((n, total))
    s0 = 0
    for bi in range(n_db):
        nc1 = d_nch[bi] + 1
        ds = d_addr[bi] + 0x27 + d_nch[bi] * 10
        for s in range(d_ns[bi]):
            for c in range(n):
                u = _uint_le(b, ds + (s * nc1 + c) * 2, 2)
                v = float(u - 32768)
                data[c, s0 + s] = (v * cal[c] + off[c]) * gain[c]
        s0 += d_ns[bi]
    raw = RawArray(data, info)
    raw._raw_kind = "RawNihon"
    # annotations from the LOG file
    on = List[float]()
    du = List[float]()
    ds2 = List[str]()
    lb = _try_read(stem, [".LOG", ".log"])
    if lb is None:
        print("RuntimeWarning: No LOG file exists. Annotations will not be read")
    else:
        lv = _cstr(lb, 0, 16)
        if lv not in _VALID:
            raise ValueError(f"Not a valid Nihon Kohden LOG file ({lv})")
        nlb = _u8(lb, 0x91)
        may_sub = nlb <= 21
        for tb in range(nlb):
            logs = _log_block(lb, tb, lv)
            subs = List[str]()
            if may_sub:
                subs = _log_block(lb, tb + 22, lv)
            for li in range(len(logs)):
                lg = logs[li]
                desc = lg[:20]
                onset = float(int(lg[20:22]) * 3600 + int(lg[22:24]) * 60 + int(lg[24:26]))
                if may_sub and len(subs) == len(logs):
                    sl = subs[li]
                    desc = desc + sl[:20]
                    onset += float(int(sl[24:30])) / 1e6
                k = desc.find("\0")
                if k >= 0:
                    desc = desc[:k]
                on.append(onset)
                du.append(0.0)
                ds2.append(desc)
    if n_db > 1:
        cur = 0
        for i in range(n_db - 1):
            cur += d_ns[i]
            tp = (float(cur) - 0.5) / float(d_sf[i])
            on.append(tp)
            du.append(0.0)
            ds2.append("BAD boundary")
            on.append(tp)
            du.append(0.0)
            ds2.append("EDGE boundary")
    raw.set_annotations(Annotations(on, du, ds2, info.meas_date), False)
    return raw


def _log_block(lb: bytes, tb: int, version: str) -> List[str]:
    out = List[str]()
    pos = 0x92 + tb * 20
    if pos + 4 > len(lb):
        return out
    addr = _uint_le(lb, pos, 4)
    if addr == 0:
        return out
    if _cstr(lb, addr + 1, 16) != version:
        print("RuntimeWarning: Event log version mismatch")
    nl = _u8(lb, addr + 0x12)
    for i in range(nl):
        out.append(_ascii(lb, addr + 0x14 + i * 45, 45))
    return out
