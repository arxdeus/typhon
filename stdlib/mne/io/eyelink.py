# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# EyeLink .asc reader (mne/io/eyelink/eyelink.py + _utils.py). Narrower:
# find_overlaps is not supported, Annotations do not carry ch_names.

import numpy as np
from math import asin, ceil, isnan
from ._bytes import _read_text
from ._fmtutil import _utc_secs
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations

_KEYS = ["START", "END", "SAMPLES", "EVENTS", "ESACC", "EBLINK", "EFIX", "MSG", "INPUT", "BUTTON", "PUPIL"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class _Block:
    samples: List[List[str]]
    start_ts: float
    end_ts: float
    rec: List[str]
    pupil_seen: bool
    efix: List[List[str]]
    esacc: List[List[str]]
    eblink: List[List[str]]
    msgs: List[List[str]]
    buttons: List[List[str]]

    def __init__(self):
        self.samples = List[List[str]]()
        self.start_ts = 0.0
        self.end_ts = 0.0
        self.rec = List[str]()
        self.pupil_seen = False
        self.efix = List[List[str]]()
        self.esacc = List[List[str]]()
        self.eblink = List[List[str]]()
        self.msgs = List[List[str]]()
        self.buttons = List[List[str]]()


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False


def _missing(s: str) -> bool:
    return s == "." or s == "MISSING_DATA"


def _fl(s: str, nan_zero: bool) -> float:
    if _missing(s):
        return 0.0 if nan_zero else np.nan
    return float(s)


def _date(txt: str) -> Optional[float]:
    for line in txt.split("\n"):
        if line.startswith("** DATE:"):
            p = line.replace("** DATE:", "").strip().split()
            # "Wed Jan 01 12:34:56 2020"
            try:
                mo = _MONTHS.index(p[1]) + 1
                hms = p[3].split(":")
                return _utc_secs(int(p[4]), mo, int(p[2]), int(hms[0]), int(hms[1]), int(hms[2]))
            except:
                return None
    return None


def _cols(rec: List[str]) -> Tuple[List[str], bool, bool, str]:
    both = ("LEFT" in rec) and ("RIGHT" in rec)
    eye = "both" if both else rec[1].lower()
    ch = List[str]()
    if both:
        ch = ["xpos_left", "ypos_left", "pupil_left", "xpos_right", "ypos_right", "pupil_right"]
    elif eye == "left":
        ch = ["xpos_left", "ypos_left", "pupil_left"]
    else:
        ch = ["xpos_right", "ypos_right", "pupil_right"]
    if "VEL" in rec:
        if both:
            ch.extend(["xvel_left", "yvel_left", "xvel_right", "yvel_right"])
        elif eye == "left":
            ch.extend(["xvel_left", "yvel_left"])
        else:
            ch.extend(["xvel_right", "yvel_right"])
    if "RES" in rec:
        ch.extend(["xres", "yres"])
    if "INPUT" in rec:
        ch.append("DIN")
    if "HTARGET" in rec:
        ch.extend(["x_head", "y_head", "distance"])
    return (ch, both, True, eye)


def read_raw_eyelink(fname, *, create_annotations = True, apply_offsets: bool = False,
                     find_overlaps: bool = False, overlap_threshold: float = 0.05,
                     verbose = None) -> RawArray:
    if find_overlaps:
        raise NotImplementedError("find_overlaps is not supported in Typhon mne")
    txt = _read_text(str(fname))
    meas = _date(txt)
    blocks = List[_Block]()
    cur = _Block()
    inrec = False
    for line in txt.split("\n"):
        if line.startswith("START"):
            inrec = True
            cur = _Block()
        if not inrec:
            continue
        tok = line.split()
        if len(tok) == 0:
            continue
        if tok[0][0].isdigit():
            cur.samples.append(tok)
        elif tok[0] in _KEYS:
            if "!V" in line or "!MODE" in line or ";" in line:
                continue
            k = tok[0]
            info = tok[1:]
            if k == "START":
                cur.start_ts = float(info[0])
            elif k == "SAMPLES":
                if len(cur.rec) == 0:
                    cur.rec = info
            elif k == "PUPIL":
                cur.pupil_seen = True
            elif k == "EFIX":
                cur.efix.append(info)
            elif k == "ESACC":
                cur.esacc.append(info)
            elif k == "EBLINK":
                cur.eblink.append(info)
            elif k == "MSG":
                cur.msgs.append(info)
            elif k == "BUTTON":
                cur.buttons.append(info)
            elif k == "END":
                cur.end_ts = float(info[0])
                blocks.append(cur)
                inrec = False
    if len(blocks) == 0:
        raise ValueError(f"Couldn't find any samples in {fname}")
    for b in blocks:
        if len(b.samples) == 0:
            raise ValueError("No samples in a recording block")
        if len(b.rec) == 0:
            raise ValueError("No SAMPLES line in a recording block")
    unit = blocks[0].rec[0]
    sfreq = float(blocks[0].rec[blocks[0].rec.index("RATE") + 1])
    first_ts = blocks[0].start_ts
    # channel union
    ch_all = List[str]()
    for b in blocks:
        cn = _cols(b.rec)[0]
        for c in cn:
            if c not in ch_all:
                ch_all.append(c)
    n_ch = len(ch_all)
    t_all = List[float]()
    rows = List[List[float]]()
    n_events_rows = 0
    for b in blocks:
        cn = _cols(b.rec)[0]
        # drop status columns (first valid value non-numeric of len 3/5/13/17)
        ncol = 0
        for r in b.samples:
            if len(r) > ncol:
                ncol = len(r)
        keep = List[int]()
        for ci in range(ncol):
            if ci < 3:
                keep.append(ci)
                continue
            fv = ""
            found = False
            for r in b.samples:
                if ci < len(r):
                    fv = r[ci]
                    found = True
                    break
            if not found:
                continue
            if _is_num(fv) or _missing(fv):
                keep.append(ci)
            else:
                ln = len(fv)
                if ln == 3 or ln == 5 or ln == 13 or ln == 17:
                    pass
                else:
                    print("RuntimeWarning: Unexpected non-numeric value in status column.")
                    keep.append(ci)
        if len(keep) - 1 != len(cn):
            raise ValueError(f"Expected the samples data in this file to have {len(cn) + 1} columns of data, but got {len(keep)}.")
        for r in b.samples:
            row = [np.nan for _ in range(n_ch)]
            for k in range(len(cn)):
                ci = keep[k + 1]
                if ci < len(r):
                    nm = cn[k]
                    row[ch_all.index(nm)] = _fl(r[ci], nm == "DIN")
            t_all.append(float(r[0]))
            rows.append(row)
    # sort by time
    order = List[int]()
    for i in range(len(t_all)):
        order.append(i)
    order.sort(key = lambda i: t_all[i])
    ts = [t_all[i] for i in order]
    rs = [rows[i] for i in order]
    # href -> radians
    if unit == "HREF":
        for ci in range(n_ch):
            nm = ch_all[ci]
            if nm.startswith("xpos") or nm.startswith("ypos"):
                for r in rs:
                    r[ci] = asin(r[ci] / 15000.0)
    din = ch_all.index("DIN") if "DIN" in ch_all else -1
    if len(blocks) > 1:
        step = 1000.0 / sfreq
        first = ts[0]
        last = ts[len(ts) - 1]
        n_new = int(ceil((last + step / 2.0 - first) / step))
        nts = List[float]()
        nrs = List[List[float]]()
        j = 0
        for k in range(n_new):
            t = first + float(k) * step
            while j + 1 < len(ts) and abs(ts[j + 1] - t) <= abs(ts[j] - t):
                j += 1
            nts.append(t)
            if abs(ts[j] - t) <= step / 2.0:
                nrs.append(list(rs[j]))
            else:
                row = [np.nan for _ in range(n_ch)]
                if din >= 0:
                    row[din] = 0.0
                nrs.append(row)
        ts = nts
        rs = nrs
    ns = len(ts)
    data = np.zeros((n_ch, ns))
    for s in range(ns):
        for c in range(n_ch):
            data[c, s] = rs[s][c]
    types = List[str]()
    for nm in ch_all:
        if nm.startswith("xpos") or nm.startswith("ypos"):
            types.append("eyegaze")
        elif nm.startswith("pupil"):
            types.append("pupil")
        elif nm == "DIN":
            types.append("stim")
        else:
            types.append("misc")
    info = Info(ch_all, sfreq, types)
    for i in range(n_ch):
        nm = ch_all[i]
        loc = [np.nan for _ in range(12)]
        if nm.endswith("left"):
            loc[3] = -1.0
        elif nm.endswith("right"):
            loc[3] = 1.0
        if nm.startswith("x"):
            loc[4] = -1.0
        elif nm.startswith("y"):
            loc[4] = 1.0
        info.locs[i] = loc
        if unit == "HREF" and (nm.startswith("xpos") or nm.startswith("ypos")):
            info.units[i] = 0
    if meas is not None:
        info.meas_date = meas
    raw = RawArray(data, info)
    raw._raw_kind = "RawEyelink"
    # annotations
    on = List[float]()
    du = List[float]()
    ds = List[str]()
    if len(blocks) > 1:
        for i in range(len(blocks) - 1):
            e = (blocks[i].end_ts - first_ts) / 1000.0
            nx = (blocks[i + 1].start_ts - first_ts) / 1000.0
            on.append(e)
            du.append(nx - e)
            ds.append("BAD_ACQ_SKIP")
    want = List[str]()
    if isinstance(create_annotations, bool):
        if create_annotations:
            want = ["blinks", "saccades", "fixations", "buttons", "messages"]
    else:
        for d in create_annotations:
            if d not in ("blinks", "saccades", "fixations", "buttons", "messages"):
                raise ValueError(f"create_annotations must be True or a list containing one or more of blinks, saccades, fixations, buttons, messages. Got '{d}' instead")
            want.append(d)
    ev_on = List[float]()
    ev_du = List[float]()
    ev_ds = List[str]()
    for key in ("blinks", "saccades", "fixations"):
        if key not in want:
            continue
        for b in blocks:
            lst = b.eblink
            if key == "saccades":
                lst = b.esacc
            elif key == "fixations":
                lst = b.efix
            for r in lst:
                if len(r) < 4:
                    raise ValueError(f"Expected the {key} data in this file to have at least 4 columns")
                ev_on.append((float(r[1]) - first_ts) / 1000.0)
                ev_du.append(float(r[3]) / 1000.0)
                d = key[:len(key) - 1]
                ev_ds.append("BAD_" + d if key == "blinks" else d)
    if "messages" in want:
        for b in blocks:
            for m in b.msgs:
                t = float(m[0])
                off = 0.0
                text = " ".join(m[1:])
                if apply_offsets and len(m) > 2:
                    if _is_num(m[1]):
                        off = float(m[1])
                        text = " ".join(m[2:])
                ev_on.append((t - first_ts) / 1000.0 + off / 1000.0)
                ev_du.append(0.0)
                ev_ds.append(text)
    if "buttons" in want:
        brow = List[Tuple[float, str]]()
        for b in blocks:
            for e in b.buttons:
                bid = int(e[1])
                act = "press" if int(e[2]) == 1 else "release"
                brow.append(((float(e[0]) - first_ts) / 1000.0, f"button_{bid}_{act}"))
        brow.sort(key = lambda x: x[0])
        for x in brow:
            ev_on.append(x[0])
            ev_du.append(0.0)
            ev_ds.append(x[1])
    on.extend(ev_on)
    du.extend(ev_du)
    ds.extend(ev_ds)
    if len(on) > 0:
        idx = List[int]()
        for i in range(len(on)):
            idx.append(i)
        idx.sort(key = lambda i: on[i])
        raw.set_annotations(Annotations([on[i] for i in idx], [du[i] for i in idx], [ds[i] for i in idx]), False)
    return raw
