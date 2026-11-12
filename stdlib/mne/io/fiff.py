# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# read_raw_fif / Raw.save, read_events / write_events (FIF part).

import numpy as np
from ._fiff import (FifFile, FifWriter, read_meas_info, read_annotations_node,
                    FIFFB_MEAS, FIFFB_RAW_DATA, FIFFB_CONTINUOUS_DATA, FIFF_FIRST_SAMPLE,
                    FIFF_DATA_SKIP, FIFF_DATA_BUFFER, FIFFB_MNE_EVENTS, FIFF_MNE_EVENT_LIST,
                    FIFF_COMMENT, FIFF_DATA_PACK, FIFFT_FLOAT, FIFFT_DOUBLE, FIFF_BLOCK_ID)
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations


def read_raw_fif(fname, allow_maxshield: bool = False, preload = False,
                 on_split_missing: str = "raise", verbose = None):
    """Read a raw FIF file (single file, float/double/int/short buffers).

    The data are always loaded into memory (preload is accepted for API
    compatibility; values are identical)."""
    f = FifFile(str(fname))
    info = read_meas_info(f)
    nodes = f.tree.find(FIFFB_RAW_DATA)
    if len(nodes) == 0:
        nodes = f.tree.find(FIFFB_CONTINUOUS_DATA)
    if len(nodes) == 0:
        raise ValueError("No raw data in " + str(fname))
    node = nodes[0]
    nchan = info.nchan
    first_samp = 0
    first_skip = 0
    k = 0
    ents = node.entries
    if k < len(ents) and ents[k].kind == FIFF_FIRST_SAMPLE:
        first_samp = f.int(ents[k])
        k += 1
    if k < len(ents) and ents[k].kind == FIFF_DATA_SKIP:
        first_skip = f.int(ents[k])
        k += 1
    bufs = List[np.ndarray[float, 2]]()
    nskip = 0
    total = 0
    buf_len = 0
    for j in range(k, len(ents)):
        t = ents[j]
        if t.kind == FIFF_DATA_BUFFER:
            b = f.buffer(t, nchan)
            if first_skip > 0:
                first_samp += b.shape[0] * first_skip
                first_skip = 0
            if nskip > 0:
                bufs.append(np.zeros((nskip * b.shape[0], nchan)))
                nskip = 0
            bufs.append(b)
            buf_len = b.shape[0]
        elif t.kind == FIFF_DATA_SKIP:
            nskip = f.int(t)
    for b in bufs:
        total += b.shape[0]
    data = np.empty((nchan, total), dtype=float)
    pos = 0
    cal = np.array([info.cals[i] * info.ranges[i] for i in range(nchan)])
    for b in bufs:
        n = b.shape[0]
        for c in range(nchan):
            data[c, pos:pos + n] = b[:, c] * cal[c]
        pos += n
    raw = RawArray(data, info, first_samp)
    raw._raw_kind = "Raw"
    meas = f.tree.find(FIFFB_MEAS)[0]
    on, du, de, found = read_annotations_node(f, meas)
    if found and len(on) > 0:
        ft = raw.first_time
        raw.set_annotations(Annotations([o - ft for o in on], du, de), False)
    return raw


def _write_raw_fif(raw, fname: str, picks, tmin: float, tmax: Optional[float],
                   buffer_size_sec: Optional[float], fmt: str):
    sf = raw.info.sfreq
    from .._info import _picks_to_idx
    pk = _picks_to_idx(raw.info, picks, "all", ())
    info = raw.info._pick(pk)
    n = raw.n_times
    start = int(round(tmin * sf))
    stop = n if tmax is None else min(n, int(round(tmax.__val__() * sf)) + 1)
    bss = 1.0 if buffer_size_sec is None else buffer_size_sec.__val__()
    buffer_size = int(np.ceil(bss * sf))
    w = FifWriter()
    w.start_block(FIFFB_MEAS)
    w.write_id(FIFF_BLOCK_ID)
    dt = 4 if fmt == "single" else 5
    w.write_meas_info(info, dt)
    ann = raw.annotations
    if len(ann) > 0:
        w.write_annotations(ann.onset, ann.duration, ann.description)
    w.start_block(FIFFB_RAW_DATA)
    w.write_int(FIFF_FIRST_SAMPLE, raw.first_samp + start)
    sel = np.array(pk, dtype=int)
    decal = np.array([1.0 / info.cals[i] for i in range(info.nchan)])
    for first in range(start, stop, buffer_size):
        last = min(first + buffer_size, stop)
        blk = raw._data[sel, first:last]
        buf = (blk * decal[:, None]).T.copy()
        if fmt == "single":
            w.write_buffer(buf)
        else:
            m = buf.shape[0] * buf.shape[1]
            w._head(FIFF_DATA_BUFFER, FIFFT_DOUBLE, 8 * m)
            for i in range(buf.shape[0]):
                for j in range(buf.shape[1]):
                    w.w.f64(buf[i, j])
    w.end_block(FIFFB_RAW_DATA)
    w.end_block(FIFFB_MEAS)
    w.end_file(fname)


def _read_events_fif(fname: str) -> Tuple[np.ndarray[int, 2], Dict[str, int], bool]:
    f = FifFile(fname)
    nodes = f.tree.find(FIFFB_MNE_EVENTS)
    if len(nodes) == 0:
        raise ValueError("Could not find event data")
    nd = nodes[0]
    t = nd.tag(FIFF_MNE_EVENT_LIST)
    if t is None:
        raise ValueError("Could not find any events")
    v = f.ints(t)
    ev = np.array(v, dtype=int).reshape((len(v) // 3, 3))
    ids = Dict[str, int]()
    d = nd.tag(FIFF_COMMENT)
    has = d is not None
    if d is not None:
        s = f.string(d)
        for m in s.split(";"):
            j = m.rfind(":")
            ids[m[:j]] = int(m[j + 1:])
    return ev, ids, has


def _write_events_fif(fname: str, events: np.ndarray[int, 2]):
    w = FifWriter()
    w.start_block(FIFFB_MNE_EVENTS)
    v = List[int]()
    for i in range(events.shape[0]):
        for j in range(3):
            v.append(int(events[i, j]))
    w.write_ints(FIFF_MNE_EVENT_LIST, v)
    w.end_block(FIFFB_MNE_EVENTS)
    w.end_file(fname)
