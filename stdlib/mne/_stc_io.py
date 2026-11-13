# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Source estimate files (.stc / .w), read_source_estimate, SourceEstimate.save,
# VectorSourceEstimate and MixedSourceEstimate constructors.

import numpy as np
from .io._bytes import _read_file, _uint_be, _f32_be, _ByteWriter, _u8
from .source_estimate import _STC


def _put_u4(w: _ByteWriter, v: int):
    w.uint_be(v, 4)


def _write_stc(filename: str, tmin: float, tstep: float, vertices: np.ndarray[int, 1],
               data: np.ndarray[float, 2]):
    """mne.source_estimate._write_stc."""
    w = _ByteWriter()
    w.f32(1000.0 * tmin)
    w.f32(1000.0 * tstep)
    _put_u4(w, vertices.shape[0])
    for v in vertices:
        _put_u4(w, int(v))
    _put_u4(w, data.shape[1])
    for t in range(data.shape[1]):
        for i in range(data.shape[0]):
            w.f32(data[i, t])
    w.save(filename)


def _read_stc(filename: str):
    """(tmin, tstep, vertices, data (n_vert, n_times))."""
    b = _read_file(filename)
    tmin = _f32_be(b, 0) / 1000.0
    tstep = _f32_be(b, 4) / 1000.0
    nv = _uint_be(b, 8, 4)
    verts = np.zeros(nv, dtype=int)
    for i in range(nv):
        verts[i] = _uint_be(b, 12 + 4 * i, 4)
    off = 12 + 4 * nv
    nt = _uint_be(b, off, 4)
    off += 4
    if nv > 0 and ((len(b) // 4 - 4 - nv) % (nt * nv)) != 0:
        raise ValueError("incorrect stc file size")
    data = np.zeros((nv, nt))
    for t in range(nt):
        for i in range(nv):
            data[i, t] = _f32_be(b, off + 4 * (t * nv + i))
    return tmin, tstep, verts, data


def _write_w(filename: str, vertices: np.ndarray[int, 1], data: np.ndarray[float, 1]):
    w = _ByteWriter()
    w.zeros(2)
    w.uint_be(vertices.shape[0], 3)
    for i in range(vertices.shape[0]):
        w.uint_be(int(vertices[i]), 3)
        w.f32(data[i])
    w.save(filename)


def _read_w(filename: str):
    b = _read_file(filename)
    nv = _uint_be(b, 2, 3)
    verts = np.zeros(nv, dtype=int)
    data = np.zeros(nv)
    off = 5
    for i in range(nv):
        verts[i] = _uint_be(b, off, 3)
        data[i] = _f32_be(b, off + 3)
        off += 7
    return verts, data


def write_stc(stc: _STC[np.ndarray[float, 2]], fname: str, ftype: str = "stc", *, overwrite: bool = False,
              verbose = None):
    """SourceEstimate.save: surface -> <fname>-lh.stc / -rh.stc, volume -> <fname>-vl.stc."""
    if stc.kind == "volume":
        name = fname
        if not name.endswith("-vl.stc") and not name.endswith("-vl.w"):
            name = fname + ("-vl.stc" if ftype == "stc" else "-vl.w")
        if ftype == "stc":
            _write_stc(name, stc.tmin, stc.tstep, stc.vertices[0], stc.data)
        else:
            _write_w(name, stc.vertices[0], stc.data[:, 0].copy())
        return
    nl = stc.vertices[0].shape[0]
    if ftype == "stc":
        _write_stc(fname + "-lh.stc", stc.tmin, stc.tstep, stc.vertices[0], stc.data[:nl].copy())
        _write_stc(fname + "-rh.stc", stc.tmin, stc.tstep, stc.vertices[1], stc.data[nl:].copy())
    else:
        if stc.data.shape[1] != 1:
            raise ValueError("w files can only contain a single time point")
        _write_w(fname + "-lh.w", stc.vertices[0], stc.data[:nl, 0].copy())
        _write_w(fname + "-rh.w", stc.vertices[1], stc.data[nl:, 0].copy())


def _sort_hemi(v: np.ndarray[int, 1], d: np.ndarray[float, 2]):
    idx = np.argsort(v)
    return v[idx], d[idx]


def read_source_estimate(fname: str, subject = None) -> _STC[np.ndarray[float, 2]]:
    """mne.read_source_estimate for .stc / .w files (surface: pass the stem without -lh/-rh)."""
    name = fname
    ftype = "none"
    if name.endswith("-vl.stc") or name.endswith("-vl.w"):
        ftype = "volume"
    elif name.endswith("-lh.stc") or name.endswith("-rh.stc"):
        name = name[:-7]
        ftype = "surface"
    elif name.endswith("-lh.w") or name.endswith("-rh.w"):
        name = name[:-5]
        ftype = "w"
    elif name.endswith(".stc"):
        name = name[:-4]
    from os.path import exists
    if ftype == "none" or ftype == "surface" or ftype == "w":
        if exists(name + "-lh.stc") and exists(name + "-rh.stc") and ftype != "w":
            ftype = "surface"
        elif exists(name + "-lh.w") and exists(name + "-rh.w"):
            ftype = "w"
        else:
            raise OSError(f"SourceEstimate File(s) not found for: {fname}")
    if ftype == "volume":
        if name.endswith(".stc"):
            tmin, tstep, v, d = _read_stc(name)
            return _STC[np.ndarray[float, 2]](d, [v], tmin, tstep, subject, "volume")
        v, d1 = _read_w(name)
        return _STC[np.ndarray[float, 2]](d1.reshape((d1.shape[0], 1)), [v], 0.0, 1.0, subject, "volume")
    if ftype == "surface":
        tl, sl, vl, dl = _read_stc(name + "-lh.stc")
        tr, sr, vr, dr = _read_stc(name + "-rh.stc")
        vl, dl = _sort_hemi(vl, dl)
        vr, dr = _sort_hemi(vr, dr)
        return _STC[np.ndarray[float, 2]](np.concatenate([dl, dr]), [vl, vr], tl, sl, subject, "surface")
    vl, wl = _read_w(name + "-lh.w")
    vr, wr = _read_w(name + "-rh.w")
    vl, dl = _sort_hemi(vl, wl.reshape((wl.shape[0], 1)))
    vr, dr = _sort_hemi(vr, wr.reshape((wr.shape[0], 1)))
    return _STC[np.ndarray[float, 2]](np.concatenate([dl, dr]), [vl, vr], 0.0, 1.0, subject, "surface")


def VectorSourceEstimate(data, vertices, tmin: float, tstep: float, subject = None, verbose = None):
    """Surface vector source estimate: data (n_vertices, 3, n_times)."""
    d = np.asarray(data, dtype=float)
    if d.ndim != 3 or d.shape[1] != 3:
        raise ValueError("Data must have shape (n_dipoles, 3, n_times)")
    return _STC[np.ndarray[float, 3]](d, vertices, tmin, tstep, subject, "surface")


def MixedSourceEstimate(data, vertices, tmin: float, tstep: float, subject = None, verbose = None):
    """Mixed (surface + volume) scalar source estimate."""
    return _STC[np.ndarray[float, 2]](np.asarray(data, dtype=float), vertices, tmin, tstep, subject, "mixed")


def MixedVectorSourceEstimate(data, vertices, tmin: float, tstep: float, subject = None, verbose = None):
    d = np.asarray(data, dtype=float)
    return _STC[np.ndarray[float, 3]](d, vertices, tmin, tstep, subject, "mixed")
