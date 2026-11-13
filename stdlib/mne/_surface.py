# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Surface io (FreeSurfer triangle binary + Wavefront obj + .tri), talairach xfm,
# head_to_mni / head_to_mri / vertex_to_mni with given transforms.

import numpy as np
from .io._bytes import _read_file, _read_text, _uint_be, _i32_be, _f32_be, _ByteWriter, _arr_be
from .transforms import Transform, invert_transform, combine_transforms, apply_trans


def read_surface(fname: str, read_metadata: bool = False, return_dict: bool = False, file_format: str = "auto",
                 verbose = None):
    """mne.read_surface: (rr, tris) from a FreeSurfer triangle file or Wavefront .obj."""
    fmt = file_format
    if fmt == "auto":
        fmt = "obj" if fname.lower().endswith(".obj") else "freesurfer"
    if fmt == "obj":
        return _read_obj(fname)
    b = _read_file(fname)
    magic = _uint_be(b, 0, 3)
    if magic != 0xFFFFFE:
        raise ValueError(f"File {fname} does not have the FreeSurfer triangle magic number (quad / other formats are not supported)")
    pos = 3
    nl = 0
    while nl < 2 and pos < len(b):
        if b[pos] == 10:
            nl += 1
        pos += 1
    vnum = _i32_be(b, pos)
    fnum = _i32_be(b, pos + 4)
    pos += 8
    rr = _arr_be(b, pos, vnum * 3, "f4").reshape((vnum, 3))
    pos += 4 * vnum * 3
    tris = _arr_be(b, pos, fnum * 3, "i4").reshape((fnum, 3)).astype(int)
    return rr, tris


def _read_obj(fname: str):
    coords = List[List[float]]()
    faces = List[List[int]]()
    for line0 in _read_text(fname).split("\n"):
        line = line0.strip()
        if len(line) == 0 or line[0] == "#":
            continue
        sp = line.split()
        if sp[0] == "v":
            coords.append([float(x) for x in sp[1:]])
        elif sp[0] == "f":
            dat = [int(x.split("/")[0]) for x in sp[1:]]
            if len(dat) != 3:
                raise RuntimeError("Only triangle faces allowed.")
            faces.append([d - 1 for d in dat])
    return np.array(coords), np.array(faces)


def write_surface(fname: str, coords, faces, create_stamp: str = "", volume_info = None,
                  file_format: str = "auto", overwrite: bool = False, verbose = None):
    """mne.write_surface (FreeSurfer triangle format, no volume_info; or .obj)."""
    c = np.asarray(coords, dtype=float)
    f = np.asarray(faces).astype(int)
    fmt = file_format
    if fmt == "auto":
        fmt = "obj" if fname.lower().endswith(".obj") else "freesurfer"
    if fmt == "obj":
        out = List[str]()
        for ln in create_stamp.split("\n"):
            if len(ln) > 0:
                out.append(f"# {ln}")
        for i in range(c.shape[0]):
            out.append(f"v {c[i, 0]} {c[i, 1]} {c[i, 2]}")
        for i in range(f.shape[0]):
            out.append(f"f {f[i, 0] + 1} {f[i, 1] + 1} {f[i, 2] + 1}")
        with open(fname, "w") as fh:
            fh.write("\n".join(out) + "\n")
        return
    w = _ByteWriter()
    w.uint_be(0xFFFFFE, 3)
    w.raw((create_stamp if len(create_stamp) > 0 else "created by typhon") + "\n\n")
    w.i32(c.shape[0])
    w.i32(f.shape[0])
    for i in range(c.shape[0]):
        for k in range(3):
            w.f32(c[i, k])
    for i in range(f.shape[0]):
        for k in range(3):
            w.i32(int(f[i, k]))
    w.save(fname)


def read_tri(fname_in: str, swap: bool = False, verbose = None):
    """mne.read_tri: (rr, tris) from an ASCII .tri file."""
    lines = _read_text(fname_in).split("\n")
    n_nodes = int(lines[0])
    n_tris = int(lines[n_nodes + 1])
    n_items = len(lines[1].split())
    off = 0
    if n_items == 3 or n_items == 6 or n_items == 14 or n_items == 17:
        off = 0
    elif n_items == 4 or n_items == 7:
        off = 1
    else:
        raise OSError("Unrecognized format of data.")
    rr = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        p = lines[1 + i].split()
        for k in range(3):
            rr[i, k] = float(p[off + k])
    tris = np.zeros((n_tris, 3), dtype=int)
    for i in range(n_tris):
        p = lines[n_nodes + 2 + i].split()
        for k in range(3):
            tris[i, k] = int(p[off + k])
    if swap:
        for i in range(n_tris):
            t = tris[i, 1]
            tris[i, 1] = tris[i, 2]
            tris[i, 2] = t
    tris -= 1
    return rr, tris


def _read_fs_xfm(fname: str) -> np.ndarray[float, 2]:
    lines = _read_text(fname).split("\n")
    i = 0
    found = False
    while i < len(lines):
        if lines[i].startswith("Linear_Transform"):
            found = True
            break
        i += 1
    if not found:
        raise ValueError(f'Failed to find "Linear_Transform" string in xfm file:\n{fname}')
    m = np.eye(4)
    for r in range(3):
        if i + 1 + r >= len(lines):
            raise ValueError("Could not find enough linear transform lines")
        parts = lines[i + 1 + r].strip().strip(";").split()
        for c in range(4):
            m[r, c] = float(parts[c])
    return m


def read_talxfm(subject: str, subjects_dir = None, mri_ras_t: Optional[Transform] = None, verbose = None) -> Transform:
    """mne.read_talxfm: MRI (surface RAS) -> MNI Talairach. The orig.mgz header is not read:
    pass mri_ras_t (the surface-RAS -> scanner-RAS transform, in meters) explicitly, or identity is used."""
    sd = str(subjects_dir)
    xfm = _read_fs_xfm(f"{sd}/{subject}/mri/transforms/talairach.xfm")
    xfm[:3, 3] = xfm[:3, 3] / 1000.0
    ras_mni = Transform("ras", "mni_tal", xfm)
    mr = mri_ras_t if mri_ras_t is not None else Transform("mri", "ras", np.eye(4))
    return combine_transforms(mr.__val__() if mri_ras_t is not None else mr, ras_mni, "mri", "mni_tal")


def head_to_mni(pos, subject: str, mri_head_t: Transform, subjects_dir = None, mri_mni_t: Optional[Transform] = None,
                verbose = None):
    """mne.head_to_mni: head (m) -> MNI (mm). The MRI -> MNI transform is read with read_talxfm
    unless mri_mni_t is given."""
    mm = mri_mni_t.__val__() if mri_mni_t is not None else read_talxfm(subject, subjects_dir)
    head_mri = invert_transform(mri_head_t) if mri_head_t.fro == 5 else mri_head_t
    head_mni = combine_transforms(head_mri, mm, "head", "mni_tal")
    return apply_trans(head_mni.trans, np.asarray(pos, dtype=float)) * 1000.0


def head_to_mri(pos, subject: str, mri_head_t: Transform, subjects_dir = None, *, kind: str = "mri",
                unscale: bool = False, mri_ras_t: Optional[Transform] = None, verbose = None):
    """mne.head_to_mri: head (m) -> MRI surface RAS (mm). kind='ras' requires mri_ras_t."""
    head_mri = invert_transform(mri_head_t) if mri_head_t.fro == 5 else mri_head_t
    dest = head_mri.trans
    if kind == "ras":
        if mri_ras_t is None:
            raise NotImplementedError("head_to_mri(kind='ras') needs mri_ras_t (the T1.mgz header is not read)")
        dest = combine_transforms(head_mri, mri_ras_t.__val__(), "head", "ras").trans
    return apply_trans(dest, np.asarray(pos, dtype=float)) * 1e3


def vertex_to_mni(vertices, hemis, subject: str, subjects_dir = None, surfaces = None,
                  mri_mni_t: Optional[Transform] = None, verbose = None):
    """mne.vertex_to_mni with explicit surfaces [lh_rr, rh_rr] (mm) and optionally mri_mni_t."""
    vs = np.atleast_1d(np.asarray(vertices).astype(int))
    hs = np.atleast_1d(np.asarray(hemis).astype(int))
    if hs.shape[0] == 1 and vs.shape[0] > 1:
        hs = np.full(vs.shape[0], int(hs[0]), dtype=int)
    if hs.shape[0] != vs.shape[0]:
        raise ValueError("hemi and vertices must match in length")
    mm = mri_mni_t.__val__() if mri_mni_t is not None else read_talxfm(subject, subjects_dir)
    t = mm.trans.copy()
    t[:3, 3] = t[:3, 3] * 1000.0
    pts = np.zeros((vs.shape[0], 3))
    for i in range(vs.shape[0]):
        if isinstance(surfaces, NoneType):
            rr = read_surface(f"{subjects_dir}/{subject}/surf/{'lh' if hs[i] == 0 else 'rh'}.white")[0]
            pts[i] = rr[int(vs[i])]
        else:
            pts[i] = np.asarray(surfaces[int(hs[i])], dtype=float)[int(vs[i])]
    return apply_trans(t, pts)


def grade_to_tris(grade: int):
    raise NotImplementedError("grade_to_tris needs the icosahedron tessellation shipped with mne (icos.fif.gz)")
