# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# SourceMorph (discrete / volume-style nearest-vertex morphing with optional smoothing steps)
# and make_bem_model / make_bem_solution for spherical (sphere-equivalent) models.
# Surface morphing through FreeSurfer spherical registration is not available.

import numpy as np
from .source_estimate import _STC
from .source_space import SourceSpaces
from .bem import ConductorModel, make_sphere_model


class SourceMorph:
    """mne.SourceMorph for a precomputed linear morph map (n_to x n_from), as mne's
    _morph_mat path: stc_to = morph_mat @ stc_from."""
    subject_from: Optional[str]
    subject_to: Optional[str]
    kind: str
    morph_mat: np.ndarray[float, 2]
    vertices_to: List[np.ndarray[int, 1]]
    smooth: Optional[int]
    spacing: str

    def __init__(self, morph_mat, vertices_to, kind: str = "volume", subject_from = None, subject_to = None,
                 smooth = None):
        self.morph_mat = np.asarray(morph_mat, dtype=float)
        self.vertices_to = [np.asarray(v).astype(int) for v in vertices_to]
        self.kind = kind
        self.subject_from = None if isinstance(subject_from, NoneType) else str(subject_from)
        self.subject_to = None if isinstance(subject_to, NoneType) else str(subject_to)
        self.smooth = None if isinstance(smooth, NoneType) else int(smooth)
        self.spacing = "ico5"

    def apply(self, stc_from, mri_resolution = None, mri_space = None, output: str = "stc", verbose = None):
        n_to = 0
        for v in self.vertices_to:
            n_to += v.shape[0]
        if stc_from.data.shape[0] != self.morph_mat.shape[1]:
            raise ValueError("Source estimate has the wrong number of vertices for this morph")
        d = np.dot(self.morph_mat, stc_from.data)
        return _STC[np.ndarray[float, 2]](d, self.vertices_to, stc_from.tmin, stc_from.tstep, self.subject_to, self.kind)

    def __repr__(self) -> str:
        return f"<SourceMorph | {self.kind} , subject_from : {self.subject_from}, subject_to : {self.subject_to}>"


def _nearest_idx(a: np.ndarray[float, 2], b: np.ndarray[float, 2]) -> np.ndarray[int, 1]:
    out = np.zeros(b.shape[0], dtype=int)
    for i in range(b.shape[0]):
        best = np.inf
        for j in range(a.shape[0]):
            d = float(np.sum((a[j] - b[i]) ** 2))
            if d < best:
                best = d
                out[i] = j
    return out


def compute_source_morph(src: SourceSpaces, subject_from = None, subject_to: str = "fsaverage", src_to: Optional[SourceSpaces] = None,
                         smooth = None, spacing = 5, verbose = None) -> SourceMorph:
    """Nearest-vertex morph between two discrete/volume source spaces (src -> src_to), as the
    morph_mat of mne's volume nearest assignment; smooth = number of mesh-free box smoothing
    steps is not supported (None only)."""
    if src_to is None:
        raise NotImplementedError("compute_source_morph needs src_to (no FreeSurfer subject tree)")
    if not isinstance(smooth, NoneType):
        raise NotImplementedError("compute_source_morph: smoothing requires a surface mesh")
    st = src_to.__val__()
    ra = List[np.ndarray[float, 2]]()
    for s in src.spaces:
        ra.append(s.rr[s.vertno])
    rb = List[np.ndarray[float, 2]]()
    vt = List[np.ndarray[int, 1]]()
    for s in st.spaces:
        rb.append(s.rr[s.vertno])
        vt.append(s.vertno)
    A = np.concatenate(ra)
    Bm = np.concatenate(rb)
    idx = _nearest_idx(A, Bm)
    M = np.zeros((Bm.shape[0], A.shape[0]))
    for i in range(Bm.shape[0]):
        M[i, int(idx[i])] = 1.0
    return SourceMorph(M, vt, "volume", subject_from, subject_to, None)


def make_bem_model(subject = None, ico = 4, conductivity = (0.3, 0.006, 0.3), subjects_dir = None, verbose = None,
                   sphere = None):
    """Only the sphere-equivalent shells are available: pass sphere=(r0, head_radius) tuple; a FreeSurfer
    subject tree is required otherwise."""
    if sphere is None:
        raise NotImplementedError("make_bem_model needs the FreeSurfer subject tree (pass sphere=(r0, radius) for a sphere-equivalent model)")
    sp = sphere.__val__()
    sig = [float(c) for c in conductivity]
    if len(sig) == 3:
        rel = (0.0 + 0.92, 0.97, 1.0)
        return make_sphere_model(sp[0], sp[1], None, rel, (sig[0], sig[1], sig[2]))
    return make_sphere_model(sp[0], sp[1], None, (1.0,), (sig[0],))


def make_bem_solution(surfs, solver: str = "mne", verbose = None) -> ConductorModel:
    """Sphere models need no solution matrix: the ConductorModel is returned as is."""
    if isinstance(surfs, ConductorModel):
        return surfs.copy()
    raise NotImplementedError("make_bem_solution supports ConductorModel (sphere) inputs only")
