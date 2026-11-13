# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Extra FIF tag helpers: recursive tag search, double / int matrix writers,
# coordinate transform, name lists and the constants used by the cov / proj /
# trans / source space / forward solution readers and writers.

import numpy as np
from ._fiff import (FifFile, FifWriter, Node, Tag, FIFFT_DOUBLE, FIFFT_FLOAT, FIFFT_INT, FIFFT_MATRIX,
                    FIFF_BLOCK_START, FIFF_NCHAN, FIFF_NAME, FIFF_COMMENT, FIFFB_MNE_BAD_CHANNELS,
                    FIFF_MNE_CH_NAME_LIST)

FIFF_COORD_TRANS = 222
FIFFT_COORD_TRANS_STRUCT = 35
FIFFB_MNE = 350
FIFFB_MNE_SOURCE_SPACE = 351
FIFFB_MNE_FORWARD_SOLUTION = 352
FIFFB_MNE_PARENT_MRI_FILE = 353
FIFFB_MNE_ENV = 358
FIFFB_MNE_COV = 355
FIFFB_PROJ = 313
FIFFB_PROJ_ITEM = 314
FIFFB_MNE_NAMED_MATRIX = 357
FIFF_MNE_COV_KIND = 3530
FIFF_MNE_COV_DIM = 3531
FIFF_MNE_COV = 3532
FIFF_MNE_COV_DIAG = 3533
FIFF_MNE_COV_EIGENVALUES = 3534
FIFF_MNE_COV_EIGENVECTORS = 3535
FIFF_MNE_COV_NFREE = 3536
FIFF_MNE_COV_METHOD = 3537
FIFF_MNE_COV_SCORE = 3538
FIFF_MNE_ROW_NAMES = 3502
FIFF_MNE_COL_NAMES = 3503
FIFF_MNE_NROW = 3504
FIFF_MNE_NCOL = 3505
FIFF_MNE_COORD_FRAME = 3506
FIFF_PROJ_ITEM_KIND = 3411
FIFF_PROJ_ITEM_TIME = 3412
FIFF_PROJ_ITEM_NVEC = 3414
FIFF_PROJ_ITEM_VECTORS = 3415
FIFF_PROJ_ITEM_CH_NAME_LIST = 3417
FIFF_MNE_PROJ_ITEM_ACTIVE = 3560
FIFF_MNE_ICA_PCA_EXPLAINED_VAR = 3605
FIFF_MNE_SOURCE_SPACE_POINTS = 3510
FIFF_MNE_SOURCE_SPACE_NORMALS = 3511
FIFF_MNE_SOURCE_SPACE_NPOINTS = 3512
FIFF_MNE_SOURCE_SPACE_SELECTION = 3513
FIFF_MNE_SOURCE_SPACE_NUSE = 3514
FIFF_MNE_SOURCE_SPACE_ID = 3517
FIFF_MNE_SOURCE_SPACE_TYPE = 3518
FIFF_MNE_SOURCE_SPACE_NTRI = 3590
FIFF_MNE_SOURCE_SPACE_TRIANGLES = 3591
FIFF_MNE_SOURCE_SPACE_NUSE_TRI = 3592
FIFF_MNE_SOURCE_SPACE_USE_TRIANGLES = 3593
FIFF_MNE_FORWARD_SOLUTION = 3520
FIFF_MNE_SOURCE_ORIENTATION = 3521
FIFF_MNE_INCLUDED_METHODS = 3522
FIFF_MNE_FILE_NAME = 3508
FIFF_BLOCK_ID = 103
FIFFV_MNE_MEG = 1
FIFFV_MNE_EEG = 2
FIFFV_MNE_SPACE_SURFACE = 1
FIFFV_MNE_SPACE_VOLUME = 2
FIFFV_MNE_SPACE_DISCRETE = 3


def _find_all_tags(n: Node, kind: int) -> List[Tag]:
    """All tags of the given kind in the node and its descendants (document order)."""
    out = List[Tag]()
    for t in n.entries:
        if t.kind == kind:
            out.append(t)
    for c in n.children:
        for t in _find_all_tags(c, kind):
            out.append(t)
    return out


def _node_tag(n: Node, kind: int) -> Optional[Tag]:
    return n.tag(kind)


def _read_name_list(s: str) -> List[str]:
    """_safe_read_name_list: ':'-separated, {COLON} escapes a colon."""
    if len(s) == 0:
        return List[str]()
    return [p.replace("{COLON}", ":") for p in s.split(":")]


def _write_name_list_str(names: List[str]) -> str:
    return ":".join([n.replace(":", "{COLON}") for n in names])


@extend
class FifWriter:
    def write_double_matrix(self, kind: int, flat: np.ndarray[float, 1], dims: List[int]):
        nd = len(dims)
        self._head(kind, FIFFT_DOUBLE | FIFFT_MATRIX, 8 * flat.shape[0] + 4 * (nd + 1))
        for i in range(flat.shape[0]):
            self.w.f64(flat[i])
        for k in range(nd):
            self.w.i32(dims[nd - 1 - k])
        self.w.i32(nd)

    def write_int_matrix(self, kind: int, flat: List[int], dims: List[int]):
        nd = len(dims)
        self._head(kind, FIFFT_INT | FIFFT_MATRIX, 4 * len(flat) + 4 * (nd + 1))
        for v in flat:
            self.w.i32(v)
        for k in range(nd):
            self.w.i32(dims[nd - 1 - k])
        self.w.i32(nd)

    def write_doubles(self, kind: int, v: np.ndarray[float, 1]):
        self._head(kind, FIFFT_DOUBLE, 8 * v.shape[0])
        for i in range(v.shape[0]):
            self.w.f64(v[i])

    def write_name_list(self, kind: int, names: List[str]):
        self.write_string(kind, _write_name_list_str(names))

    def write_coord_trans(self, fro: int, to: int, m: np.ndarray[float, 2]):
        self._head(FIFF_COORD_TRANS, FIFFT_COORD_TRANS_STRUCT, 4 * 2 * 12 + 4 * 2)
        self.w.i32(fro)
        self.w.i32(to)
        mi = np.linalg.inv(m)
        for a in (m, mi):
            for i in range(3):
                for j in range(3):
                    self.w.f32(a[i, j])
            for i in range(3):
                self.w.f32(a[i, 3])

    def write_bad_channels(self, bads: List[str]):
        if len(bads) > 0:
            self.start_block(FIFFB_MNE_BAD_CHANNELS)
            self.write_name_list(FIFF_MNE_CH_NAME_LIST, bads)
            self.end_block(FIFFB_MNE_BAD_CHANNELS)


def _f_matrix_rows(f: FifFile, t: Tag) -> np.ndarray[float, 2]:
    flat, dims = f.matrix(t)
    if len(dims) == 1:
        return flat.reshape((1, dims[0]))
    return flat.reshape((dims[0], dims[1]))
