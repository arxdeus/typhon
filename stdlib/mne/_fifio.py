# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# FIF readers/writers: read_cov / write_cov, read_proj / write_proj,
# read_source_spaces / write_source_spaces (surface and discrete),
# read_forward_solution / write_forward_solution (free and fixed orientation).

import numpy as np
from ._info import Info, Projection
from .cov import Covariance
from .io._fiff import (FifFile, FifWriter, Node, Tag, FIFFB_MEAS, FIFFB_MEAS_INFO, FIFF_NCHAN, FIFF_NAME,
                       FIFF_COMMENT, FIFFB_MNE_BAD_CHANNELS, FIFF_MNE_CH_NAME_LIST, FIFFT_DOUBLE,
                       read_meas_info)
from .io._fifext import (_find_all_tags, _read_name_list, _f_matrix_rows, FIFFB_MNE_COV, FIFFB_PROJ,
                         FIFFB_PROJ_ITEM, FIFF_MNE_COV_KIND, FIFF_MNE_COV_DIM, FIFF_MNE_COV,
                         FIFF_MNE_COV_DIAG, FIFF_MNE_COV_EIGENVALUES, FIFF_MNE_COV_EIGENVECTORS,
                         FIFF_MNE_COV_NFREE, FIFF_MNE_COV_METHOD, FIFF_MNE_COV_SCORE, FIFF_MNE_ROW_NAMES,
                         FIFF_PROJ_ITEM_KIND, FIFF_PROJ_ITEM_NVEC, FIFF_PROJ_ITEM_VECTORS,
                         FIFF_PROJ_ITEM_CH_NAME_LIST, FIFF_PROJ_ITEM_TIME, FIFF_MNE_PROJ_ITEM_ACTIVE,
                         FIFF_MNE_ICA_PCA_EXPLAINED_VAR)


def _tag_int(f: FifFile, n: Node, kind: int, default: int) -> int:
    t = n.tag(kind)
    if t is None:
        return default
    return f.int(t.__val__())


# ---------------------------------------------------------------- projections

def _write_proj_block(w: FifWriter, projs: List[Projection]):
    if len(projs) == 0:
        return
    w.start_block(FIFFB_PROJ)
    for p in projs:
        w.start_block(FIFFB_PROJ_ITEM)
        w.write_name_list(FIFF_PROJ_ITEM_CH_NAME_LIST, p.col_names)
        w.write_string(FIFF_NAME, p.desc)
        w.write_int(FIFF_NCHAN, len(p.col_names))
        w.write_int(FIFF_PROJ_ITEM_KIND, p.kind)
        if p.kind == 1:
            w.write_float(FIFF_PROJ_ITEM_TIME, 0.0)
        w.write_int(FIFF_PROJ_ITEM_NVEC, p.data.shape[0])
        w.write_int(FIFF_MNE_PROJ_ITEM_ACTIVE, 1 if p.active else 0)
        w.write_float_matrix(FIFF_PROJ_ITEM_VECTORS, p.data.reshape(p.data.shape[0] * p.data.shape[1]),
                             [p.data.shape[0], p.data.shape[1]])
        if p.explained_var is not None:
            w.write_float(FIFF_MNE_ICA_PCA_EXPLAINED_VAR, p.explained_var.__val__())
        w.end_block(FIFFB_PROJ_ITEM)
    w.end_block(FIFFB_PROJ)


def _read_proj_node(f: FifFile, node: Node) -> List[Projection]:
    projs = List[Projection]()
    blocks = node.find(FIFFB_PROJ)
    if len(blocks) == 0:
        return projs
    for item in blocks[0].find(FIFFB_PROJ_ITEM):
        td = item.tag(FIFF_COMMENT)
        if td is None:
            td = item.tag(FIFF_NAME)
        if td is None:
            raise ValueError("Projection item description missing")
        desc = f.string(td.__val__())
        kind = _tag_int(f, item, FIFF_PROJ_ITEM_KIND, -1)
        if kind < 0:
            raise ValueError("Projection item kind missing")
        nvec = _tag_int(f, item, FIFF_PROJ_ITEM_NVEC, -1)
        if nvec < 0:
            raise ValueError("Number of projection vectors not specified")
        tn = item.tag(FIFF_PROJ_ITEM_CH_NAME_LIST)
        if tn is None:
            raise ValueError("Projection item channel list missing")
        names = _read_name_list(f.string(tn.__val__()))
        tv = item.tag(FIFF_PROJ_ITEM_VECTORS)
        if tv is None:
            raise ValueError("Projection item data missing")
        data = _f_matrix_rows(f, tv.__val__())
        if data.shape[0] == len(names) and data.shape[1] == nvec:
            data = data.T.copy()
        if data.shape[1] != len(names):
            raise ValueError("Number of channel names does not match the size of data matrix")
        active = _tag_int(f, item, FIFF_MNE_PROJ_ITEM_ACTIVE, 0) != 0
        ev: Optional[float] = None
        te = item.tag(FIFF_MNE_ICA_PCA_EXPLAINED_VAR)
        if te is not None:
            ev = f.float(te.__val__())
        projs.append(Projection(desc, kind, active, names, data, ev))
    return projs


def write_proj(fname, projs, *, overwrite: bool = False, verbose = None):
    """mne.write_proj."""
    w = FifWriter()
    ps = List[Projection]()
    for p in projs:
        ps.append(p)
    _write_proj_block(w, ps)
    w.end_file(str(fname))


def read_proj(fname, verbose = None) -> List[Projection]:
    """mne.read_proj."""
    f = FifFile(str(fname))
    return _read_proj_node(f, f.tree)


# ---------------------------------------------------------------- covariance

def write_cov(fname, cov: Covariance, *, overwrite: bool = False, verbose = None):
    """mne.write_cov (mne.Covariance.save)."""
    w = FifWriter()
    w.start_block(FIFFB_MNE_COV)
    dim = len(cov.names)
    w.write_int(FIFF_MNE_COV_KIND, 1)
    w.write_int(FIFF_MNE_COV_DIM, dim)
    if cov.nfree > 0:
        w.write_int(FIFF_MNE_COV_NFREE, cov.nfree)
    if dim > 0:
        w.write_name_list(FIFF_MNE_ROW_NAMES, cov.names)
    if cov.diag:
        w.write_doubles(FIFF_MNE_COV_DIAG, np.diag(cov._C).copy())
    else:
        vals = List[float]()
        for i in range(dim):
            for j in range(i + 1):
                vals.append(cov._C[i, j])
        w.write_doubles(FIFF_MNE_COV, np.array(vals))
    if cov.eig is not None and cov.eigvec is not None:
        ev = cov.eigvec.__val__()
        w.write_float_matrix(FIFF_MNE_COV_EIGENVECTORS, ev.reshape(ev.shape[0] * ev.shape[1]),
                             [ev.shape[0], ev.shape[1]])
        w.write_doubles(FIFF_MNE_COV_EIGENVALUES, cov.eig.__val__())
    _write_proj_block(w, cov.projs)
    w.write_bad_channels(cov.bads)
    if cov.method is not None:
        w.write_string(FIFF_MNE_COV_METHOD, cov.method.__val__())
    if cov.loglik is not None:
        w.write_doubles(FIFF_MNE_COV_SCORE, np.array([cov.loglik.__val__()]))
    w.end_block(FIFFB_MNE_COV)
    w.end_file(str(fname))


def read_cov(fname, verbose = None) -> Covariance:
    """mne.read_cov (noise covariance, kind 1)."""
    f = FifFile(str(fname))
    for this in f.tree.find(FIFFB_MNE_COV):
        if _tag_int(f, this, FIFF_MNE_COV_KIND, -1) != 1:
            continue
        td = this.tag(FIFF_MNE_COV_DIM)
        if td is None:
            raise ValueError("Covariance matrix dimension not found")
        dim = f.int(td.__val__())
        nfree = _tag_int(f, this, FIFF_MNE_COV_NFREE, -1)
        method: Optional[str] = None
        tm = this.tag(FIFF_MNE_COV_METHOD)
        if tm is not None:
            method = f.string(tm.__val__())
        score: Optional[float] = None
        ts = this.tag(FIFF_MNE_COV_SCORE)
        if ts is not None:
            score = float(f.floats(ts.__val__())[0])
        names = List[str]()
        tr = this.tag(FIFF_MNE_ROW_NAMES)
        if tr is not None:
            names = _read_name_list(f.string(tr.__val__()))
            if len(names) != dim:
                raise ValueError("Number of names does not match covariance matrix dimension")
        tc = this.tag(FIFF_MNE_COV)
        C = np.zeros((dim, dim))
        diag = False
        if tc is not None:
            vals = f.floats(tc.__val__())
            k = 0
            for i in range(dim):
                for j in range(i + 1):
                    C[i, j] = vals[k]
                    C[j, i] = vals[k]
                    k += 1
        else:
            tg = this.tag(FIFF_MNE_COV_DIAG)
            if tg is None:
                raise ValueError("No covariance matrix data found")
            d = f.floats(tg.__val__())
            for i in range(dim):
                C[i, i] = d[i]
            diag = True
        eig: Optional[np.ndarray[float, 1]] = None
        eigvec: Optional[np.ndarray[float, 2]] = None
        t1 = this.tag(FIFF_MNE_COV_EIGENVALUES)
        t2 = this.tag(FIFF_MNE_COV_EIGENVECTORS)
        if t1 is not None and t2 is not None:
            eig = f.floats(t1.__val__())
            eigvec = _f_matrix_rows(f, t2.__val__())
        projs = _read_proj_node(f, this)
        bads = List[str]()
        for b in this.find(FIFFB_MNE_BAD_CHANNELS):
            tb = b.tag(FIFF_MNE_CH_NAME_LIST)
            if tb is not None:
                bads = _read_name_list(f.string(tb.__val__()))
        cov = Covariance(C, names, bads, projs, nfree, eig, eigvec, method, score)
        if diag:
            cov.diag = True
        return cov
    raise ValueError("No covariance matrix of kind 1 found")


# ---------------------------------------------------------------- source spaces
from .source_space import SourceSpace, SourceSpaces
from .io._fifext import (FIFFB_MNE_SOURCE_SPACE, FIFF_MNE_SOURCE_SPACE_ID, FIFF_MNE_SOURCE_SPACE_TYPE,
                         FIFF_MNE_COORD_FRAME, FIFF_MNE_SOURCE_SPACE_NPOINTS, FIFF_MNE_SOURCE_SPACE_POINTS,
                         FIFF_MNE_SOURCE_SPACE_NORMALS, FIFF_MNE_SOURCE_SPACE_SELECTION,
                         FIFF_MNE_SOURCE_SPACE_NUSE, FIFFV_MNE_SPACE_SURFACE, FIFFV_MNE_SPACE_DISCRETE,
                         FIFFV_MNE_SPACE_VOLUME, FIFFB_MNE, FIFFB_MNE_ENV, FIFFB_MNE_PARENT_MRI_FILE,
                         FIFF_MNE_FILE_NAME, FIFFB_MNE_FORWARD_SOLUTION, FIFFB_MNE_NAMED_MATRIX,
                         FIFF_MNE_NROW, FIFF_MNE_NCOL, FIFF_MNE_COL_NAMES, FIFF_MNE_FORWARD_SOLUTION,
                         FIFF_MNE_SOURCE_ORIENTATION, FIFF_MNE_INCLUDED_METHODS, FIFFV_MNE_MEG,
                         FIFFV_MNE_EEG, FIFF_COORD_TRANS)
from .io._fiff import FIFFB_CH_INFO


def _write_source_space_blocks(w: FifWriter, src: SourceSpaces):
    for s in src.spaces:
        w.start_block(FIFFB_MNE_SOURCE_SPACE)
        if s.type == "surf":
            w.write_int(FIFF_MNE_SOURCE_SPACE_TYPE, FIFFV_MNE_SPACE_SURFACE)
        elif s.type == "discrete":
            w.write_int(FIFF_MNE_SOURCE_SPACE_TYPE, FIFFV_MNE_SPACE_DISCRETE)
        else:
            raise NotImplementedError("write_source_spaces: only surface and discrete source spaces are supported")
        w.write_int(FIFF_MNE_COORD_FRAME, s.coord_frame)
        w.write_int(FIFF_MNE_SOURCE_SPACE_NPOINTS, s.np)
        w.write_float_matrix(FIFF_MNE_SOURCE_SPACE_POINTS, s.rr.reshape(s.np * 3), [s.np, 3])
        w.write_float_matrix(FIFF_MNE_SOURCE_SPACE_NORMALS, s.nn.reshape(s.np * 3), [s.np, 3])
        w.write_ints(FIFF_MNE_SOURCE_SPACE_SELECTION, [1 if s.inuse[i] else 0 for i in range(s.np)])
        w.write_int(FIFF_MNE_SOURCE_SPACE_NUSE, s.nuse)
        w.end_block(FIFFB_MNE_SOURCE_SPACE)


def _read_source_space_blocks(f: FifFile) -> SourceSpaces:
    spaces = List[SourceSpace]()
    for n in f.tree.find(FIFFB_MNE_SOURCE_SPACE):
        ty = _tag_int(f, n, FIFF_MNE_SOURCE_SPACE_TYPE, -1)
        if ty == FIFFV_MNE_SPACE_VOLUME:
            raise NotImplementedError("read_source_spaces: volume source spaces are not supported")
        if ty != FIFFV_MNE_SPACE_SURFACE and ty != FIFFV_MNE_SPACE_DISCRETE:
            raise ValueError("Unknown source space type")
        frame = _tag_int(f, n, FIFF_MNE_COORD_FRAME, 5)
        np_ = _tag_int(f, n, FIFF_MNE_SOURCE_SPACE_NPOINTS, -1)
        if np_ < 0:
            raise ValueError("Number of vertices not found")
        rr = _f_matrix_rows(f, n.tag(FIFF_MNE_SOURCE_SPACE_POINTS).__val__())
        nn = _f_matrix_rows(f, n.tag(FIFF_MNE_SOURCE_SPACE_NORMALS).__val__())
        sp = SourceSpace("surf" if ty == FIFFV_MNE_SPACE_SURFACE else "discrete", rr, nn, frame)
        tsel = n.tag(FIFF_MNE_SOURCE_SPACE_SELECTION)
        if tsel is not None:
            sel = f.ints(tsel.__val__())
            sp.inuse = np.array([v != 0 for v in sel])
            sp.vertno = np.array([i for i in range(np_) if sel[i] != 0], dtype=int)
            sp.nuse = len(sp.vertno)
        spaces.append(sp)
    if len(spaces) == 0:
        raise ValueError("No source spaces found")
    return SourceSpaces(spaces)


def write_source_spaces(fname, src: SourceSpaces, *, overwrite: bool = False, verbose = None):
    """mne.write_source_spaces (surface / discrete)."""
    w = FifWriter()
    w.start_block(FIFFB_MNE)
    _write_source_space_blocks(w, src)
    w.end_block(FIFFB_MNE)
    w.end_file(str(fname))


def read_source_spaces(fname, patch_stats: bool = False, verbose = None) -> SourceSpaces:
    """mne.read_source_spaces (surface / discrete)."""
    return _read_source_space_blocks(FifFile(str(fname)))


# ---------------------------------------------------------------- forward solution
from .forward import Forward


def write_forward_solution(fname, fwd: Forward, overwrite: bool = False, verbose = None):
    """mne.write_forward_solution (head-frame source spaces, free or fixed orientation)."""
    from .transforms import Transform
    w = FifWriter()
    w.start_block(FIFFB_MNE)
    w.start_block(FIFFB_MNE_ENV)
    w.write_id(FIFF_BLOCK_ID_K)
    w.end_block(FIFFB_MNE_ENV)
    w.start_block(FIFFB_MNE_PARENT_MRI_FILE)
    w.write_string(FIFF_MNE_FILE_NAME, "")
    w.write_coord_trans(4, 5, np.eye(4))
    w.end_block(FIFFB_MNE_PARENT_MRI_FILE)
    # measurement info of the forward
    w.start_block(FIFFB_MNE_PARENT_MEAS_FILE_K)
    w.write_string(FIFF_MNE_FILE_NAME, "")
    w.write_int(FIFF_NCHAN, fwd.info.nchan)
    for k in range(fwd.info.nchan):
        w.write_ch_info(fwd.info, k)
    w.write_bad_channels(fwd.info.bads)
    w.end_block(FIFFB_MNE_PARENT_MEAS_FILE_K)
    _write_source_space_blocks(w, fwd.src)
    ori = fwd._orig_source_ori
    n_vert = 0
    for s in fwd.src.spaces:
        n_vert += s.nuse
    n_col = n_vert if ori == 1 else 3 * n_vert
    sol = fwd._orig_sol
    for pass_ in range(2):
        rows = List[int]()
        for i in range(len(fwd.ch_names)):
            ti = fwd.info.ch_names.index(fwd.ch_names[i])
            t = fwd.info.ch_types[ti]
            is_meg = t == "mag" or t == "grad"
            if (pass_ == 0 and is_meg) or (pass_ == 1 and t == "eeg"):
                rows.append(i)
        if len(rows) == 0:
            continue
        data = np.zeros((n_col, len(rows)))
        names = List[str]()
        for k in range(len(rows)):
            data[:, k] = sol[rows[k], :]
            names.append(fwd.ch_names[rows[k]])
        w.start_block(FIFFB_MNE_FORWARD_SOLUTION)
        w.write_int(FIFF_MNE_INCLUDED_METHODS, FIFFV_MNE_MEG if pass_ == 0 else FIFFV_MNE_EEG)
        w.write_int(FIFF_MNE_COORD_FRAME, fwd.coord_frame)
        w.write_int(FIFF_MNE_SOURCE_ORIENTATION, ori)
        w.write_int(FIFF_MNE_SOURCE_SPACE_NPOINTS, n_vert)
        w.write_int(FIFF_NCHAN, len(rows))
        w.start_block(FIFFB_MNE_NAMED_MATRIX)
        w.write_int(FIFF_MNE_NROW, n_col)
        w.write_int(FIFF_MNE_NCOL, len(rows))
        w.write_name_list(FIFF_MNE_COL_NAMES, names)
        w.write_float_matrix(FIFF_MNE_FORWARD_SOLUTION, data.reshape(n_col * len(rows)), [n_col, len(rows)])
        w.end_block(FIFFB_MNE_NAMED_MATRIX)
        w.end_block(FIFFB_MNE_FORWARD_SOLUTION)
    w.end_block(FIFFB_MNE)
    w.end_file(str(fname))


FIFF_BLOCK_ID_K = 103
FIFFB_MNE_PARENT_MEAS_FILE_K = 354


def read_forward_solution(fname, include = (), exclude = (), verbose = None) -> Forward:
    """mne.read_forward_solution for files written by write_forward_solution (and
    sphere/head-frame solutions with the same layout)."""
    f = FifFile(str(fname))
    src = _read_source_space_blocks(f)
    meas = f.tree.find(FIFFB_MNE_PARENT_MEAS_FILE_K)
    if len(meas) == 0:
        raise ValueError("Could not find parent measurement info")
    mn = meas[0]
    nchan = _tag_int(f, mn, FIFF_NCHAN, 0)
    chs = List[Tag]()
    for t in mn.entries:
        if t.kind == 203:
            chs.append(t)
    info = Info([str(i) for i in range(len(chs))], 1.0, ["misc" for _ in range(len(chs))])
    from .io._fiff import _read_ch_info
    for i in range(len(chs)):
        _read_ch_info(f, chs[i], info, i)
    for b in mn.find(FIFFB_MNE_BAD_CHANNELS):
        tb = b.tag(FIFF_MNE_CH_NAME_LIST)
        if tb is not None:
            info.bads = _read_name_list(f.string(tb.__val__()))
    blocks = f.tree.find(FIFFB_MNE_FORWARD_SOLUTION)
    if len(blocks) == 0:
        raise ValueError("No forward solutions found")
    ori = 2
    coord = 4
    mats = List[np.ndarray[float, 2]]()
    names = List[str]()
    for b in blocks:
        ori = _tag_int(f, b, FIFF_MNE_SOURCE_ORIENTATION, 2)
        coord = _tag_int(f, b, FIFF_MNE_COORD_FRAME, 4)
        for nm in b.find(FIFFB_MNE_NAMED_MATRIX):
            tm = nm.tag(FIFF_MNE_FORWARD_SOLUTION)
            if tm is None:
                continue
            d = _f_matrix_rows(f, tm.__val__())
            mats.append(d.T.copy())
            tc = nm.tag(FIFF_MNE_COL_NAMES)
            if tc is not None:
                for c in _read_name_list(f.string(tc.__val__())):
                    names.append(c)
    sol = np.concatenate(mats, axis=0)
    rl = List[np.ndarray[float, 2]]()
    nl = List[np.ndarray[float, 2]]()
    for s in src.spaces:
        rl.append(s.rr[s.vertno])
        nl.append(s.nn[s.vertno])
    rr = np.concatenate(rl)
    nn = np.concatenate(nl)
    keep = [i for i in range(info.nchan) if info.ch_names[i] in names]
    finfo = info._pick(keep)
    fwd = Forward(sol, names, src, finfo, rr)
    fwd.coord_frame = coord
    fwd.source_ori = ori
    fwd._orig_source_ori = ori
    fwd._orig_sol = sol.copy()
    fwd.nsource = rr.shape[0]
    if ori == 1:
        fwd.source_nn = nn
    return fwd
