# Ported from mne 1.13.2 (BSD-3-Clause)
#
# ICA.save / read_ica: FIF round trip of a fitted ICA solution (-ica.fif).

import numpy as np
from ..io._fiff import (FifFile, FifWriter, read_meas_info, FIFFB_MEAS, FIFF_BLOCK_ID, FIFFT_DOUBLE,
                        FIFFT_MATRIX)
from .._info import Info
from .ica import ICA

FIFFB_MNE_ICA = 374
FIFF_MNE_ROW_NAMES = 3502
FIFF_MNE_ICA_INTERFACE_PARAMS = 3601
FIFF_MNE_ICA_WHITENER = 3603
FIFF_MNE_ICA_PCA_COMPONENTS = 3604
FIFF_MNE_ICA_PCA_EXPLAINED_VAR = 3605
FIFF_MNE_ICA_PCA_MEAN = 3606
FIFF_MNE_ICA_MATRIX = 3607
FIFF_MNE_ICA_BADS = 3608
FIFF_MNE_ICA_MISC_PARAMS = 3609
FIFFB_MNE_ICA_OLD = 123


def _write_dmat(w: FifWriter, kind: int, flat: np.ndarray[float, 1], dims: List[int]):
    nd = len(dims)
    w._head(kind, FIFFT_DOUBLE | FIFFT_MATRIX, 8 * flat.shape[0] + 4 * (nd + 1))
    for i in range(flat.shape[0]):
        w.w.f64(flat[i])
    for k in range(nd):
        w.w.i32(dims[nd - 1 - k])
    w.w.i32(nd)


def _json_dict_ints(d: Dict[str, List[int]]) -> str:
    parts = List[str]()
    for k in d:
        vs = ", ".join([str(v) for v in d[k]])
        parts.append('"' + k + '": [' + vs + "]")
    return "{" + ", ".join(parts) + "}"


def _json_dict_nums(d: Dict[str, float]) -> str:
    parts = List[str]()
    for k in d:
        v = d[k]
        s = ""
        if k == "extended" or k == "ortho":
            s = "true" if v != 0.0 else "false"
        elif v == float(int(v)):
            s = str(int(v))
        else:
            s = str(v)
        parts.append('"' + k + '": ' + s)
    return "{" + ", ".join(parts) + "}"


def _ica_save(ica: ICA, fname: str, overwrite: bool = False):
    if not (fname.endswith("-ica.fif") or fname.endswith("_ica.fif")):
        raise OSError("ICA filename should end with -ica.fif or _ica.fif")
    if not overwrite:
        import os
        if os.path.exists(fname):
            raise FileExistsError(f"Destination file exists. Please use option 'overwrite=True' to force overwriting.")
    w = FifWriter()
    w.start_block(FIFFB_MEAS)
    w.write_id(FIFF_BLOCK_ID)
    w.write_meas_info(ica.info)
    w.end_block(FIFFB_MEAS)
    w.start_block(FIFFB_MNE_ICA)
    nc = "null"
    if ica.n_components is not None:
        v = ica.n_components.__val__()
        if ica.n_components_int:
            nc = str(int(v))
        else:
            nc = str(v)
    npc = "null"
    if ica.n_pca_components is not None:
        v2 = ica.n_pca_components.__val__()
        npc = str(int(v2)) if v2 >= 1.0 else str(v2)
    init = ("noise_cov:null;n_components:" + nc + ";n_pca_components:" + npc +
            ";max_pca_components:null;current_fit:\"" + ica.current_fit + "\";allow_ref_meg:0")
    w.write_string(FIFF_MNE_ICA_INTERFACE_PARAMS, init)
    w.write_string(FIFF_MNE_ROW_NAMES, ":".join(ica.ch_names))
    misc = ("n_samples_:" + str(ica.n_samples_) + ";labels_:" + _json_dict_ints(ica.labels_) +
            ";method:\"" + ica.method + "\";n_iter_:" + str(ica.n_iter_) +
            ";fit_params:" + _json_dict_nums(ica.fit_params))
    w.write_string(FIFF_MNE_ICA_MISC_PARAMS, misc)
    pw = ica.pre_whitener_
    _write_dmat(w, FIFF_MNE_ICA_WHITENER, pw.ravel(), [pw.shape[0], pw.shape[1]])
    pc = ica.pca_components_
    _write_dmat(w, FIFF_MNE_ICA_PCA_COMPONENTS, pc.ravel(), [pc.shape[0], pc.shape[1]])
    _write_dmat(w, FIFF_MNE_ICA_PCA_MEAN, ica.pca_mean_, [ica.pca_mean_.shape[0]])
    _write_dmat(w, FIFF_MNE_ICA_PCA_EXPLAINED_VAR, ica.pca_explained_variance_,
                [ica.pca_explained_variance_.shape[0]])
    um = ica.unmixing_matrix_
    _write_dmat(w, FIFF_MNE_ICA_MATRIX, um.ravel(), [um.shape[0], um.shape[1]])
    w.write_ints(FIFF_MNE_ICA_BADS, list(ica.exclude))
    w.end_block(FIFFB_MNE_ICA)
    w.end_file(fname)


def _split_top(s: str) -> List[str]:
    """Split on ';' outside braces/brackets/quotes."""
    out = List[str]()
    depth = 0
    inq = False
    cur = ""
    for i in range(len(s)):
        c = s[i]
        if c == '"':
            inq = not inq
        if not inq:
            if c == "{" or c == "[":
                depth += 1
            elif c == "}" or c == "]":
                depth -= 1
        if c == ";" and depth == 0 and not inq:
            out.append(cur)
            cur = ""
        else:
            cur += c
    out.append(cur)
    return out


def _kv(s: str) -> Tuple[str, str]:
    j = s.find(":")
    return s[:j], s[j + 1:]


def _unq(s: str) -> str:
    t = s.strip()
    if len(t) >= 2 and t[0] == '"':
        return t[1:len(t) - 1]
    return t


def _parse_ints_dict(s: str) -> Dict[str, List[int]]:
    out = Dict[str, List[int]]()
    t = s.strip()
    if t == "null" or t == "{}":
        return out
    body = t[1:len(t) - 1]
    i = 0
    n = len(body)
    while i < n:
        q0 = body.find('"', i)
        if q0 < 0:
            break
        q1 = body.find('"', q0 + 1)
        key = body[q0 + 1:q1]
        b0 = body.find("[", q1)
        b1 = body.find("]", b0)
        inner = body[b0 + 1:b1].strip()
        vals = List[int]()
        if len(inner) > 0:
            for p in inner.split(","):
                vals.append(int(p.strip()))
        out[key] = vals
        i = b1 + 1
    return out


def _parse_nums_dict(s: str) -> Dict[str, float]:
    out = Dict[str, float]()
    t = s.strip()
    if t == "null" or t == "{}":
        return out
    body = t[1:len(t) - 1]
    for part in body.split(","):
        if ":" not in part:
            continue
        k, v = _kv(part)
        key = _unq(k)
        vs = v.strip()
        if vs == "true":
            out[key] = 1.0
        elif vs == "false":
            out[key] = 0.0
        elif vs != "null":
            out[key] = float(vs)
    return out


def read_ica(fname, verbose = None) -> ICA:
    """Restore an ICA solution from a fif file (mne 1.13.2)."""
    fn = str(fname)
    if not (fn.endswith("-ica.fif") or fn.endswith("-ica.fif.gz") or fn.endswith("_ica.fif")
            or fn.endswith("_ica.fif.gz")):
        raise OSError("ICA filename should end with -ica.fif, -ica.fif.gz, _ica.fif or _ica.fif.gz")
    f = FifFile(fn)
    info = read_meas_info(f)
    nodes = f.tree.find(FIFFB_MNE_ICA)
    if len(nodes) == 0:
        nodes = f.tree.find(FIFFB_MNE_ICA_OLD)
        if len(nodes) == 0:
            raise ValueError("Could not find ICA data")
    nd = nodes[0]
    init = ""
    misc = ""
    ch_names = List[str]()
    pw = np.zeros(0)
    pwd = List[int]()
    pc = np.zeros(0)
    pcd = List[int]()
    pe = np.zeros(0)
    pm = np.zeros(0)
    um = np.zeros(0)
    umd = List[int]()
    exclude = List[int]()
    for t in nd.entries:
        if t.kind == FIFF_MNE_ICA_INTERFACE_PARAMS:
            init = f.string(t)
        elif t.kind == FIFF_MNE_ROW_NAMES:
            ch_names = f.string(t).split(":")
        elif t.kind == FIFF_MNE_ICA_WHITENER:
            pw, pwd = f.matrix(t)
        elif t.kind == FIFF_MNE_ICA_PCA_COMPONENTS:
            pc, pcd = f.matrix(t)
        elif t.kind == FIFF_MNE_ICA_PCA_EXPLAINED_VAR:
            pe, d1 = f.matrix(t)
        elif t.kind == FIFF_MNE_ICA_PCA_MEAN:
            pm, d2 = f.matrix(t)
        elif t.kind == FIFF_MNE_ICA_MATRIX:
            um, umd = f.matrix(t)
        elif t.kind == FIFF_MNE_ICA_BADS:
            exclude = f.ints(t)
        elif t.kind == FIFF_MNE_ICA_MISC_PARAMS:
            misc = f.string(t)
    n_comp: Optional[float] = None
    n_comp_int = True
    n_pca: Optional[float] = None
    current_fit = "unfitted"
    for part in _split_top(init):
        k, v = _kv(part)
        vv = v.strip()
        if k == "n_components":
            if vv != "null":
                if "." in vv:
                    n_comp = float(vv)
                    n_comp_int = False
                else:
                    n_comp = float(int(vv))
        elif k == "n_pca_components":
            if vv != "null":
                n_pca = float(vv)
        elif k == "current_fit":
            current_fit = _unq(vv)
    method = "fastica"
    n_samples = 0
    n_iter = 0
    labels = Dict[str, List[int]]()
    fit_params = Dict[str, float]()
    for part in _split_top(misc):
        k, v = _kv(part)
        vv = v.strip()
        if k == "method":
            if vv != "null":
                method = _unq(vv)
        elif k == "n_samples_":
            if vv != "null":
                n_samples = int(vv)
        elif k == "n_iter_":
            if vv != "null":
                n_iter = int(vv)
        elif k == "labels_":
            labels = _parse_ints_dict(vv)
        elif k == "fit_params":
            fit_params = _parse_nums_dict(vv)
    ica = ICA(None, method = method)
    ica.current_fit = current_fit
    ica.n_components = n_comp
    ica.n_components_int = n_comp_int
    ica.n_pca_components = n_pca
    ica.ch_names = ch_names
    ica.pre_whitener_ = pw.reshape((pwd[0], pwd[1])).copy()
    ica.pca_components_ = pc.reshape((pcd[0], pcd[1])).copy()
    ica.pca_mean_ = pm.copy()
    ica.pca_explained_variance_ = pe.copy()
    ica.unmixing_matrix_ = um.reshape((umd[0], umd[1])).copy()
    ica.n_components_ = umd[0]
    import scipy.linalg as sl
    ica.mixing_matrix_ = sl.pinv(ica.unmixing_matrix_)
    ica.exclude = exclude
    ica.info = info
    ica.n_samples_ = n_samples
    ica.n_iter_ = n_iter
    ica.labels_ = labels
    ica.fit_params = fit_params
    return ica


@extend
class ICA:
    def save(self, fname, *, overwrite: bool = False, verbose = None):
        """Store the ICA solution into a fif file (-ica.fif)."""
        _ica_save(self, str(fname), overwrite)
        return self
