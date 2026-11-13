# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.utils.config (get_config / set_config / get_config_path), read_reject_parameters,
# read_vectorview_selection, read_freesurfer_lut, match_channel_orders, grade_to_vertices.

import os
import json
import numpy as np
from ._info import Info
from .io._bytes import _read_text

_KNOWN_CONFIG: List[str] = [
    "MNE_BROWSER_BACKEND", "MNE_BROWSER_OVERVIEW_MODE", "MNE_BROWSER_PRECOMPUTE",
    "MNE_BROWSER_THEME", "MNE_BROWSER_USE_OPENGL", "MNE_CACHE_DIR", "MNE_COREG_COPY_ANNOT",
    "MNE_COREG_FULLSCREEN", "MNE_COREG_HEAD_HIGH_RES", "MNE_COREG_HEAD_OPACITY",
    "MNE_COREG_INTERACTION", "MNE_COREG_MARK_INSIDE", "MNE_COREG_PREPARE_BEM",
    "MNE_COREG_PROJECT_EEG", "MNE_COREG_ORIENT_TO_SURFACE", "MNE_COREG_SCALE_LABELS",
    "MNE_COREG_SCALE_BY_DISTANCE", "MNE_COREG_SCENE_SCALE", "MNE_COREG_WINDOW_HEIGHT",
    "MNE_COREG_WINDOW_WIDTH", "MNE_COREG_SUBJECTS_DIR", "MNE_CUDA_DEVICE", "MNE_DATA",
    "MNE_DATASETS_BRAINSTORM_PATH", "MNE_DATASETS_EEGBCI_PATH", "MNE_DATASETS_SAMPLE_PATH",
    "MNE_DATASETS_SOMATO_PATH", "MNE_DATASETS_SPM_FACE_PATH", "MNE_DATASETS_TESTING_PATH",
    "MNE_FORCE_SERIAL", "MNE_LOGGING_LEVEL", "MNE_MEMMAP_MIN_SIZE", "MNE_SKIP_TESTING_DATASET_TESTS",
    "MNE_STIM_CHANNEL", "MNE_USE_CUDA", "MNE_USE_NUMBA", "SUBJECTS_DIR", "MNE_REPR_HTML",
    "MNE_3D_BACKEND", "MNE_3D_OPTION_ANTIALIAS", "MNE_3D_OPTION_DEPTH_PEELING",
    "MNE_3D_OPTION_MULTI_SAMPLES", "MNE_3D_OPTION_SMOOTH_SHADING", "MNE_3D_OPTION_THEME",
    "MNE_DONTWRITE_HOME", "MNE_DATA", "MNE_DATASETS_FNIRS_MOTOR_PATH",
]


def _extra_data_path(home_dir: Optional[str] = None) -> str:
    h = home_dir
    if h is None:
        e = os.environ.get("_MNE_FAKE_HOME_DIR")
        if e is not None:
            h = e
    if h is None:
        h = os.path.expanduser("~")
    return os.path.join(h, ".mne")


def get_config_path(home_dir = None) -> str:
    """mne.get_config_path: ~/.mne/mne-python.json."""
    hd: Optional[str] = None
    if not isinstance(home_dir, NoneType):
        hd = str(home_dir)
    return os.path.join(_extra_data_path(hd), "mne-python.json")


def _load_config(path: str, raise_error: bool = False) -> Dict[str, str]:
    cfg = Dict[str, str]()
    txt = _read_text(path)
    try:
        js = json.loads(txt)
        for k, v in js.items():
            cfg[k] = v.as_str()
    except ValueError:
        if raise_error:
            raise RuntimeError(f"The MNE-Python config file ({path}) is not a valid JSON file and might be corrupted")
    return cfg


def _all_config(home_dir: Optional[str], use_env: bool) -> Dict[str, str]:
    path = os.path.join(_extra_data_path(home_dir), "mne-python.json")
    cfg = Dict[str, str]()
    if os.path.isfile(path):
        cfg = _load_config(path)
    if use_env:
        names = Set[str](_KNOWN_CONFIG)
        for k in cfg.keys():
            names.add(k)
        for k in names:
            if k in os.environ:
                cfg[k] = os.environ[k]
    return cfg


def get_config(key = None, default = None, raise_error: bool = False, home_dir = None,
               use_env: bool = True):
    """mne.get_config. key=None gives the dict of all entries, '' the known keys,
    a string the value (or default)."""
    hd: Optional[str] = None
    if not isinstance(home_dir, NoneType):
        hd = str(home_dir)
    if isinstance(key, NoneType):
        return _all_config(hd, use_env)
    else:
        k = str(key)
        if use_env and k in os.environ:
            return Optional[str](os.environ[k])
        path = os.path.join(_extra_data_path(hd), "mne-python.json")
        cfg = Dict[str, str]()
        if os.path.isfile(path):
            cfg = _load_config(path)
        if k in cfg:
            return Optional[str](cfg[k])
        if raise_error:
            raise KeyError(f'Key "{k}" not found in the environment or in the mne-python config file ({path}).')
        if isinstance(default, NoneType):
            return Optional[str](None)
        return Optional[str](str(default))


def set_config(key: str, value = None, home_dir = None, set_env: bool = True):
    """mne.set_config: write key (or delete it when value is None) in the json config."""
    hd: Optional[str] = None
    if not isinstance(home_dir, NoneType):
        hd = str(home_dir)
    path = os.path.join(_extra_data_path(hd), "mne-python.json")
    cfg = Dict[str, str]()
    if os.path.isfile(path):
        cfg = _load_config(path, True)
    if isinstance(value, NoneType):
        if key in cfg:
            del cfg[key]
    else:
        cfg[key] = str(value)
        if set_env:
            os.environ[key] = str(value)
    d = _extra_data_path(hd)
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    keys = sorted(list(cfg.keys()))
    parts = List[str]()
    for k in keys:
        parts.append("    " + json.dumps(k) + ": " + json.dumps(cfg[k]))
    body = "{\n" + ",\n".join(parts) + "\n}" if len(parts) > 0 else "{}"
    with open(path, "w") as f:
        f.write(body)


def read_reject_parameters(fname: str) -> Dict[str, float]:
    """mne.read_reject_parameters: gradReject/magReject/... from a .cov/.ave file."""
    names = ["gradReject", "magReject", "eegReject", "eogReject", "ecgReject"]
    pynames = ["grad", "mag", "eeg", "eog", "ecg"]
    out = Dict[str, float]()
    for line in _read_text(fname).split("\n"):
        words = line.split()
        if len(words) == 0:
            continue
        if words[0] in names:
            out[pynames[names.index(words[0])]] = float(words[1])
    return out


def _data_dir() -> str:
    f = __file__
    return f[:f.rfind("/") + 1] + "data/"


def read_vectorview_selection(name, fname = None, info: Optional[Info] = None, verbose = None) -> List[str]:
    """mne.read_vectorview_selection: channel names of the named Vectorview selection(s)."""
    nm = List[str]()
    if isinstance(name, str):
        nm.append(name)
    else:
        for n in name:
            nm.append(str(n))
    spacing = "old"
    if info is not None:
        inf = info.__val__()
        for i in range(inf.nchan):
            if inf.ch_types[i] == "mag" or inf.ch_types[i] == "grad":
                if " " not in inf.ch_names[i]:
                    spacing = "new"
                break
    path = _data_dir() + "mne_analyze.sel"
    if not isinstance(fname, NoneType):
        path = str(fname)
    found = Dict[str, bool]()
    for n in nm:
        found[n] = False
    sel = List[str]()
    for line0 in _read_text(path).split("\n"):
        line = line0.strip()
        if len(line) == 0 or line[0] == "#":
            continue
        pos = line.find(":")
        if pos < 0:
            continue
        sel_name = line[:pos]
        for n in nm:
            if sel_name.find(n) >= 0:
                for c in line[pos + 1:].split("|"):
                    sel.append(c)
                found[n] = True
                break
    for n in nm:
        if not found[n]:
            raise ValueError(f'No match for selection name "{n}" found')
    sel = sorted(list(Set[str](sel)))
    if spacing == "new":
        sel = [s.replace("MEG ", "MEG") for s in sel]
    return sel


def read_freesurfer_lut(fname = None):
    """mne.read_freesurfer_lut: (name -> id, name -> RGBA colors)."""
    path = _data_dir() + "FreeSurferColorLUT.txt"
    if not isinstance(fname, NoneType):
        path = str(fname)
    ids = Dict[str, int]()
    colors = Dict[str, np.ndarray[float, 1]]()
    for line0 in _read_text(path).split("\n"):
        line = line0.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        p = line.split()
        if len(p) != 6:
            raise RuntimeError(f"LUT is improperly formatted: {path}")
        ids[p[1]] = int(p[0])
        colors[p[1]] = np.array([float(p[2]), float(p[3]), float(p[4]), float(p[5])])
    return ids, colors


def match_channel_orders(insts, copy: bool = True):
    """mne.io.match_channel_orders: reorder all instances like the first."""
    out = [i.copy() if copy else i for i in insts]
    order = list(out[0].ch_names)
    for k in range(1, len(out)):
        out[k].reorder_channels(order)
    return out


def grade_to_vertices(subject: str, grade, subjects_dir = None, n_jobs = None, verbose = None):
    """mne.grade_to_vertices: fsaverage grade 5 (or an explicit vertex list) only,
    since the FreeSurfer spheres are not shipped."""
    if isinstance(grade, int):
        if subject == "fsaverage" and grade == 5:
            return [np.arange(10242), np.arange(10242)]
        raise NotImplementedError("grade_to_vertices needs the subject's sphere.reg surfaces (only fsaverage grade 5 is built in)")
    else:
        return [np.asarray(grade[0]).astype(int), np.asarray(grade[1]).astype(int)]


def _known_config_types() -> Dict[str, str]:
    """mne.get_config('') : the valid keys (values are the type names)."""
    out = Dict[str, str]()
    for n in _KNOWN_CONFIG:
        out[n] = "str"
    return out
