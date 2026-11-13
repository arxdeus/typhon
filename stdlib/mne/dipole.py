# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.dipole: Dipole, DipoleFixed, read_dipole (text format), fit_dipole with a
# spherical conductor model (COBYLA, as mne) and make_forward_dipole.
# Deviations: no BEM surfaces (sphere models only), the device->head transform is the
# identity (coils are given in head coordinates), trans must be None.

import numpy as np
from scipy.optimize import fmin_cobyla
from scipy.linalg import pinvh
from ._info import Info
from ._containers import Evoked
from ._proj import make_projector
from .cov import Covariance, compute_whitener
from .bem import ConductorModel
from .source_space import SourceSpaces, setup_volume_source_space, _make_discrete_source_space
from .source_estimate import _STC
from .forward import (Forward, make_forward_solution, convert_forward_solution, _sphere_field, _eeg_spherepot,
                      _eeg_els, _Coils, _coil_def)

_CONF_KEYS: List[str] = ["vol", "depth", "long", "trans", "qlong", "qtrans"]


class Dipole:
    """mne.Dipole: sequence of single dipoles (positions in m, amplitude in Am, gof in %)."""
    times: np.ndarray[float, 1]
    pos: np.ndarray[float, 2]
    amplitude: np.ndarray[float, 1]
    ori: np.ndarray[float, 2]
    gof: np.ndarray[float, 1]
    name: Optional[str]
    conf: Dict[str, np.ndarray[float, 1]]
    khi2: np.ndarray[float, 1]
    nfree: np.ndarray[float, 1]
    has_stats: bool

    def __init__(self, times, pos, amplitude, ori, gof, name = None, conf = None, khi2 = None, nfree = None,
                 *, verbose = None):
        self.times = np.array(times, dtype=float)
        self.pos = np.array(pos, dtype=float)
        self.amplitude = np.array(amplitude, dtype=float)
        self.ori = np.array(ori, dtype=float)
        self.gof = np.array(gof, dtype=float)
        self.name = None if isinstance(name, NoneType) else str(name)
        self.conf = Dict[str, np.ndarray[float, 1]]()
        if not isinstance(conf, NoneType):
            for k, v in conf.items():
                self.conf[str(k)] = np.array(v, dtype=float)
        self.has_stats = not isinstance(khi2, NoneType)
        self.khi2 = np.array(khi2, dtype=float) if not isinstance(khi2, NoneType) else np.zeros(0)
        self.nfree = np.array(nfree, dtype=float) if not isinstance(nfree, NoneType) else np.zeros(0)

    def __len__(self) -> int:
        return self.pos.shape[0]

    def __repr__(self) -> str:
        return f"<Dipole | n_times : {self.times.shape[0]}, tmin : {np.min(self.times):0.3f}, tmax : {np.max(self.times):0.3f}>"

    def copy(self):
        return self[np.arange(self.times.shape[0])]

    def __getitem__(self, item):
        if isinstance(item, int):
            idx = np.array([item], dtype=int)
        elif isinstance(item, slice):
            idx = np.arange(self.times.shape[0])[item]
        else:
            idx = np.asarray(item)
            if idx.dtype == bool:
                idx = np.nonzero(idx)[0]
            idx = idx.astype(int)
        conf = Dict[str, np.ndarray[float, 1]]()
        for k, v in self.conf.items():
            conf[k] = v[idx]
        if self.has_stats:
            return Dipole(self.times[idx], self.pos[idx], self.amplitude[idx], self.ori[idx], self.gof[idx],
                          self.name, conf, self.khi2[idx], self.nfree[idx])
        return Dipole(self.times[idx], self.pos[idx], self.amplitude[idx], self.ori[idx], self.gof[idx],
                      self.name, conf)

    def crop(self, tmin = None, tmax = None, include_tmax: bool = True, verbose = None):
        from ._info import _time_mask
        sfreq: Optional[float] = None
        if self.times.shape[0] > 1:
            sfreq = 1.0 / float(np.median(np.diff(self.times)))
        mask = _time_mask(self.times, tmin, tmax, sfreq, True, include_tmax)
        self.times = self.times[mask]
        self.pos = self.pos[mask]
        self.amplitude = self.amplitude[mask]
        self.ori = self.ori[mask]
        self.gof = self.gof[mask]
        if self.has_stats:
            self.khi2 = self.khi2[mask]
            self.nfree = self.nfree[mask]
        for k in list(self.conf.keys()):
            self.conf[k] = self.conf[k][mask]
        return self

    def save(self, fname: str, overwrite: bool = False, *, verbose = None):
        """Write the text (.dip) format of mne's _write_dipole_text."""
        header = "#   begin     end   X (mm)   Y (mm)   Z (mm)   Q(nAm)  Qx(nAm)  Qy(nAm)  Qz(nAm)    g/%"
        n = self.times.shape[0]
        cols = List[str]()
        extra_fmt = List[Tuple[str, str, float, int]]()
        if self.has_stats:
            header += "    khi^2  free"
        for k, h, s in (("vol", "  vol/mm^3", 1e9), ("depth", "  depth/mm", 1e3), ("long", "  long/mm", 1e3),
                        ("trans", "  trans/mm", 1e3), ("qlong", "  Qlong/nAm", 1e9), ("qtrans", "  Qtrans/nAm", 1e9)):
            if k in self.conf:
                header += h
        lines = List[str]()
        lines.append('# CoordinateSystem "Head"')
        lines.append(header)
        for i in range(n):
            t = self.times[i] * 1000.0
            amp = 1e9 * self.amplitude[i]
            s = f"  {t:7.1f} {t:7.1f} {self.pos[i, 0] / 1e-3:8.2f} {self.pos[i, 1] / 1e-3:8.2f} {self.pos[i, 2] / 1e-3:8.2f} {amp:8.3f} {self.ori[i, 0] * amp:8.3f} {self.ori[i, 1] * amp:8.3f} {self.ori[i, 2] * amp:8.3f} {self.gof[i]:6.2f}"
            if self.has_stats:
                s += f" {self.khi2[i]:8.1f} {int(self.nfree[i]):5d}"
            if "vol" in self.conf:
                s += f" {self.conf['vol'][i] * 1e9:9.3f}"
            if "depth" in self.conf:
                s += f" {self.conf['depth'][i] * 1e3:9.3f}"
            if "long" in self.conf:
                s += f" {self.conf['long'][i] * 1e3:8.3f}"
            if "trans" in self.conf:
                s += f" {self.conf['trans'][i] * 1e3:9.3f}"
            if "qlong" in self.conf:
                s += f" {self.conf['qlong'][i] * 1e9:10.3f}"
            if "qtrans" in self.conf:
                s += f" {self.conf['qtrans'][i] * 1e9:11.3f}"
            lines.append(s)
        text = "\n".join(lines) + "\n"
        if self.name is not None:
            text += f'## Name "{self.name.__val__()} dipoles" Style "Dipoles"'
        with open(fname, "w") as f:
            f.write(text)


class DipoleFixed:
    """mne.DipoleFixed: fixed-position dipole fit (data rows: amplitude, goodness of fit)."""
    info: Info
    data: np.ndarray[float, 2]
    times: np.ndarray[float, 1]
    nave: int
    comment: str

    def __init__(self, info: Info, data, times, nave: int, aspect_kind: int = 100, comment: str = "", *,
                 verbose = None):
        self.info = info
        self.data = np.asarray(data, dtype=float)
        self.times = np.array(times, dtype=float)
        self.nave = nave
        self.comment = comment

    @property
    def ch_names(self) -> List[str]:
        return self.info.ch_names

    def copy(self):
        return DipoleFixed(self.info.copy(), self.data.copy(), self.times.copy(), self.nave, 100, self.comment)

    def __repr__(self) -> str:
        return f"<DipoleFixed | n_times : {self.times.shape[0]}, tmin : {np.min(self.times)}, tmax : {np.max(self.times)}>"


def _std_field(tok: str) -> str:
    return tok


def read_dipole(fname: str, verbose = None) -> Dipole:
    """mne.read_dipole (text .dip / .txt format)."""
    from .io._bytes import _read_text
    def_line = ""
    have_def = False
    need_header = True
    name: Optional[str] = None
    rows = List[List[float]]()
    for line in _read_text(fname).split("\n"):
        if len(line.strip()) == 0:
            continue
        if not (line.startswith("%") or line.startswith("#")):
            need_header = False
            rows.append([float(x) for x in line.strip().split()])
        else:
            if need_header:
                def_line = line
                have_def = True
            if line.startswith("##") or line.startswith("%%"):
                a = line.find('Name "')
                b = line.find(' dipoles"')
                if a >= 0 and b > a:
                    name = line[a + 6:b]
    if not have_def:
        raise OSError(f"Dipole text file is missing field definition comment, cannot parse {fname}")
    d = def_line.lstrip("%").lstrip("#").strip()
    toks = List[str]()
    for t in d.split():
        if t.startswith("(") and t.endswith(")") and len(toks) > 0:
            toks[len(toks) - 1] = toks[len(toks) - 1] + "/" + t[1:-1]
        elif "(" in t:
            toks.append(t.replace("(", "/").replace(")", ""))
        else:
            toks.append(t)
    fields = List[str]()
    for t in toks:
        tl = t.lower()
        if tl == "begin" or tl == "end":
            tl = tl + "/ms"
        fields.append(tl)
    required = ["begin/ms", "x/mm", "y/mm", "z/mm", "q/nam", "qx/nam", "qy/nam", "qz/nam", "g/%"]
    for r in required:
        if r not in fields:
            raise RuntimeError(f"Could not find necessary fields in header: {r}")
    n = len(rows)
    if n > 0 and len(fields) != len(rows[0]):
        raise OSError(f"More data fields ({len(fields)}) found than data columns ({len(rows[0])}): {fields}")
    def col(f: str) -> np.ndarray[float, 1]:
        k = fields.index(f)
        return np.array([rows[i][k] for i in range(n)])
    times = col("begin/ms") / 1000.0
    pos = np.zeros((n, 3))
    pos[:, 0] = 1e-3 * col("x/mm")
    pos[:, 1] = 1e-3 * col("y/mm")
    pos[:, 2] = 1e-3 * col("z/mm")
    amp = col("q/nam")
    norm = amp.copy()
    for i in range(n):
        if norm[i] == 0.0:
            norm[i] = 1.0
    amplitude = amp / 1e9
    ori = np.zeros((n, 3))
    ori[:, 0] = col("qx/nam") / norm
    ori[:, 1] = col("qy/nam") / norm
    ori[:, 2] = col("qz/nam") / norm
    gof = col("g/%")
    conf = Dict[str, np.ndarray[float, 1]]()
    for f, s in (("vol/mm^3", 1e-9), ("depth/mm", 1e-3), ("long/mm", 1e-3), ("trans/mm", 1e-3),
                 ("qlong/nam", 1e-9), ("qtrans/nam", 1e-9)):
        if f in fields:
            conf[f[:f.find("/")]] = s * col(f)
    if "khi^2" in fields and "free" in fields:
        return Dipole(times, pos, amplitude, ori, gof, name, conf, col("khi^2"), col("free"))
    return Dipole(times, pos, amplitude, ori, gof, name, conf)


# ---------------------------------------------------------------- sphere forward for fitting

def _coil_def_norm(ct: int):
    """coil_def.dat accuracy=1 ('normal') definitions."""
    if ct == 2000:
        return 1, np.array([1.0]), np.zeros((1, 3))
    if ct == 3012 or ct == 3013 or ct == 3014:
        rm = np.zeros((4, 3))
        w = np.array([29.7619, 29.7619, -29.7619, -29.7619])
        sx = [1.0, 1.0, -1.0, -1.0]
        sy = [1.0, -1.0, 1.0, -1.0]
        for k in range(4):
            rm[k, 0] = sx[k] * 8.4e-3
            rm[k, 1] = sy[k] * 6.713e-3
            rm[k, 2] = 3e-4
        return 3, w, rm
    if ct == 3022 or ct == 3023 or ct == 3024:
        a = 6.45e-3 if ct != 3024 else 5.25e-3
        rm = np.zeros((4, 3))
        sx = [-1.0, -1.0, 1.0, 1.0]
        sy = [-1.0, 1.0, -1.0, 1.0]
        for k in range(4):
            rm[k, 0] = sx[k] * a
            rm[k, 1] = sy[k] * a
            rm[k, 2] = 3e-4
        return 1, np.full(4, 0.25), rm
    raise RuntimeError(f"Desired coil definition not found (type = {ct} acc = 1)")


def _meg_coils_acc(info: Info, picks: List[int], accurate: bool) -> _Coils:
    rl = List[np.ndarray[float, 2]]()
    cl = List[np.ndarray[float, 2]]()
    wl = List[np.ndarray[float, 1]]()
    bl = List[int]()
    r0s = np.zeros((len(picks), 3))
    for k in range(len(picks)):
        loc = info.locs[picks[k]]
        T = np.eye(4)
        for i in range(3):
            T[i, 3] = loc[i]
        for c in range(3):
            for i in range(3):
                T[i, c] = loc[3 + 3 * c + i]
        ct = info.coil_types[picks[k]]
        if accurate:
            _, w, rm = _coil_def(ct)
        else:
            _, w, rm = _coil_def_norm(ct)
        R = T[:3, :3]
        rr = np.dot(rm, R.T) + T[:3, 3][None, :]
        cz = np.zeros((rm.shape[0], 3))
        cz[:, 2] = 1.0
        cc = np.dot(cz, R.T)
        rl.append(rr)
        cl.append(cc)
        wl.append(w)
        for _ in range(rm.shape[0]):
            bl.append(k)
        r0s[k] = T[:3, 3]
    return _Coils(np.concatenate(rl), np.concatenate(cl), np.concatenate(wl),
                  np.array(bl, dtype=int), len(picks), r0s)


class _Sens:
    """Prepared sensors + the sphere model for dipole forward fields."""
    meg_picks: List[int]
    eeg_picks: List[int]
    meg: Optional[_Coils]
    eeg: Optional[_Coils]
    sphere: ConductorModel
    r0: np.ndarray[float, 1]
    radius: float
    whitener: np.ndarray[float, 2]
    order: List[int]

    def __init__(self, sphere: ConductorModel, radius: float):
        self.meg_picks = List[int]()
        self.eeg_picks = List[int]()
        self.meg = None
        self.eeg = None
        self.sphere = sphere
        self.r0 = sphere.r0
        self.radius = radius
        self.whitener = np.zeros((0, 0))
        self.order = List[int]()

    def forwards(self, rr: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
        """(3 * n_rr, n_chan) fields in `order` (picks) channel order."""
        parts = List[np.ndarray[float, 2]]()
        if self.meg is not None:
            parts.append(_sphere_field(rr, self.meg.__val__(), self.r0))
        if self.eeg is not None:
            parts.append(_eeg_spherepot(rr, self.eeg.__val__(), self.sphere))
        return np.concatenate(parts, axis=1)


def _fwd_whitened(s: _Sens, rr: np.ndarray[float, 2]):
    B = s.forwards(rr)
    return np.dot(B, s.whitener.T), B


def _dipole_gof(sing, vv, B, B2: float):
    s0 = float(sing[0]) if sing[0] > 0 else 1.0
    ncomp = 3 if float(sing[2]) / s0 > 0.2 else 2
    one = np.dot(vv[:ncomp], B)
    Bm2 = float(np.dot(one, one))
    return Bm2 / B2, one


class _FitEval:
    s: _Sens
    B: np.ndarray[float, 1]
    B2: float

    def __init__(self, s: _Sens, B, B2: float):
        self.s = s
        self.B = B
        self.B2 = B2

    def __call__(self, x) -> float:
        rd = np.asarray(x, dtype=float)
        fwd, _ = _fwd_whitened(self.s, rd.reshape((1, 3)))
        u, sing, vv = np.linalg.svd(fwd, full_matrices=False)
        gof, _ = _dipole_gof(sing, vv, self.B, self.B2)
        return 1.0 - gof


class _SphereCons:
    r0: np.ndarray[float, 1]
    R_adj: float

    def __init__(self, r0, R_adj: float):
        self.r0 = np.asarray(r0, dtype=float)
        self.R_adj = R_adj

    def __call__(self, x) -> float:
        d = np.asarray(x, dtype=float) - self.r0
        return self.R_adj - float(np.sqrt(np.sum(d * d)))


def _fit_Q(s: _Sens, B, B2: float, B_orig, rd, ori: Optional[np.ndarray[float, 1]]):
    fwd, fwd_orig = _fwd_whitened(s, rd.reshape((1, 3)))
    Q = np.zeros(3)
    gof = 0.0
    ncomp = 3
    if ori is None:
        u, sing, vv = np.linalg.svd(fwd, full_matrices=False)
        g, one = _dipole_gof(sing, vv, B, B2)
        gof = g
        ncomp = one.shape[0]
        for i in range(ncomp):
            one[i] = one[i] / sing[i]
        Q = np.dot(one, u.T[:ncomp])
    else:
        o = ori.__val__()
        f1 = np.dot(o, fwd)
        sg = float(np.linalg.norm(f1))
        one1 = float(np.dot(f1 / sg, B))
        gof = one1 * one1 / B2
        Q = o * (one1 / sg)
    resid = B_orig - np.dot(fwd_orig.T, Q)
    return Q, gof, resid, ncomp


def _fit_confidence(s: _Sens, rd, Q, ori):
    direction = np.empty((3, 3))
    direction[0] = ori
    rvec = rd - s.r0
    direction[2] = rvec - ori * np.dot(ori, rvec)
    direction[2] = direction[2] / np.linalg.norm(direction[2])
    direction[1] = np.cross(direction[2], direction[0])
    nch = s.whitener.shape[0]
    J = np.empty((nch, 6))
    dl = [-1e-4, 1e-4]
    for ii in range(3):
        f0 = np.dot(Q, _fwd_whitened(s, (rd + dl[0] * direction[ii]).reshape((1, 3)))[0])
        f1 = np.dot(Q, _fwd_whitened(s, (rd + dl[1] * direction[ii]).reshape((1, 3)))[0])
        J[:, ii] = (f1 - f0) / (dl[1] - dl[0])
    qn = float(np.linalg.norm(Q))
    dq = [-0.01 * qn, 0.01 * qn]
    this_fwd = _fwd_whitened(s, rd.reshape((1, 3)))[0]
    for ii in range(3):
        f0 = np.dot(Q + dq[0] * direction[ii], this_fwd)
        f1 = np.dot(Q + dq[1] * direction[ii], this_fwd)
        J[:, ii + 3] = (f1 - f0) / (dq[1] - dq[0])
    direction_norm = float(np.linalg.norm(J[:, :3]))
    q_norm = float(np.linalg.norm(J[:, 3:5]))
    norm = np.array([direction_norm, direction_norm, direction_norm, q_norm, q_norm, q_norm])
    for c in range(6):
        J[:, c] = J[:, c] / norm[c]
    J = np.dot(J.T, J)
    C = pinvh(J, rtol=1e-14)
    for a in range(6):
        for b in range(6):
            C[a, b] = C[a, b] / norm[a] / norm[b]
    conf = np.empty(7)
    for i in range(6):
        conf[i] = 1.96 * np.sqrt(C[i, i])
    ev = np.linalg.eigvalsh(C[:3, :3])
    conf[6] = 4.0 * np.pi / 3.0 * np.sqrt(476.379541 * float(np.prod(ev)))
    return np.array([conf[6], conf[2], conf[0], conf[1], conf[3], conf[4]])


def _fit_dipole_fn(s: _Sens, guess_rrs, guess_svd_vv, guess_svd_s, min_dist_to_inner_skull: float,
                   B_orig, rank: int, ori: Optional[np.ndarray[float, 1]], rhoend: float):
    B = np.dot(s.whitener, B_orig)
    B2 = float(np.dot(B, B))
    if B2 == 0.0:
        return np.zeros(3), 0.0, np.zeros(3), 0.0, np.zeros(6), 0.0, 0, np.zeros(B_orig.shape[0]), True
    best = np.inf
    idx = 0
    for fi in range(guess_rrs.shape[0]):
        g, _ = _dipole_gof(guess_svd_s[fi], guess_svd_vv[fi], B, B2)
        f = 1.0 - g
        if f < best:
            best = f
            idx = fi
    x0 = guess_rrs[idx].copy()
    fun = _FitEval(s, B, B2)
    cons = _SphereCons(s.r0, s.radius - min_dist_to_inner_skull)
    rd_final = np.asarray(fmin_cobyla(fun, x0, (cons,), consargs=(), rhobeg=5e-2, rhoend=rhoend, disp=False),
                          dtype=float)
    Q, gof, resid, ncomp = _fit_Q(s, B, B2, B_orig, rd_final, ori)
    khi2 = (1.0 - gof) * B2
    nfree = rank - ncomp
    amp = float(np.sqrt(np.dot(Q, Q)))
    nrm = 1.0 if amp == 0.0 else amp
    o = Q / nrm
    conf = _fit_confidence(s, rd_final, Q, o)
    return rd_final, amp, o, gof, conf, khi2, nfree, resid, False


def _fit_dipole_fixed_fn(s: _Sens, rd, B_orig, ori: Optional[np.ndarray[float, 1]]):
    B = np.dot(s.whitener, B_orig)
    B2 = float(np.dot(B, B))
    if B2 == 0.0:
        return 0.0, np.zeros(3), 0.0, np.zeros(B_orig.shape[0])
    Q, gof, resid, ncomp = _fit_Q(s, B, B2, B_orig, rd, ori)
    amp = 0.0
    o = np.zeros(3)
    if ori is None:
        amp = float(np.sqrt(np.dot(Q, Q)))
        o = Q / (1.0 if amp == 0.0 else amp)
    else:
        o = ori.__val__().copy()
        amp = float(np.dot(Q, o))
    return amp, o, gof, resid


def fit_dipole(evoked: Evoked, cov: Covariance, bem: ConductorModel, trans = None, min_dist: float = 5.0,
               n_jobs = None, pos = None, ori = None, rank = None, accuracy: str = "normal", tol: float = 5e-5,
               verbose = None):
    """mne.fit_dipole with a spherical conductor model. Returns (Dipole | DipoleFixed, residual Evoked)."""
    if accuracy != "accurate" and accuracy != "normal":
        raise ValueError(f"accuracy must be 'accurate' or 'normal', got {accuracy}")
    ev = evoked.copy()
    if min_dist < 0:
        raise ValueError(f"min_dist should be positive. Got {min_dist}")
    if not isinstance(ori, NoneType) and isinstance(pos, NoneType):
        raise ValueError("pos must be provided if ori is not None")
    if not isinstance(trans, NoneType):
        raise NotImplementedError("fit_dipole: only trans=None is supported (sphere model in head coordinates)")
    if not np.isfinite(ev.data).all():
        raise ValueError("Evoked data must be finite")
    info = ev.info
    times = ev.times.copy()
    comment = ev.comment
    mdist = min_dist / 1000.0
    have_eeg = False
    have_meg = False
    for t in info.ch_types:
        if t == "eeg":
            have_eeg = True
        if t == "mag" or t == "grad":
            have_meg = True
    r0 = bem.r0.copy()
    R = 0.0
    if bem.nlayers > 0:
        R = bem.layer_rad[0]
    else:
        if have_eeg:
            raise RuntimeError("Spherical model has zero shells, cannot use with EEG data")
        dmin = np.inf
        for i in range(info.nchan):
            if info.ch_types[i] == "mag" or info.ch_types[i] == "grad":
                d = r0 - np.array([info.locs[i][0], info.locs[i][1], info.locs[i][2]])
                dmin = min(dmin, float(np.linalg.norm(d)))
        if not np.isfinite(dmin):
            raise RuntimeError("No MEG channels found, but MEG-only sphere model used")
        R = dmin
    fixed_position = not isinstance(pos, NoneType)
    pos_a = np.zeros(3)
    ori_a: Optional[np.ndarray[float, 1]] = None
    if not isinstance(pos, NoneType):
        pos_a = np.array(pos, dtype=float)
        if pos_a.shape[0] != 3:
            raise ValueError(f"pos must be None or a 3-element array-like, got {pos_a}")
        if not isinstance(ori, NoneType):
            o = np.array(ori, dtype=float)
            if o.shape[0] != 3:
                raise ValueError(f"oris must be None or a 3-element array-like, got {o}")
            nm = float(np.sqrt(np.sum(o * o)))
            if abs(nm - 1.0) > 1e-5 + 1e-8:
                raise ValueError(f"ori must be a unit vector, got length {nm}")
            ori_a = o
    if not have_eeg and not have_meg:
        raise RuntimeError("No MEG or EEG channels found.")
    # data channels (meg + eeg, no bads), as in pick_types(meg=True, eeg=True, ref_meg=False)
    picks = List[int]()
    for i in range(info.nchan):
        t = info.ch_types[i]
        if (t == "mag" or t == "grad" or t == "eeg") and info.ch_names[i] not in info.bads:
            picks.append(i)
    s = _Sens(bem, R)
    for p in picks:
        if info.ch_types[p] == "eeg":
            s.eeg_picks.append(p)
        else:
            s.meg_picks.append(p)
    # column order of the forward is meg then eeg: reorder channels the same way
    order = List[int]()
    for p in s.meg_picks:
        order.append(p)
    for p in s.eeg_picks:
        order.append(p)
    if len(s.meg_picks) > 0:
        s.meg = _meg_coils_acc(info, s.meg_picks, accuracy == "accurate")
    if len(s.eeg_picks) > 0:
        s.eeg = _eeg_els(info, s.eeg_picks)
    Wp, wnames, rk = compute_whitener(cov, info, picks, rank, return_rank=True)
    # whitener was computed for picks in info order; permute to forward order
    perm = [picks.index(p) for p in order]
    pa = np.array(perm, dtype=int)
    s.whitener = Wp[:, pa].copy()
    inner = abs(1)
    # guesses
    guess_rrs = np.zeros((1, 3))
    if fixed_position:
        chk = R - mdist - float(np.sqrt(np.sum((pos_a - r0) ** 2)))
        if chk <= 0:
            raise ValueError(f"fixed position is {-1000 * chk:0.1f}mm outside the inner skull boundary")
        guess_rrs = pos_a.reshape((1, 3))
    else:
        grid = 0.02
        gmind = max(0.005, mdist)
        gsrc = setup_volume_source_space(None, grid * 1000.0, None, (float(r0[0]), float(r0[1]), float(r0[2]), R),
                                         mindist=1000.0 * gmind, exclude=20.0)
        sp = gsrc.spaces[0]
        guess_rrs = sp.rr[sp.vertno].copy()
    ng = guess_rrs.shape[0]
    gB, _ = _fwd_whitened(s, guess_rrs)
    gvv = List[np.ndarray[float, 2]]()
    gs = List[np.ndarray[float, 1]]()
    for g in range(ng):
        u, sg, vv = np.linalg.svd(gB[3 * g:3 * g + 3], full_matrices=False)
        gvv.append(vv)
        gs.append(sg)
    # data
    data = ev.data[np.array(order, dtype=int)]
    ch_names = [info.ch_names[p] for p in order]
    proj_op = make_projector(info.projs, ch_names, info.bads)[0]
    nt = times.shape[0]
    resid = np.zeros((len(order), nt))
    pos_out = np.zeros((nt, 3))
    amp_out = np.zeros(nt)
    ori_out = np.zeros((nt, 3))
    gof_out = np.zeros(nt)
    khi2_out = np.zeros(nt)
    nfree_out = np.zeros(nt)
    conf_out = np.zeros((nt, 6))
    for t in range(nt):
        Bo = data[:, t].copy()
        if fixed_position:
            a, o2, g2, r2 = _fit_dipole_fixed_fn(s, guess_rrs[0], Bo, ori_a)
            pos_out[t] = guess_rrs[0]
            amp_out[t] = a
            ori_out[t] = o2
            gof_out[t] = g2 * 100.0
            resid[:, t] = r2
        else:
            rd, a, o2, g2, cf, k2, nf, r2, zero = _fit_dipole_fn(s, guess_rrs, gvv, gs, mdist, Bo, rk, ori_a, tol)
            pos_out[t] = rd
            amp_out[t] = a
            ori_out[t] = o2
            gof_out[t] = g2 * 100.0
            conf_out[t] = cf
            khi2_out[t] = k2
            nfree_out[t] = float(nf)
            resid[:, t] = r2
    residual = ev.copy()
    residual.apply_proj()
    rows = np.array(order, dtype=int)
    rp = np.dot(proj_op, resid)
    for k in range(len(order)):
        residual.data[order[k], :] = rp[k, :]
    if isinstance(pos, NoneType) or isinstance(ori, NoneType):
        conf = Dict[str, np.ndarray[float, 1]]()
        if not fixed_position:
            for ki in range(6):
                conf[_CONF_KEYS[ki]] = conf_out[:, ki].copy()
            dip = Dipole(times, pos_out, amp_out, ori_out, gof_out, comment, conf, khi2_out, nfree_out)
            return dip, residual
        dip2 = Dipole(times, pos_out, amp_out, ori_out, gof_out, comment, None)
        return dip2, residual
    else:
        dd = np.zeros((2, nt))
        dd[0] = amp_out
        dd[1] = gof_out
        oinfo = Info(["dip 01", "goodness"], info.sfreq, ["dipole", "gof"])
        oa = ori_a.__val__()
        oinfo.locs[0] = [float(pos_a[0]), float(pos_a[1]), float(pos_a[2]), float(oa[0]),
                         float(oa[1]), float(oa[2]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return DipoleFixed(oinfo, dd, times, ev.nave, 100, comment), residual


def make_forward_dipole(dipole: Dipole, bem: ConductorModel, info: Info, trans = None, n_jobs = None,
                        *, on_inside: str = "raise", verbose = None):
    """mne.make_forward_dipole (sphere model): returns (fwd, list of stc | stc)."""
    times = dipole.times.copy()
    pos = dipole.pos.copy()
    amplitude = dipole.amplitude.copy()
    ori = dipole.ori.copy()
    src = SourceSpaces([_make_discrete_source_space(pos, ori)])
    fwd = make_forward_solution(info, trans, src, bem)
    fwd = convert_forward_solution(fwd, surf_ori=False, force_fixed=True, copy=False, use_cps=False)
    if fwd.src.spaces[0].nuse != pos.shape[0]:
        raise ValueError("One or more dipoles outside the inner skull.")
    tps = np.unique(times)
    tstep = 0.001
    if tps.shape[0] > 1:
        td = np.diff(tps)
        if not np.allclose(td, td[0]):
            tstep = -1.0
        else:
            tstep = float(td[0])
    data = np.zeros((amplitude.shape[0], tps.shape[0]))
    row = 0
    for tp in range(tps.shape[0]):
        sel = [i for i in range(times.shape[0]) if times[i] == tps[tp]]
        for k in range(len(sel)):
            data[row + k, tp] = amplitude[sel[k]]
        row += len(sel)
    vert = fwd.src.spaces[0].vertno
    if tstep <= 0:
        raise NotImplementedError("make_forward_dipole: unevenly spaced dipole times (a list of STCs) are not supported")
    return fwd, _STC[np.ndarray[float, 2]](data, [vert], float(tps[0]), tstep, None, "volume")
