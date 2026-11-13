# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Signal-space projection: Projection handling, make_projector, SSP
# computation (compute_proj_raw/epochs/evoked), average reference projector.

import numpy as np
from ._info import Info, Projection, _picks_to_idx
from ._containers import RawArray, EpochsArray, Evoked

FIFFV_PROJ_ITEM_FIELD: int = 1
FIFFV_PROJ_ITEM_EEG_AVREF: int = 10


def _proj_equal(a: Projection, b: Projection, check_active: bool = True) -> bool:
    if check_active and a.active != b.active:
        return False
    if a.kind != b.kind or a.desc != b.desc or a.col_names != b.col_names:
        return False
    if a.data.shape[0] != b.data.shape[0] or a.data.shape[1] != b.data.shape[1]:
        return False
    return bool((a.data == b.data).all())


def _uniquify_projs(projs: List[Projection], check_active: bool = True) -> List[Projection]:
    out = List[Projection]()
    for p in projs:
        dup = False
        for q in out:
            if _proj_equal(q, p, check_active):
                dup = True
                break
        if not dup:
            out.append(p)
    return out


def make_projector(projs, ch_names, bads = (), include_active: bool = True):
    """mne.make_projector: (proj, nproj, U) for the given channel names."""
    names = [str(c) for c in ch_names]
    nchan = len(names)
    if nchan == 0:
        raise ValueError("No channel names specified")
    bset = Set[str]([str(b) for b in bads])
    use = List[Projection]()
    for p in projs:
        if (not p.active) or include_active:
            use.append(p)
    nvec = 0
    for p in use:
        nvec += p.data.shape[0]
    if len(use) == 0:
        return np.eye(nchan), 0, np.zeros((nchan, 0))
    vecs = np.zeros((nchan, nvec))
    col = 0
    nonzero = 0
    for p in use:
        pidx = Dict[str, int]()
        for i in range(len(p.col_names)):
            pidx[p.col_names[i]] = i
        nrow = p.data.shape[0]
        for c in range(nchan):
            if names[c] not in bset and names[c] in pidx:
                vi = pidx[names[c]]
                for r in range(nrow):
                    vecs[c, col + r] = p.data[r, vi]
        for v in range(nrow):
            s = 0.0
            for c in range(nchan):
                s += vecs[c, col + v] ** 2
            psize = np.sqrt(s)
            if psize > 0:
                for c in range(nchan):
                    vecs[c, col + v] /= psize
                nonzero += 1
        col += nrow
    if nonzero == 0:
        return np.eye(nchan), 0, np.zeros((nchan, 0))
    U, S, _ = np.linalg.svd(vecs, full_matrices=False)
    nproj = 0
    for k in range(S.shape[0]):
        if S[k] / S[0] > 1e-2:
            nproj += 1
    U = U[:, :nproj].copy()
    proj = np.eye(nchan) - np.dot(U, U.T)
    if nproj >= nchan:
        raise RuntimeError(f"Application of {nproj} projectors for {nchan} channels will yield no components.")
    return proj, nproj, U


def make_projector_info(info: Info, include_active: bool = True):
    p, n, _ = make_projector(info.projs, info.ch_names, info.bads, include_active)
    return p, n


def _setup_proj(info: Info, activate: bool) -> Optional[np.ndarray[float, 2]]:
    """setup_proj(add_eeg_ref=False): returns the projector (None if no-op)."""
    if len(info.projs) == 0:
        return None
    p, n = make_projector_info(info, True)
    if n == 0:
        return None
    if activate:
        for q in info.projs:
            q.active = True
    return p


def make_eeg_average_ref_proj(info: Info, ch_type = "eeg", activate: bool = True, verbose = None) -> Projection:
    t = str(ch_type) if isinstance(ch_type, str) else str(list(ch_type)[0])
    if t not in ("eeg", "ecog", "seeg", "dbs"):
        raise ValueError(f"ch_type must be one of eeg, ecog, seeg, dbs, got {t}")
    names = List[str]()
    for i in range(info.nchan):
        if info.ch_types[i] == t and info.ch_names[i] not in info.bads:
            names.append(info.ch_names[i])
    if len(names) == 0:
        raise ValueError(f"Cannot create {t.upper()} average reference projector (no {t.upper()} data found)")
    n = len(names)
    vec = np.ones((1, n)) / np.sqrt(float(n))
    return Projection(f"Average {t.upper()} reference", FIFFV_PROJ_ITEM_EEG_AVREF, activate, names, vec, None)


def _has_eeg_average_ref_proj(info: Info) -> bool:
    for p in info.projs:
        if p.kind == FIFFV_PROJ_ITEM_EEG_AVREF:
            return True
    return False


def _compute_proj_from_cov(cov: np.ndarray[float, 2], info: Info, n_grad, n_mag, n_eeg,
                           desc_prefix: str, meg: str = "separate") -> List[Projection]:
    ng = float(n_grad)
    nm = float(n_mag)
    ne = float(n_eeg)
    for v in (ng, nm, ne):
        if v < 0:
            raise ValueError(f"Argument must be either a positive integer or a float between 0 and 1. '{v}' is invalid.")
    if meg != "separate" and meg != "combined":
        raise ValueError(f"Invalid value for the 'meg' parameter. Allowed values are 'separate' and 'combined', but got '{meg}' instead.")
    types = List[str]()
    nvec = List[float]()
    kinds = List[str]()
    if meg == "combined":
        if ng != nm:
            raise ValueError(f"n_grad ({n_grad}) must be equal to n_mag ({n_mag}) when using meg='combined'.")
        types = ["meg", "eeg"]
        nvec = [ng, ne]
        kinds = ["meg", "eeg"]
    else:
        types = ["grad", "mag", "eeg"]
        nvec = [ng, nm, ne]
        kinds = ["planar", "axial", "eeg"]
    projs = List[Projection]()
    for ti in range(len(types)):
        idx = List[int]()
        for i in range(info.nchan):
            if info.ch_names[i] in info.bads:
                continue
            t = info.ch_types[i]
            if t == types[ti] or (types[ti] == "meg" and (t == "mag" or t == "grad")):
                idx.append(i)
        if len(idx) == 0:
            continue
        ia = np.array(idx, dtype=int)
        sub = cov[ia][:, ia].copy()
        U, S, _ = np.linalg.svd(sub, full_matrices=False)
        tot = S.sum()
        var = S / tot
        nv = nvec[ti]
        k_use = int(nv)
        if nv > 0 and nv < 1:
            cs = np.cumsum(var)
            k_use = 1
            for j in range(cs.shape[0]):
                if cs[j] < nv:
                    k_use = j + 2
            if k_use > cs.shape[0]:
                k_use = cs.shape[0]
        k_use = min(k_use, U.shape[1])
        names = [info.ch_names[i] for i in idx]
        for k in range(k_use):
            vec = U[:, k].copy().reshape((1, len(idx)))
            desc = f"{kinds[ti]}-{desc_prefix}-PCA-{k + 1:02d}"
            projs.append(Projection(desc, FIFFV_PROJ_ITEM_FIELD, False, list(names), vec, float(var[k])))
    return projs


def compute_proj_evoked(evoked: Evoked, n_grad = 2, n_mag = 2, n_eeg = 2,
                        meg: str = "separate", verbose = None) -> List[Projection]:
    cov = np.dot(evoked.data, evoked.data.T)
    prefix = f"{evoked.times[0]:.3f}-{evoked.times[evoked.times.shape[0] - 1]:.3f}"
    return _compute_proj_from_cov(cov, evoked.info, n_grad, n_mag, n_eeg, prefix, meg)


def compute_proj_epochs(epochs: EpochsArray, n_grad = 2, n_mag = 2, n_eeg = 2,
                        n_jobs = None, desc_prefix: Optional[str] = None, meg: str = "separate",
                        verbose = None) -> List[Projection]:
    n_ep = epochs._data.shape[0]
    if n_ep == 0:
        raise RuntimeError("No good epochs found")
    nch = epochs._data.shape[1]
    cov = np.zeros((nch, nch))
    for i in range(n_ep):
        e = epochs._data[i]
        cov = cov + np.dot(e, e.T)
    if len(epochs.event_id) == 0:
        eid = "0"
    elif len(epochs.event_id) == 1:
        eid = str(list(epochs.event_id.values())[0])
    else:
        eid = "Multiple-events"
    prefix = f"{eid}-{epochs.times[0]:.3f}-{epochs.times[epochs.times.shape[0] - 1]:.3f}"
    if desc_prefix is not None:
        prefix = desc_prefix.__val__()
    return _compute_proj_from_cov(cov, epochs.info, n_grad, n_mag, n_eeg, prefix, meg)


def compute_proj_raw(raw: RawArray, start: float = 0.0, stop: Optional[float] = None,
                     duration: Optional[float] = 1.0, n_grad = 2, n_mag = 2,
                     n_eeg = 0, reject = None, flat = None, n_jobs = None,
                     meg: str = "separate", verbose = None) -> List[Projection]:
    from .epochs import Epochs
    from .event import make_fixed_length_events
    sf = raw.info.sfreq
    if duration is not None:
        dur = np.round(duration.__val__() * sf) / sf
        events = make_fixed_length_events(raw, 999, start, stop, dur)
        picks = List[int]()
        for i in range(raw.info.nchan):
            t = raw.info.ch_types[i]
            if (t in ("mag", "grad", "eeg", "eog", "ecg", "emg")) and raw.info.ch_names[i] not in raw.info.bads:
                picks.append(i)
        ep = Epochs(raw, events, None, 0.0, dur - 1.0 / sf, None, picks, True, reject, flat, False)
        n_ep = ep._data.shape[0]
        if n_ep == 0:
            raise RuntimeError("No good epochs found")
        nch = ep._data.shape[1]
        cov = np.zeros((nch, nch))
        for i in range(n_ep):
            e = ep._data[i]
            cov = cov + np.dot(e, e.T)
        t_stop = raw._data.shape[1] / sf
        if stop is not None and stop.__val__() != 0.0:
            t_stop = stop.__val__()
        prefix = f"Raw-{start:.3f}-{t_stop:.3f}"
        return _compute_proj_from_cov(cov, ep.info, n_grad, n_mag, n_eeg, prefix, meg)
    n_t = raw._data.shape[1]
    s0 = max(int(round(start * sf)), 0)
    s1 = n_t
    if stop is not None and stop.__val__() != 0.0:
        s1 = min(int(round(stop.__val__() * sf)), n_t)
    d = raw._data[:, s0:s1]
    cov = np.dot(d, d.T)
    prefix = f"Raw-{s0 / sf:.3f}-{s1 / sf:.3f}"
    return _compute_proj_from_cov(cov, raw.info, n_grad, n_mag, n_eeg, prefix, meg)


def _add_projs(info: Info, projs, remove_existing: bool):
    new = List[Projection]()
    for p in projs:
        q = p.copy()
        q.active = False
        new.append(q)
    if remove_existing:
        for p in info.projs:
            if p.active:
                raise ValueError("Cannot remove projectors that have already been applied")
        info.projs = _uniquify_projs(new, False)
    else:
        allp = List[Projection]()
        for p in info.projs:
            allp.append(p)
        for p in new:
            allp.append(p)
        info.projs = _uniquify_projs(allp, False)


def _del_proj(info: Info, idx: int):
    n = len(info.projs)
    k = idx if idx >= 0 else idx + n
    if k < 0 or k >= n:
        raise IndexError("index out of range")
    p = info.projs[k]
    if p.active:
        for c in p.col_names:
            if c in info.ch_names:
                raise ValueError(f"Cannot remove projector that has already been applied, unless you first remove all channels it applies to. The problematic projector is: {p}")
    info.projs = [info.projs[i] for i in range(n) if i != k]


@extend
class RawArray:
    @property
    def proj(self) -> bool:
        if len(self.info.projs) == 0:
            return False
        for p in self.info.projs:
            if not p.active:
                return False
        return True

    def add_proj(self, projs, remove_existing: bool = False, verbose = None):
        if isinstance(projs, Projection):
            _add_projs(self.info, [projs], remove_existing)
        else:
            _add_projs(self.info, projs, remove_existing)
        return self

    def del_proj(self, idx = "all"):
        if isinstance(idx, str):
            self.info.projs = List[Projection]()
        elif isinstance(idx, int):
            _del_proj(self.info, idx)
        else:
            ids = sorted([int(i) if int(i) >= 0 else int(i) + len(self.info.projs) for i in idx], reverse=True)
            for i in ids:
                _del_proj(self.info, i)
        return self

    def apply_proj(self, verbose = None):
        if len(self.info.projs) == 0 or self.proj:
            return self
        P = _setup_proj(self.info, True)
        if P is not None:
            self._data = np.dot(P.__val__(), self._data)
        return self


@extend
class Evoked:
    @property
    def proj(self) -> bool:
        if len(self.info.projs) == 0:
            return False
        for p in self.info.projs:
            if not p.active:
                return False
        return True

    def add_proj(self, projs, remove_existing: bool = False, verbose = None):
        if isinstance(projs, Projection):
            _add_projs(self.info, [projs], remove_existing)
        else:
            _add_projs(self.info, projs, remove_existing)
        return self

    def del_proj(self, idx = "all"):
        if isinstance(idx, str):
            self.info.projs = List[Projection]()
        elif isinstance(idx, int):
            _del_proj(self.info, idx)
        else:
            ids = sorted([int(i) if int(i) >= 0 else int(i) + len(self.info.projs) for i in idx], reverse=True)
            for i in ids:
                _del_proj(self.info, i)
        return self

    def apply_proj(self, verbose = None):
        if len(self.info.projs) == 0 or self.proj:
            return self
        P = _setup_proj(self.info, True)
        if P is not None:
            self.data = np.dot(P.__val__(), self.data)
        return self


@extend
class EpochsArray:
    @property
    def proj(self) -> bool:
        if len(self.info.projs) == 0:
            return False
        for p in self.info.projs:
            if not p.active:
                return False
        return True

    def add_proj(self, projs, remove_existing: bool = False, verbose = None):
        if isinstance(projs, Projection):
            _add_projs(self.info, [projs], remove_existing)
        else:
            _add_projs(self.info, projs, remove_existing)
        return self

    def del_proj(self, idx = "all"):
        if isinstance(idx, str):
            self.info.projs = List[Projection]()
        elif isinstance(idx, int):
            _del_proj(self.info, idx)
        else:
            ids = sorted([int(i) if int(i) >= 0 else int(i) + len(self.info.projs) for i in idx], reverse=True)
            for i in ids:
                _del_proj(self.info, i)
        return self

    def apply_proj(self, verbose = None):
        if len(self.info.projs) == 0 or self.proj:
            return self
        P = _setup_proj(self.info, True)
        if P is not None:
            Pm = P.__val__()
            for i in range(self._data.shape[0]):
                self._data[i] = np.dot(Pm, self._data[i])
        return self

    def set_eeg_reference(self, ref_channels = "average", projection: bool = False, ch_type = "eeg",
                          joint: bool = False, verbose = None):
        """Epochs re-referencing (average / channel list / projection)."""
        if projection:
            if not (isinstance(ref_channels, str) and ref_channels == "average"):
                raise ValueError("Setting projection=True is only supported for ref_channels=\"average\"")
            if not _has_eeg_average_ref_proj(self.info):
                self.add_proj(make_eeg_average_ref_proj(self.info, ch_type, False))
            return self
        eeg = [i for i in range(self.info.nchan)
               if self.info.ch_types[i] == "eeg" and self.info.ch_names[i] not in self.info.bads]
        refs = List[int]()
        if isinstance(ref_channels, str):
            if ref_channels == "average":
                refs = eeg
            else:
                refs = [self.info.ch_names.index(ref_channels)]
        else:
            for c in ref_channels:
                refs.append(self.info.ch_names.index(str(c)))
        if len(refs) == 0:
            return self
        ra = np.array(refs, dtype=int)
        for e in range(self._data.shape[0]):
            ref = self._data[e][ra, :].mean(axis=0)
            for i in eeg:
                self._data[e, i, :] -= ref
        return self
