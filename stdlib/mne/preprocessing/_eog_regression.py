# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.preprocessing.EOGRegression for Raw, Epochs and Evoked (fit / apply with
# the average-reference and projection checks of mne). read_eog_regression
# (h5io) is not supported.

import numpy as np
from .._info import Info, _picks_to_idx
from .._containers import RawArray, EpochsArray, Evoked
from .._proj import _has_eeg_average_ref_proj


def _arr3(inst):
    """View of the instance data as (n_epochs, n_channels, n_times)."""
    if isinstance(inst, RawArray):
        return inst._data.reshape((1, inst._data.shape[0], inst._data.shape[1]))
    elif isinstance(inst, Evoked):
        return inst.data.reshape((1, inst.data.shape[0], inst.data.shape[1]))
    else:
        return inst._data


def _is_proj(inst) -> bool:
    return inst.proj


class EOGRegression:
    """Remove EOG artifacts by regression (mne.preprocessing.EOGRegression)."""
    picks: Optional[List[int]]
    exclude: List[str]
    exclude_bads: bool
    picks_artifact: str
    picks_artifact_idx: Optional[List[int]]
    proj: bool
    coef_: np.ndarray[float, 2]
    info_: Info
    fitted: bool

    def __init__(self, picks = None, exclude = "bads", picks_artifact = "eog", proj: bool = True):
        self.picks = None
        if not isinstance(picks, NoneType):
            if isinstance(picks, str):
                compile_error("picks given as a string is not supported, pass indices")
            self.picks = [int(p) for p in picks]
        self.exclude = List[str]()
        self.exclude_bads = False
        if isinstance(exclude, str):
            self.exclude_bads = exclude == "bads"
        else:
            for e in exclude:
                self.exclude.append(str(e))
        self.picks_artifact = "eog"
        self.picks_artifact_idx = None
        if isinstance(picks_artifact, str):
            self.picks_artifact = picks_artifact
        else:
            self.picks_artifact_idx = [int(p) for p in picks_artifact]
        self.proj = proj
        self.coef_ = np.zeros((0, 0))
        self.info_ = Info(List[str](), 1.0, List[str]())
        self.fitted = False

    def _excl(self, info: Info) -> List[str]:
        e = list(self.exclude)
        if self.exclude_bads:
            for b in info.bads:
                e.append(b)
        return e

    def _sel(self, info: Info):
        if self.picks is None:
            p = _picks_to_idx(info, None, "data", self._excl(info))
        else:
            p = _picks_to_idx(info, self.picks.__val__(), "data", self._excl(info))
        if self.picks_artifact_idx is None:
            pa = _picks_to_idx(info, self.picks_artifact, "all", ())
        else:
            pa = _picks_to_idx(info, self.picks_artifact_idx.__val__(), "all", ())
        return p, pa

    def _check_inst(self, inst):
        picks, pa = self._sel(inst.info)
        allp = sorted(list(Set[int](picks + pa)))
        use = inst.info._pick(allp)
        has_eeg = False
        for t in use.ch_types:
            if t == "eeg":
                has_eeg = True
        if has_eeg and not _has_eeg_average_ref_proj(use):
            raise RuntimeError("No average reference for the EEG channels has been set. Use inst.set_eeg_reference(projection=True) to do so.")
        if self.proj and not _is_proj(inst):
            inst.apply_proj()
        if not _is_proj(inst) and len(use.projs) > 0:
            raise RuntimeError("Projections need to be applied before regression can be performed. Use the .apply_proj() method to do so.")
        return picks, pa

    def fit(self, inst):
        picks, pa = self._check_inst(inst)
        d3 = _arr3(inst)
        n_ep = d3.shape[0]
        nt = d3.shape[2]
        n_art = len(pa)
        ref = np.zeros((n_art, n_ep * nt))
        for e in range(n_ep):
            for a in range(n_art):
                row = d3[e, pa[a], :]
                ref[a, e * nt:(e + 1) * nt] = row - np.mean(row)
        cov_ref = np.dot(ref, ref.T)
        coef = np.zeros((len(picks), n_art))
        for pi in range(len(picks)):
            cd = np.zeros((1, n_ep * nt))
            for e in range(n_ep):
                row = d3[e, picks[pi], :]
                cd[0, e * nt:(e + 1) * nt] = row - np.mean(row)
            rhs = np.dot(ref, cd.T)
            coef[pi] = np.linalg.solve(cov_ref, rhs).T[0]
        self.coef_ = coef
        self.info_ = inst.info.copy()
        self.fitted = True
        return self

    def apply(self, inst, copy: bool = True):
        out = inst.copy() if copy else inst
        picks, pa = self._check_inst(out)
        ref_picks, ref_pa = self._sel(self.info_)
        for k in range(min(len(picks), len(ref_picks))):
            if out.info.ch_names[picks[k]] != self.info_.ch_names[ref_picks[k]]:
                raise ValueError("Selected data channels are not compatible with the regression weights. Make sure that all data channels are present and in the correct order.")
        for k in range(min(len(pa), len(ref_pa))):
            if out.info.ch_names[pa[k]] != self.info_.ch_names[ref_pa[k]]:
                raise ValueError("Selected artifact channels are not compatible with the regression weights. Make sure that all artifact channels are present and in the correct order.")
        d3 = _arr3(out)
        n_ep = d3.shape[0]
        nt = d3.shape[2]
        ref = np.zeros((n_ep, len(pa), nt))
        for e in range(n_ep):
            for a in range(len(pa)):
                row = d3[e, pa[a], :]
                ref[e, a, :] = row - np.mean(row)
        for pi in range(len(picks)):
            for e in range(n_ep):
                corr = np.dot(self.coef_[pi], ref[e])
                d3[e, picks[pi], :] -= corr
        return out

    def __repr__(self) -> str:
        if self.fitted:
            n = self.coef_.shape[1]
            return f"<EOGRegression | fitted to {n} artifact channel{'s' if n > 1 else ''}>"
        return "<EOGRegression | not fitted>"
