# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Epochs/Evoked remainder: apply_function, detrend, grand_average.

import numpy as np
from ._info import Info, _picks_to_idx
from ._containers import EpochsArray, Evoked
from .epochs import combine_evoked
from .channels import equalize_channels
from scipy.signal import detrend as _sp_detrend


@extend
class EpochsArray:
    def apply_function(self, fun, picks = None, dtype = None, n_jobs = None,
                       channel_wise: bool = True, verbose = None, **kwargs):
        """Apply fun to the data in place (channel_wise: fun gets each (epoch, channel) row,
        else the 3-D (n_epochs, n_picks, n_times) array)."""
        pk = _picks_to_idx(self.info, picks, "data_or_ica", ())
        if channel_wise:
            for ci in pk:
                for e in range(self._data.shape[0]):
                    row = self._data[e, ci, :].copy()
                    self._data[e, ci, :] = np.asarray(fun(row, **kwargs))
        else:
            sel = np.array(pk, dtype=int)
            out = np.asarray(fun(self._data[:, sel, :].copy(), **kwargs))
            for k in range(len(pk)):
                self._data[:, pk[k], :] = out[:, k, :]
        return self


@extend
class Evoked:
    def apply_function(self, fun, picks = None, dtype = None, n_jobs = None,
                       channel_wise: bool = True, verbose = None, **kwargs):
        pk = _picks_to_idx(self.info, picks, "data_or_ica", ())
        if channel_wise:
            for ci in pk:
                row = self.data[ci, :].copy()
                self.data[ci, :] = np.asarray(fun(row, **kwargs))
        else:
            sel = np.array(pk, dtype=int)
            out = np.asarray(fun(self.data[sel, :].copy(), **kwargs))
            for k in range(len(pk)):
                self.data[pk[k], :] = out[k]
        return self

    def drop_channels(self, ch_names, on_missing: str = "raise"):
        names = [ch_names] if isinstance(ch_names, str) else [str(c) for c in ch_names]
        for c in names:
            if c not in self.info.ch_names and on_missing == "raise":
                raise ValueError(f"Channel(s) {c} not found, nothing dropped.")
        keep = [i for i in range(self.info.nchan) if self.info.ch_names[i] not in names]
        return self._pick_keep(keep)

    def detrend(self, order: int = 1, picks = None):
        """Detrend (order 0 constant, 1 linear) the selected channels in place."""
        pk = _picks_to_idx(self.info, picks, "data_or_ica", ())
        if order != 0 and order != 1:
            raise ValueError("order must be 0 or 1")
        sel = np.array(pk, dtype=int)
        d = _sp_detrend(self.data[sel, :], axis=-1, type="constant" if order == 0 else "linear")
        for k in range(len(pk)):
            self.data[pk[k], :] = d[k]
        return self


def grand_average(all_inst, interpolate_bads: bool = True, drop_bads: bool = True):
    """Grand average of Evoked objects (equal weights, nave = number of inputs)."""
    insts = [e.copy() for e in all_inst]
    if len(insts) == 0:
        raise ValueError("Please pass a list of Evoked, AverageTFR, or Spectrum objects, got an empty list.")
    if interpolate_bads:
        for e in insts:
            if len(e.info.bads) > 0:
                e.interpolate_bads()
    if drop_bads:
        bads = Set[str]()
        for e in insts:
            for b in e.info.bads:
                bads.add(b)
        if len(bads) > 0:
            for e in insts:
                e.drop_channels(list(bads))
    eq = equalize_channels(insts, False)
    ga = combine_evoked(eq, "equal")
    ga.nave = len(eq)
    ga.comment = f"Grand average (n = {ga.nave})"
    return ga
