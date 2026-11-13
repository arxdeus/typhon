# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# eXimia .nxe reader (mne/io/eximia/eximia.py): multiplexed int16, 64 channels.

import numpy as np
from ._bytes import _read_file, _i16_le
from .._info import Info
from .._containers import RawArray

_EX_NAMES = ("Fp1 Fpz Fp2 AF1 AFz AF2 F7 F3 F1 Fz F2 F4 F8 FT9 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 FT10 "
             "T7 C5 C3 C1 Cz C2 C4 C6 T8 TP9 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 TP10 "
             "P9 P7 P3 P1 Pz P2 P4 P8 P10 PO3 POz PO4 O1 Oz O2 Iz")


def read_raw_eximia(fname, preload = False, verbose = None) -> RawArray:
    fn = str(fname)
    b = _read_file(fn)
    n_chan = 64
    names = ["GateIn", "Trig1", "Trig2", "EOG"]
    types = ["stim", "stim", "stim", "eog"]
    cals = [0.0015259021896696422, 0.0015259021896696422, 0.0015259021896696422, 0.3814755474174106]
    for nm in _EX_NAMES.split():
        names.append(nm)
        types.append("eeg")
        cals.append(0.07629510948348212)
    if len(names) != n_chan:
        raise RuntimeError("eXimia channel table mismatch")
    n_samples = len(b) // (n_chan * 2)
    if len(b) % (n_chan * 2) != 0:
        print("RuntimeWarning: Incorrect number of samples in file, the file is likely truncated")
    info = Info(names, 1450.0, types)
    for k in range(n_chan):
        info.cals[k] = cals[k]
        if types[k] == "stim":
            info.units[k] = -1
            info.coil_types[k] = 0
    data = np.zeros((n_chan, n_samples))
    for s in range(n_samples):
        base = s * n_chan * 2
        for c in range(n_chan):
            data[c, s] = float(_i16_le(b, base + c * 2)) * cals[c]
    raw = RawArray(data, info)
    raw._raw_kind = "RawEximia"
    return raw
