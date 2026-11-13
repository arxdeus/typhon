import numpy as np
import mne

def tryit(name: str, ok: bool):
    print(name, ok)

D = "/nonexistent_dir/x"
def t_nihon():
    try:
        mne.io.read_raw_nihon(D + ".EEG")
    except Exception:
        pass
def t_kit():
    try:
        mne.io.read_raw_kit(D + ".sqd")
    except Exception:
        pass
def t_curry():
    try:
        mne.io.read_raw_curry(D + ".dat")
    except Exception:
        pass
def t_nirx():
    try:
        mne.io.read_raw_nirx(D)
    except Exception:
        pass
def t_ft():
    try:
        mne.io.read_raw_fieldtrip(D + ".mat", None)
    except Exception:
        pass
def t_rr():
    try:
        mne.io.read_raw(D + ".vhdr")
    except Exception:
        pass
def t_fc():
    info = mne.create_info(["MEG0111", "MEG0112"], 1000.0, "mag")
    raw = mne.io.RawArray(np.zeros((2, 100)), info)
    try:
        mne.preprocessing.compute_fine_calibration(raw)
    except Exception:
        pass
t_fc()
t_nihon()
t_kit()
t_curry()
t_nirx()
t_ft()
t_rr()
print("done")
