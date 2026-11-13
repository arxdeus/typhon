import numpy as np
import mne

D = "test/mne/data/"
a = mne.io.read_raw(D + "test_edf.edf", preload=True)
b = mne.io.read_raw_edf(D + "test_edf.edf", preload=True)
assert a.ch_names == b.ch_names
assert np.allclose(a.get_data(), b.get_data())
a = mne.io.read_raw(D + "test_bdf.bdf", preload=True)
assert a.n_times == 1024 and a.info["sfreq"] == 256.0
a = mne.io.read_raw(D + "test_gdf2.gdf", preload=True)
assert a.ch_names == ["Fp1", "Fp2", "Cz", "Status"]
a = mne.io.read_raw(D + "test_raw.fif", preload=True)
assert a.n_times > 0
a = mne.io.read_raw(D + "test_eeglab.set", preload=True)
assert a.n_times > 0
a = mne.io.read_raw(D + "test_bv_i16.vhdr", preload=True)
assert a.info["sfreq"] == 250.0
try:
    mne.io.read_raw("x.unknown")
    print("TEST FAILED: no error")
except ValueError:
    pass
print("done")
