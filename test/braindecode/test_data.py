# Expected values: test/braindecode/gen_data_expected.py (braindecode 2.0.0, mne 1.13.2)
import numpy as np
import mne
from braindecode.datasets import create_from_X_y, RawDataset, BaseConcatDataset, Description
from braindecode.preprocessing import create_windows_from_events, create_fixed_length_windows

def make_X(n_ep: int = 40, n_ch: int = 8, n_t: int = 256):
    X = np.zeros((n_ep, n_ch, n_t))
    y = np.arange(n_ep) % 2
    for e in range(n_ep):
        for c in range(n_ch):
            for t in range(n_t):
                v = (np.sin(0.37 * t * (c + 1) + 0.11 * e) + 0.5 * np.cos(0.071 * t * (e % 7 + 1) + c)
                     + 0.3 * np.sin(1.3 * t + 0.7 * e * c))
                if c % 3 == 0:
                    v += (y[e] * 0.8 + 0.2) * np.sin(2 * np.pi * 12.0 / 128.0 * t)
                v += 0.05 * c
                X[e, c, t] = v
    return X, y

def close(a, b, rtol: float = 1e-12) -> bool:
    x = np.asarray(a).astype(float).ravel()
    z = np.asarray(b).astype(float).ravel()
    if x.shape[0] != z.shape[0]:
        print("length mismatch", x.shape[0], z.shape[0])
        return False
    for i in range(x.shape[0]):
        if abs(x[i] - z[i]) > rtol * max(abs(z[i]), 1e-12):
            print("mismatch at", i, x[i], z[i])
            return False
    return True

def ints(s) -> List[int]:
    return [int(v) for v in s.to_list()]

@test
def test_create_from_X_y():
    X, y = make_X()
    ds = create_from_X_y(X, y, drop_last_window=False, sfreq=128.0,
                         ch_names=[f"E{i}" for i in range(8)],
                         window_size_samples=128, window_stride_samples=64)
    assert len(ds) == 120
    assert ds.cumulative_sizes[:4] == [3, 6, 9, 12]
    x0, y0, c0 = ds[0]
    assert x0.shape == (8, 128)
    assert isinstance(x0, np.ndarray[np.float32, 2])
    assert y0 == 0 and c0 == [0, 0, 128]
    # float32 window: exact bitwise equality with the float32 cast
    assert close(x0[3, :6], [-0.34499624371528625, 1.0473076105117798, 0.1700226217508316, -1.3220294713974, -0.8238071799278259, 0.6636046767234802], 0.0)
    x5, y5, c5 = ds[5]
    assert y5 == 1 and c5 == [2, 128, 256]
    assert close(x5[7, -4:], [-0.6093491315841675, 1.7705016136169434, 0.020050426945090294, 1.434334635734558], 0.0)
    md = ds.get_metadata()
    assert ints(md["i_start_in_trial"])[:6] == [0, 64, 128, 0, 64, 128]
    assert ints(md["target"])[:6] == [0, 0, 0, 1, 1, 1]
    assert ints(md["i_dataset"])[-3:] == [39, 39, 39]
    assert [str(c) for c in md.columns] == ["i_window_in_trial", "i_start_in_trial", "i_stop_in_trial", "target", "i_dataset"]
    sp = ds.split("target")
    assert list(sp.keys()) == ["0", "1"]
    assert len(sp["0"]) == 60 and len(sp["1"]) == 60
    sp2 = ds.split([[0, 2], [1]])
    assert len(sp2["0"].datasets) == 2 and len(sp2["1"].datasets) == 1
    desc = ds.description
    assert desc.shape == (40, 1) and int(desc.loc[3, "target"]) == 1
    ds2 = create_from_X_y(X, y, drop_last_window=True, sfreq=128.0)
    assert len(ds2) == 40
    assert ds2[3][1] == 1 and ds2[3][2] == [0, 0, 256]
    ds3 = create_from_X_y(X[:3, :, :200], y[:3], drop_last_window=True, sfreq=128.0,
                          window_size_samples=64, window_stride_samples=48)
    assert len(ds3) == 9
    assert ints(ds3.get_metadata()["i_start_in_trial"])[:4] == [0, 48, 96, 0]
    # transforms
    ds.transform = lambda a: a * np.float32(2)
    assert close(ds[0][0][3, :1], [2 * -0.34499624371528625], 0.0)

def _events_concat():
    xr = np.arange(3000.).reshape(3, 1000) / 1000
    dss = List[RawDataset]()
    for k in range(2):
        raw = mne.io.RawArray(xr + k, mne.create_info(["a", "b", "c"], 100., "eeg"))
        raw.set_annotations(mne.Annotations([1.0 + k * 0.1, 3.0, 5.5], [2.0, 2.0, 2.0], ["left", "right", "left"]))
        d = Description()
        d["subject"] = k
        d["session"] = "s"
        dss.append(RawDataset(raw, d))
    return BaseConcatDataset(dss)

@test
def test_windows_from_events():
    c = _events_concat()
    w = create_windows_from_events(c, trial_start_offset_samples=-20, trial_stop_offset_samples=10,
                                   window_size_samples=100, window_stride_samples=60)
    md = w.get_metadata()
    assert len(w) == 24
    assert ints(md["i_start_in_trial"]) == [80, 140, 200, 210, 280, 340, 400, 410, 530, 590, 650, 660, 90, 150, 210, 220, 280, 340, 400, 410, 530, 590, 650, 660]
    assert ints(md["target"]) == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert ints(md["i_trial_in_dataset"]) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert [str(c) for c in md.columns] == ["i_window_in_trial", "i_trial_in_dataset", "i_start_in_trial", "i_stop_in_trial", "target", "subject", "session", "i_dataset"]
    assert str(md.iloc[13, 6]) == "s"
    x, t, ci = w[15]
    assert t == 0 and ci == [3, 220, 320]
    assert close(x[:, 0], [1.2200000286102295, 2.2200000286102295, 3.2200000286102295], 0.0)
    m = Dict[str, int]()
    m["left"] = 1
    m["right"] = 0
    w2 = create_windows_from_events(c, mapping=m)
    md2 = w2.get_metadata()
    assert len(w2) == 6
    assert ints(md2["i_start_in_trial"]) == [100, 300, 550, 110, 300, 550]
    assert ints(md2["i_stop_in_trial"]) == [300, 500, 750, 310, 500, 750]
    assert ints(md2["target"]) == [1, 0, 1, 1, 0, 1]
    w3 = create_windows_from_events(c, trial_start_offset_samples=0, trial_stop_offset_samples=0,
                                    window_size_samples=150, window_stride_samples=100, on_last_window="drop")
    assert len(w3) == 6
    assert ints(w3.get_metadata()["i_start_in_trial"]) == [100, 300, 550, 110, 300, 550]
    f = create_fixed_length_windows(c, window_size_samples=300, window_stride_samples=250, drop_last_window=False)
    assert len(f) == 8
    assert ints(f.get_metadata()["i_start_in_trial"]) == [0, 250, 500, 700, 0, 250, 500, 700]
    assert ints(f.get_metadata()["target"]) == [-1, -1, -1, -1, -1, -1, -1, -1]
    # errors
    raised = False
    try:
        create_windows_from_events(c, trial_start_offset_samples=-200, trial_stop_offset_samples=0,
                                   window_size_samples=100, window_stride_samples=100)
    except ValueError:
        raised = True
    assert raised

@test
def test_mne_raw_epochs_constructors():
    from braindecode.datasets import create_from_mne_raw, create_from_mne_epochs
    c = _events_concat()
    raws = [d.raw for d in c.datasets]
    w = create_from_mne_raw(raws, 0, 0, 200, 200, drop_last_window=False)
    assert len(w) == 6
    X = np.arange(4 * 3 * 100.).reshape(4, 3, 100)
    y = np.arange(4) % 2
    ev = np.c_[np.arange(4) * 100, np.zeros(4, int), y.astype(int)]
    ep = mne.EpochsArray(X, mne.create_info(3, 100.0, "eeg"), ev)
    we = create_from_mne_epochs([ep], 50, 30, drop_last_window=False)
    assert len(we.datasets) == 4 and len(we) == 12
    xx, tt, cc = we[4]
    assert tt == 1 and cc == [1, 130, 180]
    assert close(xx[1, :3], [430.0, 431.0, 432.0], 0.0)

test_create_from_X_y()
test_windows_from_events()
test_mne_raw_epochs_constructors()
