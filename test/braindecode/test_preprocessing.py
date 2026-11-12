# Expected values: test/braindecode/gen_data_expected.py (braindecode 2.0.0, mne 1.13.2)
import numpy as np
import mne
from braindecode.datasets import RawDataset, BaseConcatDataset, Description
from braindecode.preprocessing import Preprocessor, preprocess, exponential_moving_standardize, exponential_moving_demean, filterbank

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


@test
def test_exponential_moving():
    X, y = make_X()
    s = exponential_moving_standardize(X[0], factor_new=1e-3, init_block_size=100)
    assert close(s.sum(), -31.365074429029605, 1e-11)
    assert close(s[0, 95:105], [-0.6375184347147742, -0.8109875318677084, -0.46268681619492713, -0.2247772245624635, -0.41505645474859315, -0.6062834141414714, -0.25673631162247046, 0.445538611298846, 0.8448024713461774, 0.7513523003555033])
    assert close(s[7, -5:], [1.696141805660064, -0.2965364533339456, 2.063826075212794, -0.5218443740487122, 1.1594876773071763])
    s2 = exponential_moving_standardize(X[1], factor_new=0.05)
    assert close(s2[2, :5], [0.0, 1.396424004376894, -1.0227619643445214, -1.8094857123221575, -1.546464986062447])
    assert close(s2.sum(), -35.758207161604616, 1e-11)
    d = exponential_moving_demean(X[2], factor_new=0.01, init_block_size=30)
    assert close(d[1, 25:35], [0.44738345357615084, 0.7709991295645141, 1.0362511402205743, 1.1804071356489843, 0.8087216950824682, -0.18332226452456607, -1.101019341152818, -1.2072861877358398, -0.4824639138588797, 0.32772908959200997])
    assert close(d.sum(), -18.18225837961286, 1e-11)
    raised = False
    try:
        exponential_moving_standardize(X[0], factor_new=0.0)
    except ValueError:
        raised = True
    assert raised

def scale(a, factor: float = 1.0):
    return a * factor

@test
def test_preprocess_pipeline():
    X, y = make_X()
    dss = List[RawDataset]()
    for k in range(2):
        raw = mne.io.RawArray(X[k] * 1e-5, mne.create_info([f"E{i}" for i in range(8)], 128., "eeg"))
        d = Description()
        d["subject"] = k
        dss.append(RawDataset(raw, d))
    c = BaseConcatDataset(dss)
    preprocess(c, [
        Preprocessor("pick", picks=["E0", "E1", "E3", "E5"]),
        Preprocessor(scale, factor=1e5),
        Preprocessor("filter", l_freq=4.0, h_freq=30.0),
        Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=50),
    ])
    d0 = c.datasets[0].raw.get_data()
    assert d0.shape == (4, 256)
    assert close(d0.sum(), -20.116885382417294, 1e-9)
    assert close(d0[2, 100:105], [-0.6792377921608542, -1.527107974661831, 0.3625008045499165, 1.495509156364647, -0.3183775819492299], 1e-9)
    d1 = c.datasets[1].raw.get_data()
    assert close(d1[3, -4:], [-3.5578024676549402, -4.002742766299167, -2.780288190914959, 0.06593307500612898], 1e-9)
    assert len(c.datasets[0].raw_preproc_kwargs) == 4

@test
def test_filterbank_resample_crop_reference():
    X, y = make_X()
    r = mne.io.RawArray(X[3, :2] * 1.0, mne.create_info(["C3", "Cz"], 128., "eeg"))
    filterbank(r, ((4, 8), (8.5, 30.0)), drop_original_signals=False)
    assert r.ch_names == ["C3", "C3_4-8", "C3_8.5-30.0", "Cz", "Cz_4-8", "Cz_8.5-30.0"]
    g = r.get_data()
    assert close(g[1, 60:63], [-0.6179100155916685, -0.7472131541009666, -0.7720836591283003], 1e-9)
    assert close(g[5, 60:63], [0.3828975764981366, 0.9135492418939646, 1.046062466383028], 1e-9)
    r2 = mne.io.RawArray(X[3, :2] * 1.0, mne.create_info(["C3", "Cz"], 128., "eeg"))
    preprocess(BaseConcatDataset([RawDataset(r2)]), [Preprocessor("resample", sfreq=64.0),
               Preprocessor("crop", tmin=0.25, tmax=1.5), Preprocessor("set_eeg_reference", ref_channels="average")])
    g2 = r2.get_data()
    assert g2.shape == (2, 81)
    assert close(g2[0, :4], [0.13948129747342475, 0.18568284754335873, 0.25863113758475675, -0.13671173653714752], 1e-9)

test_exponential_moving()
test_preprocess_pipeline()
test_filterbank_resample_crop_reference()
