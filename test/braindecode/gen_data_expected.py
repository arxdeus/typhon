"""Generate expected values for test/braindecode/test_data*.py and
test_preprocessing*.py from real braindecode 2.0.0 / mne 1.13.2.

Run with /Users/mind/orca/toolchains/bd-venv/bin/python and paste the output
literals into the tests.
"""
import warnings

import mne
import numpy as np

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

from braindecode.datasets import BaseConcatDataset, RawDataset, create_from_X_y  # noqa: E402
from braindecode.preprocessing import (  # noqa: E402
    Preprocessor,
    create_fixed_length_windows,
    create_windows_from_events,
    exponential_moving_demean,
    exponential_moving_standardize,
    filterbank,
    preprocess,
)


def make_X(n_ep=40, n_ch=8, n_t=256):
    # mirrors make_X in the typhon tests (deterministic, no RNG)
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


def L(a):
    return "[" + ", ".join(repr(float(v)) for v in np.asarray(a).ravel()) + "]"


def Li(a):
    return "[" + ", ".join(str(int(v)) for v in np.asarray(a).ravel()) + "]"


X, y = make_X()
print("# acceptance create_from_X_y")
ds = create_from_X_y(X, y, drop_last_window=False, sfreq=128.0,
                     ch_names=[f"E{i}" for i in range(8)],
                     window_size_samples=128, window_stride_samples=64)
print("len", len(ds), "cum", ds.cumulative_sizes[:4])
x0, y0, c0 = ds[0]
print("x0 dtype", x0.dtype, x0.shape, "y0", y0, "c0", c0)
print("x0[3,:6]", L(x0[3, :6]))
x5, y5, c5 = ds[5]
print("ds[5]", y5, c5, L(x5[7, -4:]))
md = ds.get_metadata()
print("md starts", Li(md["i_start_in_trial"].to_numpy()[:6]), "targets", Li(md["target"].to_numpy()[:6]))
print("md i_dataset tail", Li(md["i_dataset"].to_numpy()[-3:]))
sp = ds.split("target")
print("split", {k: len(v) for k, v in sp.items()})
ds2 = create_from_X_y(X, y, drop_last_window=True, sfreq=128.0)
print("ds2", len(ds2), ds2[3][1], ds2[3][2])
ds3 = create_from_X_y(X[:3, :, :200], y[:3], drop_last_window=True, sfreq=128.0,
                      window_size_samples=64, window_stride_samples=48)
print("ds3", len(ds3), Li(ds3.get_metadata()["i_start_in_trial"].to_numpy()[:4]))

print("# exponential moving")
s = exponential_moving_standardize(X[0], factor_new=1e-3, init_block_size=100)
print("ems sum", repr(float(s.sum())), "ems[0,95:105]", L(s[0, 95:105]))
print("ems[7,-5:]", L(s[7, -5:]))
s2 = exponential_moving_standardize(X[1], factor_new=0.05)
print("ems2[2,:5]", L(s2[2, :5]), "sum", repr(float(s2.sum())))
d = exponential_moving_demean(X[2], factor_new=0.01, init_block_size=30)
print("emd[1,25:35]", L(d[1, 25:35]), "sum", repr(float(d.sum())))

print("# windows from events")
xr = np.arange(3000.).reshape(3, 1000) / 1000
raws = []
for k in range(2):
    raw = mne.io.RawArray(xr + k, mne.create_info(["a", "b", "c"], 100., "eeg"))
    raw.set_annotations(mne.Annotations([1.0 + k * 0.1, 3.0, 5.5], [2.0, 2.0, 2.0], ["left", "right", "left"]))
    raws.append(raw)
c = BaseConcatDataset([RawDataset(r, {"subject": i, "session": "s"}) for i, r in enumerate(raws)])
w = create_windows_from_events(c, trial_start_offset_samples=-20, trial_stop_offset_samples=10,
                               window_size_samples=100, window_stride_samples=60)
md = w.get_metadata()
print("w", len(w), "starts", Li(md["i_start_in_trial"].to_numpy()), "targets", Li(md["target"].to_numpy()),
      "trial", Li(md["i_trial_in_dataset"].to_numpy()))
print("w[15]", w[15][1], w[15][2], L(w[15][0][:, 0]))
w2 = create_windows_from_events(c, mapping={"left": 1, "right": 0})
md2 = w2.get_metadata()
print("w2", len(w2), Li(md2["i_start_in_trial"].to_numpy()), Li(md2["i_stop_in_trial"].to_numpy()),
      Li(md2["target"].to_numpy()))
w3 = create_windows_from_events(c, trial_start_offset_samples=0, trial_stop_offset_samples=0,
                                window_size_samples=150, window_stride_samples=100, on_last_window="drop")
print("w3", len(w3), Li(w3.get_metadata()["i_start_in_trial"].to_numpy()))
f = create_fixed_length_windows(c, window_size_samples=300, window_stride_samples=250, drop_last_window=False)
print("fixed", len(f), Li(f.get_metadata()["i_start_in_trial"].to_numpy()), Li(f.get_metadata()["target"].to_numpy()))

print("# preprocess")
raws = []
for k in range(2):
    raw = mne.io.RawArray(X[k] * 1e-5, mne.create_info([f"E{i}" for i in range(8)], 128., "eeg"))
    raws.append(raw)
c = BaseConcatDataset([RawDataset(r, {"subject": i}) for i, r in enumerate(raws)])


def scale(a, factor=1.0):
    return a * factor


preprocess(c, [
    Preprocessor("pick", picks=["E0", "E1", "E3", "E5"]),
    Preprocessor(scale, factor=1e5),
    Preprocessor("filter", l_freq=4.0, h_freq=30.0),
    Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=50),
])
d0 = c.datasets[0].raw.get_data()
print("pp shape", d0.shape, "sum", repr(float(d0.sum())), "d0[2,100:105]", L(d0[2, 100:105]))
d1 = c.datasets[1].raw.get_data()
print("pp1[3,-4:]", L(d1[3, -4:]))
r = mne.io.RawArray(X[3, :2] * 1.0, mne.create_info(["C3", "Cz"], 128., "eeg"))
filterbank(r, [(4, 8), (8.5, 30.0)], drop_original_signals=False)
print("fb", r.ch_names, L(r.get_data()[1, 60:63]), L(r.get_data()[5, 60:63]))
r = mne.io.RawArray(X[3, :2] * 1.0, mne.create_info(["C3", "Cz"], 128., "eeg"))
preprocess(BaseConcatDataset([RawDataset(r)]), [Preprocessor("resample", sfreq=64.0), Preprocessor("crop", tmin=0.25, tmax=1.5),
                                                  Preprocessor("set_eeg_reference", ref_channels="average")])
print("rs", r.get_data().shape, L(r.get_data()[0, :4]))

print("# augmentation")
import torch  # noqa: E402
from braindecode.augmentation import (  # noqa: E402
    BandstopFilter, ChannelsDropout, ChannelsShuffle, Compose, FrequencyShift, FTSurrogate,
    GaussianNoise, Mixup, SignFlip, SmoothTimeMask, TimeReverse)
from numpy.random import RandomState  # noqa: E402
X = (np.sin(np.arange(4*3*64).reshape(4,3,64)*0.37)+0.1*np.cos(np.arange(4*3*64).reshape(4,3,64)*1.3)).astype(np.float32)
y = np.array([0,1,0,1])

for name, t in [("trev", TimeReverse(0.5, random_state=1)), ("sflip", SignFlip(1.0, random_state=1)),
    ("fts", FTSurrogate(0.8, phase_noise_magnitude=0.7, channel_indep=True, random_state=3)),
    ("fts2", FTSurrogate(1.0, phase_noise_magnitude=1.0, channel_indep=False, random_state=4)),
    ("cdrop", ChannelsDropout(1.0, p_drop=0.5, random_state=5)), ("cshuf", ChannelsShuffle(1.0, p_shuffle=0.6, random_state=6)),
    ("gauss", GaussianNoise(0.7, std=0.3, random_state=7)), ("stm", SmoothTimeMask(1.0, mask_len_samples=10, random_state=8)),
    ("bstop", BandstopFilter(1.0, sfreq=64.0, bandwidth=2.0, random_state=9)), ("fshift", FrequencyShift(1.0, sfreq=64.0, max_delta_freq=3.0, random_state=10)),
    ("mix", Mixup(0.4, random_state=11))]:
    o = t(torch.as_tensor(X), torch.as_tensor(y))
    ox = o[0]
    print(name, repr(float(ox.double().sum())), L(ox[1, 2, 5:9]), L(ox[3, 0, -3:]))
    if name == "mix": print("mixy", o[1][1].tolist(), L(o[1][2]))
c = Compose([TimeReverse(1.0, random_state=1), GaussianNoise(1.0, std=0.1, random_state=2)])
o = c(torch.as_tensor(X), torch.as_tensor(y)); print("comp", repr(float(o[0].double().sum())), L(o[0][2,1,:3]))
r = RandomState(42)
print("rs", r.uniform(size=3).tolist(), r.normal(size=2).tolist(), r.randint(10, size=3).tolist(),
      r.permutation(5).tolist(), r.beta(0.5, 0.5, 2).tolist(), r.random((2,)).tolist(), r.beta(2.0, 3.0))

print("# training")
from braindecode.training import trial_preds_from_window_preds  # noqa: E402
p = np.arange(5 * 2 * 6, dtype=float).reshape(5, 2, 6)
for a in trial_preds_from_window_preds(p, [0, 1, 2, 0, 1], [6, 9, 12, 6, 10]):
    print(a.shape, a.astype(int).tolist())

print("# EEGClassifier acceptance")
from braindecode import EEGClassifier  # noqa: E402
from braindecode.models import ShallowFBCSPNet  # noqa: E402
from braindecode.util import set_random_seeds  # noqa: E402
set_random_seeds(0, cuda=False)
X, y = make_X()
ds = create_from_X_y(X.astype(np.float32), y, drop_last_window=False, sfreq=128.0,
                     ch_names=[f"E{i}" for i in range(8)],
                     window_size_samples=128, window_stride_samples=64)
torch.manual_seed(0)
model = ShallowFBCSPNet(n_chans=8, n_outputs=2, n_times=128, final_conv_length="auto")
clf = EEGClassifier(model, criterion=torch.nn.CrossEntropyLoss, optimizer=torch.optim.AdamW,
                    optimizer__lr=0.001, train_split=None, batch_size=8, max_epochs=2, verbose=0)
clf.fit(ds, y=None)
print("train_loss", [repr(v) for v in clf.history[:, "train_loss"]])
print("pred", clf.predict(ds)[:10].tolist())
