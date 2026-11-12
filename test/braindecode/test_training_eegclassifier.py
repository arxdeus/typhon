# Expected values: test/braindecode/gen_data_expected.py ("EEGClassifier acceptance")
# Real braindecode 2.0.0 + skorch 1.4.0 + torch 2.15.0 CPU.
import numpy as np
import torch
from torch import nn
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.datasets import create_from_X_y
from braindecode.util import set_random_seeds

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

@test
def test_eegclassifier_fit():
    set_random_seeds(0, cuda=False)
    X, y = make_X()
    ds = create_from_X_y(X.astype(np.float32), y, drop_last_window=False, sfreq=128.0,
                         ch_names=[f"E{i}" for i in range(8)],
                         window_size_samples=128, window_stride_samples=64)
    torch.manual_seed(0)
    model = ShallowFBCSPNet(n_chans=8, n_outputs=2, n_times=128, final_conv_length="auto")
    clf = EEGClassifier(model, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.AdamW,
                        optimizer__lr=0.001, train_split=None, batch_size=8, max_epochs=2)
    clf.fit(ds, y=None)
    tl = clf.history[:, "train_loss"]
    # float32 training: observed rel diff 4.5e-8 / 1.9e-7 (float32 reduction order)
    assert abs(tl[0] - 0.3507235186795394) <= 1e-6 * 0.3507235186795394
    assert abs(tl[1] - 0.07846576652179162) <= 1e-6 * 0.07846576652179162
    assert clf.predict(ds)[:10].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 1]

test_eegclassifier_fit()
