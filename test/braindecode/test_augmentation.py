# Expected values: test/braindecode/gen_data_expected.py (augmentation section, braindecode 2.0.0)
import numpy as np
from braindecode.augmentation import *
from braindecode._random_state import RandomState

def close(a, b, rtol: float = 0.0) -> bool:
    x = np.asarray(a).astype(float).ravel()
    z = np.asarray(b).astype(float).ravel()
    if x.shape[0] != z.shape[0]:
        print("length mismatch", x.shape[0], z.shape[0])
        return False
    for i in range(x.shape[0]):
        if abs(x[i] - z[i]) > rtol * max(abs(z[i]), 1e-6):
            print("mismatch at", i, x[i], z[i])
            return False
    return True

def data():
    A = np.arange(4 * 3 * 64).reshape(4, 3, 64).astype(float)
    X = (np.sin(A * 0.37) + 0.1 * np.cos(A * 1.3)).astype(np.float32)
    return X, np.array([0, 1, 0, 1])

def check(t, s: float, a, b, rtol: float = 0.0):
    X, y = data()
    ox, oy = t(X, y)
    ok = close(ox.astype(float).sum(), s, max(rtol, 1e-13)) and close(ox[1, 2, 5:9], a, rtol) and close(ox[3, 0, -3:], b, rtol)
    return ok

@test
def test_random_state_legacy():
    r = RandomState(42)
    assert close(r.uniform(size=3), [0.3745401188473625, 0.9507143064099162, 0.7319939418114051])
    assert close(r.normal(size=2), [-1.1118801180469204, 0.31890218468938336])
    assert r.randint(10, size=3).tolist() == [2, 6, 7]
    assert r.permutation(5).tolist() == [0, 1, 2, 3, 4]
    assert close(r.beta(0.5, 0.5, 2), [0.0060676485984432664, 0.9999993115123513])
    assert close(r.random((2,)), [0.9922115592912175, 0.6174815096277165])
    assert close(r.beta(2.0, 3.0), 0.45543839870822056)

@test
def test_transforms():
    assert check(TimeReverse(0.5, random_state=1), 1.7384711587801576, [0.768412709236145, 0.8505353331565857, 0.9439648985862732, 0.9831460118293762], [0.1448722779750824, -0.2116633951663971, -0.44141820073127747])
    assert check(SignFlip(1.0, random_state=1), -1.7384711587801576, [-0.768412709236145, -0.8505353331565857, -0.9439648985862732, -0.9831460118293762], [0.04186507314443588, 0.3264569938182831, 0.6998047828674316])
    assert check(FTSurrogate(0.8, phase_noise_magnitude=0.7, channel_indep=True, random_state=3), 1.7384716146916617, [0.12956514954566956, -0.16070516407489777, -0.40076538920402527, -0.6332867741584778], [0.9316272735595703, 0.9824253916740417, 0.880519688129425])
    assert check(FTSurrogate(1.0, phase_noise_magnitude=1.0, channel_indep=False, random_state=4), 1.7384712893981487, [0.09075471013784409, 0.008324194699525833, -0.06553149223327637, -0.3195401728153229], [-1.0482680797576904, -0.6117464303970337, -0.2092052400112152])
    assert check(ChannelsDropout(1.0, p_drop=0.5, random_state=5), -5.684649964258426, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert check(ChannelsShuffle(1.0, p_shuffle=0.6, random_state=6), 1.7384711587801576, [0.768412709236145, 0.8505353331565857, 0.9439648985862732, 0.9831460118293762], [-0.04186507314443588, -0.3264569938182831, -0.6998047828674316])
    assert check(GaussianNoise(0.7, std=0.3, random_state=7), -5.332206592429429, [0.768412709236145, 0.8505353331565857, 0.9439648985862732, 0.9831460118293762], [-0.04186507314443588, -0.3264569938182831, -0.6998047828674316])
    assert check(SmoothTimeMask(1.0, mask_len_samples=10, random_state=8), -4.761267437014295, [1.359158290172165e-30, 4.067784976709079e-32, 2.7572115982674747e-25, 1.7538141841290716e-18], [-0.04186507314443588, -0.3264569938182831, -0.6998047828674316], 1e-12)
    assert check(BandstopFilter(1.0, sfreq=64.0, bandwidth=2.0, random_state=9), 2.103814373840578, [0.7284473180770874, 0.7928965091705322, 0.901869535446167, 0.9781960844993591], [-0.004872145131230354, -0.30521178245544434, -0.7098106741905212])
    # torch runs the analytic signal in complex64; Typhon uses complex128 (diff ~1e-7)
    assert check(FrequencyShift(1.0, sfreq=64.0, max_delta_freq=3.0, random_state=10), -1.9878321643918753, [0.25476914644241333, 0.3654872179031372, 0.4727175831794739, 0.6928086876869202], [0.005392380058765411, 0.3746348023414612, 0.6988199949264526], 2e-5)
    assert check(Mixup(0.4, random_state=11), 1.7384709119796753, [0.5738527178764343, 0.36269524693489075, 0.11878621578216553, -0.053274452686309814], [-0.4014671742916107, -0.3456936478614807, -0.22369439899921417])
    X, y = data()
    xm, yy = Mixup(0.4, random_state=11).forward_mix(X, y)
    assert yy[1].tolist() == [1, 0, 0, 1]
    assert close(yy[2], [0.41078421473503113, 0.41078421473503113, 0.41078421473503113, 0.41078421473503113])
    c = Compose([TimeReverse(1.0, random_state=1), GaussianNoise(1.0, std=0.1, random_state=2)])
    oc, _ = c(X, y)
    assert close(oc.astype(float).sum(), -2.711723046377301, 1e-13)
    assert close(oc[2, 1, :3], [0.5123372077941895, 0.1409490406513214, -0.18460917472839355])

test_random_state_legacy()
test_transforms()
