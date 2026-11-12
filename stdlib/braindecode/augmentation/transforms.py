# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# Transform classes. Codon is statically typed, so every transform is the same
# class `Transform` with a kind tag and its parameters; the braindecode names
# (TimeReverse, SignFlip, ...) are constructors returning a Transform. That
# lets Compose hold a plain List[Transform]. Batches are
# ndarray[float32, 3] and int labels ndarray[int, 1]. Mixup returns the
# mixed batch; its (y_a, y_b, lam) targets are available via `forward_mix`.

import numpy as np
from .._random_state import RandomState, check_random_state
from . import functional as F

_IDENT = 0
_TREV = 1
_SFLIP = 2
_FTS = 3
_CDROP = 4
_CSHUF = 5
_GAUSS = 6
_STM = 7
_BSTOP = 8
_FSHIFT = 9
_MIXUP = 10
_CSYM = 11
_COMPOSE = 12

_NAMES = ["IdentityTransform", "TimeReverse", "SignFlip", "FTSurrogate", "ChannelsDropout",
          "ChannelsShuffle", "GaussianNoise", "SmoothTimeMask", "BandstopFilter",
          "FrequencyShift", "Mixup", "ChannelsSymmetry", "Compose"]


class Transform:
    kind: int
    _probability: float
    rng: RandomState
    p1: float      # phase_noise_magnitude | p_drop | p_shuffle | std | sfreq | alpha
    p2: float      # bandwidth | max_delta_freq
    p3: float      # max_freq
    flag: bool     # channel_indep | beta_per_sample
    n: int         # mask_len_samples
    perm: List[int]
    transforms: List[Transform]

    def __init__(self, kind: int, probability: float = 1.0, random_state = None):
        if not (0.0 <= probability <= 1.0):
            raise AssertionError("probability should be between 0 and 1.")
        self.kind = kind
        self._probability = probability
        self.rng = check_random_state(random_state)
        self.p1 = 0.0
        self.p2 = 0.0
        self.p3 = 0.0
        self.flag = False
        self.n = 0
        self.perm = List[int]()
        self.transforms = List[Transform]()

    @property
    def probability(self) -> float:
        return self._probability

    def __repr__(self) -> str:
        return f"{_NAMES[self.kind]}(probability={self._probability})"

    def _op(self, X: np.ndarray[np.float32, 3], y: np.ndarray[int, 1]):
        k = self.kind
        if k == _IDENT:
            return X, y
        if k == _TREV:
            return F.time_reverse(X, y)
        if k == _SFLIP:
            return F.sign_flip(X, y)
        if k == _FTS:
            return F.ft_surrogate(X, y, self.p1, self.flag, self.rng)
        if k == _CDROP:
            return F.channels_dropout(X, y, self.p1, self.rng)
        if k == _CSHUF:
            return F.channels_shuffle(X, y, self.p1, self.rng)
        if k == _GAUSS:
            return F.gaussian_noise(X, y, self.p1, self.rng)
        if k == _CSYM:
            return F.channels_permute(X, y, self.perm)
        if k == _STM:
            T = X.shape[-1]
            u = self.rng.uniform(low=0, high=1, size=X.shape[0])
            ms = u * float(T - self.n)
            return F.smooth_time_mask(X, y, ms, self.n)
        if k == _BSTOP:
            fr = self.rng.uniform(low=1 + self.p2 / 2, high=self.p3 - 1 - self.p2 / 2, size=X.shape[0])
            return F.bandstop_filter(X, y, self.p1, self.p2, fr)
        if k == _FSHIFT:
            u = self.rng.uniform(size=X.shape[0])
            d = u * 2 * self.p2 - self.p2
            return F.frequency_shift(X, y, d, self.p1)
        if k == _MIXUP:
            Xm, yy = self._mix(X, y)
            return Xm, y
        raise ValueError("unknown transform")

    def _mix(self, X, y):
        B = X.shape[0]
        lam = np.ones(B)
        if self.p1 > 0:
            if self.flag:
                lam = self.rng.beta(self.p1, self.p1, B)
            else:
                lam = lam * self.rng.beta(self.p1, self.p1)
        idx = self.rng.permutation(B)
        return F.mixup(X, y, lam, idx)

    def _get_mask(self, B: int) -> List[bool]:
        u = self.rng.uniform(size=B)
        return [self._probability > float(u[i]) for i in range(B)]

    def forward(self, X: np.ndarray[np.float32, 3], y: np.ndarray[int, 1]):
        """Returns (X_out, y_out). Per-example application with probability."""
        if self.kind == _COMPOSE:
            for t in self.transforms:
                X, y = t.forward(X, y)
            return X, y
        out_X = X.copy()
        out_y = y.copy()
        mask = self._get_mask(X.shape[0])
        sel = [i for i in range(len(mask)) if mask[i]]
        if len(sel) == X.shape[0]:
            return self._op(out_X, out_y)
        elif len(sel) > 0:
            si = np.array(sel, dtype=int)
            tX, ty = self._op(out_X[si].copy(), out_y[si].copy())
            for k in range(len(sel)):
                out_X[sel[k]] = tX[k]
                out_y[sel[k]] = ty[k]
        return out_X, out_y

    def forward_mix(self, X: np.ndarray[np.float32, 3], y: np.ndarray[int, 1]):
        """Mixup with mixed targets: (X_mix, (y_a, y_b, lam)), same draws as forward."""
        self._get_mask(X.shape[0])
        return self._mix(X.copy(), y)

    def __call__(self, X: np.ndarray[np.float32, 3], y: np.ndarray[int, 1]):
        return self.forward(X, y)

    def __call__(self, X: np.ndarray[np.float32, 2]):
        Xb = X.reshape((1, X.shape[0], X.shape[1])).copy()
        o, _ = self.forward(Xb, np.zeros(1, dtype=int))
        return o.reshape((X.shape[0], X.shape[1])).copy()


def IdentityTransform():
    return Transform(_IDENT)

def TimeReverse(probability: float, random_state = None):
    return Transform(_TREV, probability, random_state)

def SignFlip(probability: float, random_state = None):
    return Transform(_SFLIP, probability, random_state)

def FTSurrogate(probability: float, phase_noise_magnitude: float = 1.0, channel_indep: bool = False,
                random_state = None):
    t = Transform(_FTS, probability, random_state)
    t.p1 = phase_noise_magnitude
    t.flag = channel_indep
    return t

def ChannelsDropout(probability: float, p_drop: float = 0.2, random_state = None):
    t = Transform(_CDROP, probability, random_state)
    t.p1 = p_drop
    return t

def ChannelsShuffle(probability: float, p_shuffle: float = 0.2, random_state = None):
    t = Transform(_CSHUF, probability, random_state)
    t.p1 = p_shuffle
    return t

def GaussianNoise(probability: float, std: float = 0.1, random_state = None):
    t = Transform(_GAUSS, probability, random_state)
    t.p1 = std
    return t

def SmoothTimeMask(probability: float, mask_len_samples: int = 100, random_state = None):
    t = Transform(_STM, probability, random_state)
    t.n = mask_len_samples
    return t

def BandstopFilter(probability: float, sfreq: float, bandwidth: float = 1.0,
                   max_freq: Optional[float] = None, random_state = None):
    t = Transform(_BSTOP, probability, random_state)
    nyq = sfreq / 2
    mf = nyq if (max_freq is None or max_freq.__val__() > nyq) else max_freq.__val__()
    if not bandwidth < mf - 2:
        raise AssertionError(f"`bandwidth` needs to be smaller than max_freq - 2={mf - 2}")
    t.p1 = sfreq
    t.p2 = bandwidth
    t.p3 = mf
    return t

def FrequencyShift(probability: float, sfreq: float, max_delta_freq: float = 2.0, random_state = None):
    t = Transform(_FSHIFT, probability, random_state)
    t.p1 = sfreq
    t.p2 = max_delta_freq
    return t

def Mixup(alpha: float, beta_per_sample: bool = False, random_state = None):
    t = Transform(_MIXUP, 1.0, random_state)
    t.p1 = alpha
    t.flag = beta_per_sample
    return t

def ChannelsSymmetry(probability: float, ordered_ch_names: List[str], random_state = None):
    t = Transform(_CSYM, probability, random_state)
    perm = List[int]()
    for idx in range(len(ordered_ch_names)):
        ch = ordered_ch_names[idx]
        pos = idx
        d = "".join([c for c in ch if c.isdigit()])
        if len(d) > 0:
            di = int(d)
            sym = di - 1 if di % 2 == 0 else di + 1
            sym_name = ch.replace(d, str(sym))
            if sym_name in ordered_ch_names:
                pos = ordered_ch_names.index(sym_name)
        perm.append(pos)
    t.perm = perm
    return t

def Compose(transforms: List[Transform]):
    t = Transform(_COMPOSE)
    t.transforms = transforms
    return t


class AugmentedDataLoader[D]:
    """Iterates (X, y) float32/int batches of a windows dataset and applies the
    transforms to each batch. shuffle uses the given RandomState (or a torch
    generator once available through `order`)."""
    dataset: D
    transform: Transform
    batch_size: int
    shuffle: bool
    drop_last: bool
    rng: RandomState

    def __init__(self, dataset, transforms = None, batch_size: int = 1, shuffle: bool = False,
                 drop_last: bool = False, random_state = None, **kwargs):
        self.dataset = dataset
        if isinstance(transforms, NoneType):
            self.transform = IdentityTransform()
        elif isinstance(transforms, Transform):
            self.transform = transforms
        else:
            self.transform = Compose(transforms)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = check_random_state(random_state)

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        k = 0
        while k < n:
            e = min(k + self.batch_size, n)
            if self.drop_last and e - k < self.batch_size:
                break
            X, y = self.dataset.get_xy([int(order[i]) for i in range(k, e)])
            yield self.transform.forward(X, y)
            k = e
