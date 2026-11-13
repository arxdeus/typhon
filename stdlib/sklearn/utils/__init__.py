# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

class _LegacyMT:
    """MT19937 with numpy.random.RandomState(int) seeding (init_genrand)."""
    mt: List[u32]
    pos: int

    def __init__(self, seed: int):
        if seed < 0 or seed > 0xffffffff:
            raise ValueError("Seed must be between 0 and 2**32 - 1")
        self.mt = [u32(0)] * 624
        s = u32(seed)
        for i in range(624):
            self.mt[i] = s
            s = u32(1812433253) * (s ^ (s >> u32(30))) + u32(i + 1)
        self.pos = 624

    def _gen(self):
        mt = self.mt
        for k in range(624):
            y = (mt[k] & u32(0x80000000)) | (mt[(k + 1) % 624] & u32(0x7fffffff))
            v = mt[(k + 397) % 624] ^ (y >> u32(1))
            if y & u32(1):
                v ^= u32(0x9908b0df)
            mt[k] = v
        self.pos = 0

    def genrand_int32(self) -> u32:
        if self.pos >= 624:
            self._gen()
        y = self.mt[self.pos]
        self.pos += 1
        y ^= (y >> u32(11))
        y ^= (y << u32(7)) & u32(0x9d2c5680)
        y ^= (y << u32(15)) & u32(0xefc60000)
        y ^= (y >> u32(18))
        return y

    def random_sample(self) -> float:
        a = int(self.genrand_int32() >> u32(5))
        b = int(self.genrand_int32() >> u32(6))
        return (a * 67108864.0 + b) / 9007199254740992.0

class _LegacyRS:
    """numpy.random.RandomState(int) subset: shuffle / permutation / choice
    without replacement (all built on the legacy masked bounded draw)."""
    g: _LegacyMT

    def __init__(self, seed: int):
        self.g = _LegacyMT(seed)

    def _interval(self, mx: int) -> int:
        # random_interval(max): masked rejection sampling in [0, mx]
        if mx == 0:
            return 0
        mask = u32(mx)
        mask |= mask >> u32(1); mask |= mask >> u32(2); mask |= mask >> u32(4)
        mask |= mask >> u32(8); mask |= mask >> u32(16)
        v = self.g.genrand_int32() & mask
        while v > u32(mx):
            v = self.g.genrand_int32() & mask
        return int(v)

    def shuffle(self, a):
        # in place, Fisher-Yates from the end (RandomState.shuffle, 1-d)
        i = a.shape[0] - 1
        while i > 0:
            j = self._interval(i)
            t = a[i]; a[i] = a[j]; a[j] = t
            i -= 1

    def permutation(self, n: int):
        import numpy as np
        a = np.arange(n)
        self.shuffle(a)
        return a

    def permutation_of(self, arr):
        a = arr.copy()
        self.shuffle(a)
        return a

    def choice_noreplace(self, pop, size: int):
        # RandomState.choice(pop, size, replace=False) = permutation(len)[:size]
        idx = self.permutation(pop.shape[0])[:size]
        return pop[idx]

class _LegacyGauss:
    """RandomState(seed).normal / standard_normal: legacy_gauss (polar
    Box-Muller with a cached second value), uniforms from random_sample."""
    g: _LegacyMT
    has: bool
    cached: float

    def __init__(self, seed: int):
        self.g = _LegacyMT(seed)
        self.has = False
        self.cached = 0.0

    def gauss(self) -> float:
        import math
        if self.has:
            self.has = False
            return self.cached
        x1 = 0.0; x2 = 0.0; r2 = 0.0
        while True:
            x1 = 2.0 * self.g.random_sample() - 1.0
            x2 = 2.0 * self.g.random_sample() - 1.0
            r2 = x1 * x1 + x2 * x2
            if r2 < 1.0 and r2 != 0.0:
                break
        f = math.sqrt(-2.0 * math.log(r2) / r2)
        self.cached = f * x1
        self.has = True
        return f * x2

    def normal(self, n: int, m: int):
        import numpy as np
        out = np.empty((n, m))
        for i in range(n):
            for j in range(m):
                out[i, j] = self.gauss()
        return out

# ------------------------------------------------------------ public API

def Bunch(**kwargs):
    """Container exposing keys as attributes (b.key). Typhon: a statically
    typed named tuple, so keys are fixed at construction."""
    return kwargs

from numpy.random.legacy import check_random_state
from .validation import (check_array, check_X_y, check_consistent_length, column_or_1d,
                         check_is_fitted, check_scalar, check_symmetric, indexable,
                         assert_all_finite, as_float_array, check_non_negative,
                         has_fit_parameter, _check_sample_weight, _num_samples,
                         NotFittedError)
from .multiclass import unique_labels, type_of_target, is_multilabel, check_classification_targets
from .class_weight import compute_class_weight, compute_sample_weight
from .extmath import safe_sqr, safe_sparse_dot, row_norms, squared_norm, stable_cumsum
from ._indexing import (_safe_indexing, resample, shuffle, gen_batches, gen_even_slices,
                        murmurhash3_32, get_chunk_n_rows, _approximate_mode)
from .validation import validate_data
from ._tags import Tags, InputTags, TargetTags, TransformerTags, ClassifierTags, RegressorTags, get_tags
from ._misc import FLOAT_DTYPES, is_scalar_nan, safe_mask, check_memory, Memory, all_estimators
