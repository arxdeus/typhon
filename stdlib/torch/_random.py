# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Bit-exact port of ATen's CPU random engine (MT19937, aten/src/ATen/core/
# MT19937RNGEngine.h) and of the CPU generator wrapper (CPUGeneratorImpl):
# random() returns one 32-bit draw, random64() combines two draws as
# (first << 32) | second, and the Box-Muller normal cache is kept per dtype.

_N = 624
_M = 397

class _MTState:
    seed: int
    left: int
    next: int
    state: Ptr[u32]
    next_float: Optional[float32]
    next_double: Optional[float]

    def __init__(self):
        self.seed = 0
        self.left = 1
        self.next = 0
        self.state = Ptr[u32](_N)
        self.next_float = None
        self.next_double = None

    def copy(self) -> _MTState:
        s = _MTState()
        s.seed = self.seed
        s.left = self.left
        s.next = self.next
        for i in range(_N):
            s.state[i] = self.state[i]
        s.next_float = self.next_float
        s.next_double = self.next_double
        return s


class Generator:
    """torch.Generator for the CPU device (MT19937, bit-identical to torch)."""
    _s: _MTState
    device: str

    def __init__(self, device: str = "cpu"):
        self._s = _MTState()
        self.device = device
        self._seed_engine(67280421310721)

    def _seed_engine(self, seed: int):
        s = self._s
        s.seed = seed
        st = s.state
        st[0] = u32(seed & 0xffffffff)
        for j in range(1, _N):
            prev = st[j - 1]
            st[j] = u32(1812433253) * (prev ^ (prev >> u32(30))) + u32(j)
        s.left = 1
        s.next = 0
        s.next_float = None
        s.next_double = None

    def manual_seed(self, seed: int) -> Generator:
        self._seed_engine(seed)
        return self

    def initial_seed(self) -> int:
        return self._s.seed

    def seed(self) -> int:
        from time import time_ns
        sd = time_ns() & 0x7fffffffffffffff
        self._seed_engine(sd)
        return sd

    def get_state(self) -> _MTState:
        return self._s.copy()

    def set_state(self, state: _MTState):
        self._s = state.copy()

    def _next_state(self):
        s = self._s
        p = s.state
        s.left = _N
        s.next = 0
        UM = u32(0x80000000)
        LM = u32(0x7fffffff)
        MA = u32(0x9908b0df)
        i = 0
        while i < _N - _M:
            u = p[i]
            v = p[i + 1]
            t = ((u & UM) | (v & LM)) >> u32(1)
            if v & u32(1):
                t ^= MA
            p[i] = p[i + _M] ^ t
            i += 1
        while i < _N - 1:
            u = p[i]
            v = p[i + 1]
            t = ((u & UM) | (v & LM)) >> u32(1)
            if v & u32(1):
                t ^= MA
            p[i] = p[i + _M - _N] ^ t
            i += 1
        u = p[_N - 1]
        v = p[0]
        t = ((u & UM) | (v & LM)) >> u32(1)
        if v & u32(1):
            t ^= MA
        p[_N - 1] = p[_M - 1] ^ t

    def random(self) -> u32:
        s = self._s
        s.left -= 1
        if s.left == 0:
            self._next_state()
        y = s.state[s.next]
        s.next += 1
        y ^= (y >> u32(11))
        y ^= (y << u32(7)) & u32(0x9d2c5680)
        y ^= (y << u32(15)) & u32(0xefc60000)
        y ^= (y >> u32(18))
        return y

    def random64(self) -> u64:
        a = self.random()
        b = self.random()
        return (u64(a) << u64(32)) | u64(b)

    # uniform_real_distribution<float>(0,1)
    def _uniform_f32(self) -> float32:
        v = self.random()
        return float32(int(v & u32((1 << 24) - 1))) * float32(1.0 / 16777216.0)

    def _uniform_f64(self) -> float:
        v = self.random64()
        return float(int(v & u64((1 << 53) - 1))) * (1.0 / 9007199254740992.0)

    # at::normal_distribution<double>(mean, std) with sample caching
    def _normal_f64(self, mean: float, std: float) -> float:
        s = self._s
        if s.next_double is not None:
            r = s.next_double
            s.next_double = None
            return r * std + mean
        from math import sqrt, log1p, sin, cos
        u1 = self._uniform_f64()
        u2 = self._uniform_f64()
        r = sqrt(-2.0 * log1p(-u2))
        theta = 2.0 * 3.141592653589793 * u1
        s.next_double = r * sin(theta)
        return r * cos(theta) * std + mean


default_generator = Generator()
