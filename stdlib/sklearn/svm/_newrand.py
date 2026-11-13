# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn/svm/src/newrand/newrand.h: std::mt19937 seeded with set_seed and
# bounded_rand_int (Lemire's tweaked post-processor), shared by the libsvm
# and liblinear ports. _fma mirrors the fused multiply-adds that clang emits
# (-ffp-contract=on) for `a*b + c` inside one expression in those sources.

@llvm
def _fma(a: float, b: float, c: float) -> float:
    declare double @llvm.fma.f64(double, double, double)
    %0 = call double @llvm.fma.f64(double %a, double %b, double %c)
    ret double %0

def _f32(x: float) -> float:
    return float(float32(x))

class _StdMT:
    mt: List[u32]
    pos: int

    def __init__(self, seed: u32):
        self.mt = [u32(0)] * 624
        self.mt[0] = seed
        for i in range(1, 624):
            prev = self.mt[i - 1]
            self.mt[i] = u32(1812433253) * (prev ^ (prev >> u32(30))) + u32(i)
        self.pos = 624

    def next(self) -> u32:
        if self.pos >= 624:
            mt = self.mt
            for k in range(624):
                y = (mt[k] & u32(0x80000000)) | (mt[(k + 1) % 624] & u32(0x7fffffff))
                v = mt[(k + 397) % 624] ^ (y >> u32(1))
                if y & u32(1):
                    v ^= u32(0x9908b0df)
                mt[k] = v
            self.pos = 0
        y = self.mt[self.pos]
        self.pos += 1
        y ^= (y >> u32(11))
        y ^= (y << u32(7)) & u32(0x9d2c5680)
        y ^= (y << u32(15)) & u32(0xefc60000)
        y ^= (y >> u32(18))
        return y

    def bounded(self, rng: u32) -> u32:
        x = self.next()
        m = u64(x) * u64(rng)
        l = u32(m)
        if l < rng:
            t = u32(0) - rng
            if t >= rng:
                t -= rng
                if t >= rng:
                    t %= rng
            while l < t:
                x = self.next()
                m = u64(x) * u64(rng)
                l = u32(m)
        return u32(m >> u64(32))

    def rint(self, n: int) -> int:
        return int(self.bounded(u32(n)))
