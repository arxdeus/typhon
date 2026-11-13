# Ported from CPython 3.14 random (PSF); expected values from the reference interpreter
import random as R

@test
def test_seed_streams():
    R.seed(12345)
    assert R.random() == 0.41661987254534116
    assert R.getrandbits(40) == 897691841093
    R.seed("hello")
    assert R.random() == 0.3537754404730722
    R.seed(b"abc")
    assert R.random() == 0.7720246314157545

@test
def test_binomial_bytes():
    R.seed(77)
    assert R.binomialvariate(10, 0.3) == 4
    assert R.binomialvariate(100, 0.4) == 40
    assert R.binomialvariate(1000, 0.5) == 505
    assert R.binomialvariate(50, 0.9) == 45
    assert R.randbytes(7) == b'\xfb\x0c8\xd0\x7f\xf40'

test_seed_streams()
test_binomial_bytes()
