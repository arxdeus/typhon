"""Generate test_legacy.py: expected values from real numpy's legacy
RandomState / module-level np.random.* functions.

Run with numpy 2.5.x:
    python test/numpy/random_tests/gen_legacy_expected.py > test/numpy/random_tests/test_legacy.py

Each case is an expression valid both in Python and Typhon, evaluated against a
fresh RandomState(seed). Kinds:
  'exact' : MT outputs / integer draws / pure arithmetic on MT doubles, ==
  'close' : involves libm transcendental functions (log/exp/pow/sqrt/cos/acos,
            loggam); compared with rel tol 2e-15 + tiny abs tol because Typhon
            and numpy may be linked against libm builds differing by a few ulp
            (e.g. vonmises on glibc vs Apple libm differs by ~1.1e-15 relative).
"""
import numpy as np

SEEDS = [0, 1, 42, 12345, 4294967295]

CASES = [
    # (expr, kind)
    ("rs.random_sample()", "exact"),
    ("rs.random_sample(5)", "exact"),
    ("rs.random_sample((2, 3))", "exact"),
    ("rs.random(4)", "exact"),
    ("rs.rand(3)", "exact"),
    ("rs.rand(2, 2)", "exact"),
    ("rs.uniform(-2.0, 3.0, 5)", "exact"),
    ("rs.uniform(np.array([0.0, 10.0, 20.0]), 30.0)", "exact"),
    ("rs.uniform(1.0, 2.0)", "exact"),
    ("rs.randn(5)", "close"),
    ("rs.randn(2, 3)", "close"),
    ("rs.standard_normal(7)", "close"),
    ("rs.normal(3.0, 2.0, 5)", "close"),
    ("rs.normal(np.array([0.0, 10.0, 100.0]), 1.0)", "close"),
    ("rs.normal(0.0, np.array([1.0, 2.0]), (3, 2))", "close"),
    ("rs.randint(0, 10, 8)", "exact"),
    ("rs.randint(5)", "exact"),
    ("rs.randint(-100, 100, (2, 3))", "exact"),
    ("rs.randint(0, 2**40, 4)", "exact"),
    ("rs.randint(-2**62, 2**62, 3)", "exact"),
    ("rs.randint(0, 7, 9, dtype=np.int32)", "exact"),
    ("rs.randint(0, 7, 9, dtype=np.int16)", "exact"),
    ("rs.randint(-5, 7, 9, dtype=np.int8)", "exact"),
    ("rs.randint(0, 200, 9, dtype=np.uint8)", "exact"),
    ("rs.randint(0, 60000, 9, dtype=np.uint16)", "exact"),
    ("rs.randint(0, 2**32, 5, dtype=np.uint32)", "exact"),
    ("rs.randint(0, 2**63, 3, dtype=np.uint64)", "exact"),
    ("rs.randint(0, 2, 9, dtype=bool)", "exact"),
    ("rs.randint(np.array([0, 10, 100]), np.array([5, 20, 1000]))", "exact"),
    ("rs.randint(0, np.array([3, 300, 30000]), (2, 3))", "exact"),
    ("rs.random_integers(5, size=6)", "exact"),
    ("rs.random_integers(-3, 3, 6)", "exact"),
    ("rs.tomaxint(3)", "exact"),
    ("rs.permutation(10)", "exact"),
    ("rs.permutation(np.array([1.5, 2.5, 3.5, 4.5]))", "exact"),
    ("rs.permutation(np.arange(12).reshape(4, 3))", "exact"),
    ("rs.choice(10, 5)", "exact"),
    ("rs.choice(10, 5, replace=False)", "exact"),
    ("rs.choice(np.array([10, 20, 30, 40]), (2, 2))", "exact"),
    ("rs.choice(5, 6, p=np.array([0.1, 0.2, 0.3, 0.2, 0.2]))", "exact"),
    ("rs.choice(5, 3, replace=False, p=np.array([0.1, 0.2, 0.3, 0.2, 0.2]))", "exact"),
    ("rs.choice(7)", "exact"),
    ("rs.exponential(2.0, 5)", "close"),
    ("rs.standard_exponential(5)", "close"),
    ("rs.standard_gamma(0.5, 5)", "close"),
    ("rs.standard_gamma(1.0, 3)", "close"),
    ("rs.standard_gamma(3.5, 5)", "close"),
    ("rs.standard_gamma(np.array([0.3, 2.0, 9.0]))", "close"),
    ("rs.gamma(2.0, 3.0, 5)", "close"),
    ("rs.beta(0.5, 0.7, 5)", "close"),
    ("rs.beta(2.0, 3.0, 5)", "close"),
    ("rs.binomial(10, 0.3, 8)", "exact"),
    ("rs.binomial(1000, 0.4, 8)", "exact"),
    ("rs.binomial(1000, 0.9, 8)", "exact"),
    ("rs.binomial(20, 0.9, 8)", "exact"),
    ("rs.binomial(np.array([5, 50, 500]), 0.5)", "exact"),
    ("rs.binomial(100, np.array([0.1, 0.5, 0.95]), (2, 3))", "exact"),
    ("rs.poisson(3.0, 8)", "exact"),
    ("rs.poisson(25.0, 8)", "exact"),
    ("rs.poisson(np.array([0.5, 5.0, 50.0]))", "exact"),
    ("rs.geometric(0.5, 8)", "exact"),
    ("rs.geometric(0.05, 8)", "exact"),
    ("rs.chisquare(3.0, 5)", "close"),
    ("rs.f(3.0, 7.0, 5)", "close"),
    ("rs.standard_t(4.0, 5)", "close"),
    ("rs.lognormal(0.5, 0.3, 5)", "close"),
    ("rs.laplace(1.0, 2.0, 5)", "close"),
    ("rs.logistic(1.0, 2.0, 5)", "close"),
    ("rs.gumbel(1.0, 2.0, 5)", "close"),
    ("rs.standard_cauchy(5)", "close"),
    ("rs.pareto(3.0, 5)", "close"),
    ("rs.power(3.0, 5)", "close"),
    ("rs.rayleigh(2.0, 5)", "close"),
    ("rs.triangular(0.0, 1.0, 3.0, 5)", "close"),
    ("rs.triangular(np.array([0.0, 1.0]), 2.0, 4.0)", "close"),
    ("rs.vonmises(0.5, 2.0, 5)", "close"),
    ("rs.vonmises(0.0, 1e-9, 3)", "close"),
    ("rs.wald(1.5, 2.0, 5)", "close"),
    ("rs.weibull(1.5, 5)", "close"),
    ("rs.zipf(2.5, 8)", "exact"),
    ("rs.logseries(0.7, 8)", "exact"),
    ("rs.negative_binomial(5.0, 0.4, 8)", "exact"),
    ("rs.hypergeometric(10, 15, 5, 8)", "exact"),
    ("rs.hypergeometric(100, 200, 50, 8)", "exact"),
    ("rs.hypergeometric(np.array([5, 50]), 30, 12)", "exact"),
    ("rs.noncentral_chisquare(3.0, 2.0, 5)", "close"),
    ("rs.noncentral_chisquare(0.5, 2.0, 5)", "close"),
    ("rs.noncentral_f(3.0, 5.0, 1.5, 5)", "close"),
    ("rs.multinomial(20, np.array([0.2, 0.3, 0.5]))", "exact"),
    ("rs.multinomial(100, np.array([0.1, 0.6, 0.3]), 3)", "exact"),
    ("rs.dirichlet(np.array([1.0, 2.0, 3.0]), 2)", "close"),
    ("rs.multivariate_normal(np.array([0.0, 1.0]), np.array([[1.0, 0.5], [0.5, 2.0]]), 3)", "mvn"),
]

# sequences of calls against the module-level API (global RandomState)
MODULE = [
    "np.random.rand(3)",
    "np.random.randn(3)",
    "np.random.randint(0, 10, 4)",
    "np.random.normal(size=4)",
    "np.random.uniform(0.0, 5.0, 3)",
    "np.random.choice(10, 4)",
    "np.random.permutation(6)",
    "np.random.random_sample(2)",
    "np.random.ranf(2)",
    "np.random.sample(2)",
    "np.random.random_integers(4, size=3)",
    "np.random.standard_normal(3)",
    "np.random.binomial(10, 0.5, 3)",
    "np.random.poisson(4.0, 3)",
    "np.random.beta(2.0, 2.0, 2)",
    "np.random.gamma(2.0, 1.0, 2)",
]


def lit(v):
    if isinstance(v, np.ndarray):
        if v.dtype == bool:
            body = repr(v.tolist())
            return f"np.array({body})"
        dt = v.dtype
        body = repr(v.tolist())
        if np.issubdtype(dt, np.floating):
            return f"np.array({body}, dtype=float)"
        tname = {np.dtype('int64'): 'int', np.dtype('int32'): 'np.int32',
                 np.dtype('int16'): 'np.int16', np.dtype('int8'): 'np.int8',
                 np.dtype('uint8'): 'np.uint8', np.dtype('uint16'): 'np.uint16',
                 np.dtype('uint32'): 'np.uint32', np.dtype('uint64'): 'np.uint64'}[dt]
        return f"np.array({body}, dtype={tname})"
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return repr(int(v))


def main():
    out = []
    w = out.append
    w("# GENERATED by gen_legacy_expected.py from numpy %s. Do not edit." % np.__version__)
    w("import numpy as np")
    w("")
    w("def _close(a: float, b: float):")
    w("    if a == b:")
    w("        return True")
    w("    return abs(a - b) <= 2e-15 * max(abs(a), abs(b)) + 1e-300")
    w("")
    w("def check_exact(got, exp, what: str):")
    w("    if isinstance(exp, np.ndarray):")
    w("        ok = got.shape == exp.shape and (got == exp).all()")
    w("    else:")
    w("        ok = got == exp")
    w("    if not ok:")
    w("        print('MISMATCH', what, got, exp)")
    w("    return ok")
    w("")
    w("def check_close(got, exp, what: str):")
    w("    ok = True")
    w("    if isinstance(exp, np.ndarray):")
    w("        if got.shape != exp.shape:")
    w("            ok = False")
    w("        else:")
    w("            g = got.ravel()")
    w("            e = exp.ravel()")
    w("            for i in range(e.size):")
    w("                if not _close(float(g[i]), float(e[i])):")
    w("                    ok = False")
    w("    else:")
    w("        ok = _close(float(got), float(exp))")
    w("    if not ok:")
    w("        print('MISMATCH', what, got, exp)")
    w("    return ok")
    w("")
    w("def check_mvn(got, exp, what: str):")
    w("    # SVD factor may differ from LAPACK in the last bits")
    w("    ok = got.shape == exp.shape and np.allclose(got, exp, rtol=1e-12, atol=1e-12)")
    w("    if not ok:")
    w("        print('MISMATCH', what, got, exp)")
    w("    return ok")
    w("")

    n = 0
    for seed in SEEDS:
        w("@test")
        w(f"def test_randomstate_seed_{seed}():")
        for expr, kind in CASES:
            rs = np.random.RandomState(seed)
            v = eval(expr, {"np": np, "rs": rs})
            # follow up with draws to check the stream position and gauss cache
            tail = rs.random_sample(2)
            tailn = rs.standard_normal()
            w(f"    rs = np.random.RandomState({seed})")
            w(f"    assert check_{kind}({expr}, {lit(np.asarray(v) if isinstance(v, np.ndarray) else v)}, {expr!r})")
            w(f"    assert check_exact(rs.random_sample(2), {lit(tail)}, 'tail ' + {expr!r})")
            w(f"    assert check_close(rs.standard_normal(), {lit(tailn)}, 'tailn ' + {expr!r})")
            n += 1
        w("")
        w(f"test_randomstate_seed_{seed}()")
        w("")

    # bytes
    w("@test")
    w("def test_bytes():")
    for seed in SEEDS:
        rs = np.random.RandomState(seed)
        b = rs.bytes(11)
        codes = list(b)
        w(f"    b = np.random.RandomState({seed}).bytes(11)")
        w(f"    assert [ord(c) for c in b] == {codes}")
    w("")
    w("test_bytes()")
    w("")

    # shuffle (in place) incl. rows, lists
    w("@test")
    w("def test_shuffle():")
    for seed in SEEDS[:3]:
        rs = np.random.RandomState(seed)
        a = np.arange(10.0); rs.shuffle(a)
        m = np.arange(15).reshape(5, 3); rs.shuffle(m)
        lst = list(range(8)); rs.shuffle(lst)
        w(f"    rs = np.random.RandomState({seed})")
        w("    a = np.arange(10.0)")
        w("    rs.shuffle(a)")
        w(f"    assert check_exact(a, {lit(a)}, 'shuffle 1d')")
        w("    m = np.arange(15).reshape(5, 3)")
        w("    rs.shuffle(m)")
        w(f"    assert check_exact(m, {lit(m)}, 'shuffle rows')")
        w("    lst = list(range(8))")
        w("    rs.shuffle(lst)")
        w(f"    assert lst == {lst}")
    w("")
    w("test_shuffle()")
    w("")

    # state
    rs = np.random.RandomState(7)
    rs.standard_normal()  # leaves a cached gaussian
    st = rs.get_state()
    after = rs.standard_normal(3)
    w("@test")
    w("def test_state():")
    w("    rs = np.random.RandomState(7)")
    w("    rs.standard_normal()")
    w("    st = rs.get_state()")
    w("    assert st[0] == 'MT19937'")
    w(f"    assert st[1].size == 624 and int(st[1][0]) == {int(st[1][0])} and int(st[1][623]) == {int(st[1][623])}")
    w(f"    assert st[2] == {st[2]}")
    w(f"    assert st[3] == {st[3]}")
    w(f"    assert st[4] == {float(st[4])!r}")
    w(f"    assert check_close(rs.standard_normal(3), {lit(after)}, 'after')")
    w("    rs2 = np.random.RandomState(99)")
    w("    rs2.set_state(st)")
    w(f"    assert check_close(rs2.standard_normal(3), {lit(after)}, 'restored')")
    w("    rs.seed(7)")
    w("    st0 = rs.get_state()")
    w("    assert st0[2] == 624 and st0[3] == 0 and st0[4] == 0.0")
    # array seed (init_by_array)
    rs = np.random.RandomState([1, 2, 3])
    v = rs.random_sample(3)
    w(f"    assert check_exact(np.random.RandomState([1, 2, 3]).random_sample(3), {lit(v)}, 'array seed')")
    w("")
    w("test_state()")
    w("")

    # module level
    for seed in [0, 42]:
        np.random.seed(seed)
        w("@test")
        w(f"def test_module_seed_{seed}():")
        w(f"    np.random.seed({seed})")
        for expr in MODULE:
            v = eval(expr, {"np": np})
            w(f"    assert check_close({expr}, {lit(v)}, {expr!r})")
        st = np.random.get_state()
        w("    st = np.random.get_state()")
        w(f"    assert st[2] == {st[2]} and st[3] == {st[3]}")
        w("")
        w(f"test_module_seed_{seed}()")
        w("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
