# Generates test_ewm.py (ewm, rolling cov/corr/quantile) from real pandas 3.1.1.
import numpy as np, pandas as pd, os
D = os.path.dirname(__file__)
def make_F(n, m, seed=12345):
    F = np.empty((n, m)); s = seed
    for i in range(n):
        for j in range(m):
            s = (s * 1103515245 + 12345) % 2147483648
            F[i, j] = (s % 20001 - 10000) * 0.000123 * (j + 1)
    return F
def lit(a): return repr(np.asarray(a, dtype=float).ravel().tolist())
F = make_F(30, 3); F[4, 1] = np.nan; F[5, 1] = np.nan; F[12, 0] = np.nan
df = pd.DataFrame(F, columns=["a", "b", "c"])
L = []; T = []
def add(name, res, call, tol="1e-12"):
    L.append(f"E_{name} = {lit(res.to_numpy())}")
    T.append(f"    assert close(({call}).to_numpy(), E_{name}, {tol})")
for adj in (True, False):
    for ign in (False, True):
        k = f"{int(adj)}{int(ign)}"
        py = f"adjust={adj}, ignore_na={ign}"
        add("mean"+k, df.ewm(com=1.5, adjust=adj, ignore_na=ign).mean(), f"df.ewm(com=1.5, {py}).mean()")
        if adj: add("sum"+k, df.ewm(alpha=0.3, adjust=adj, ignore_na=ign).sum(), f"df.ewm(alpha=0.3, {py}).sum()")
        add("var"+k, df.ewm(span=5, adjust=adj, ignore_na=ign, min_periods=2).var(), f"df.ewm(span=5, min_periods=2, {py}).var()")
        add("std"+k, df.ewm(halflife=3, adjust=adj, ignore_na=ign).std(bias=True), f"df.ewm(halflife=3, {py}).std(True)")
add("cov", df[["a", "b"]].ewm(com=2.0).cov(df["c"]), 'df[["a", "b"]].ewm(com=2.0).cov(df["c"])')
add("corr", df[["a", "b"]].ewm(com=2.0).corr(df["c"]), 'df[["a", "b"]].ewm(com=2.0).corr(df["c"])')
add("rcov", df[["a", "b"]].rolling(6, min_periods=3).cov(df["c"]), 'df[["a", "b"]].rolling(6, min_periods=3).cov(df["c"])')
add("rcorr", df[["a", "b"]].rolling(6, min_periods=3).corr(df["c"]), 'df[["a", "b"]].rolling(6, min_periods=3).corr(df["c"])')
for ip in ("linear", "lower", "higher", "nearest", "midpoint"):
    add("rq"+ip, df.rolling(7, min_periods=4).quantile(0.37, interpolation=ip), f'df.rolling(7, min_periods=4).quantile(0.37, "{ip}")')
src = open(os.path.join(D, "test_rolling.py")).read()
head = src.split("@test")[0]
head = head.split("def _r(")[0]
head += '''nan = float("nan")

def close(got, exp: List[float], tol: float) -> bool:
    ok = True
    if got.size != len(exp):
        print("size mismatch", got.size, len(exp))
        return False
    g = got.ravel()
    for i in range(len(exp)):
        a = g[i]
        b = exp[i]
        if a != a and b != b:
            continue
        if abs(a - b) > tol + tol * abs(b) or (a != a) != (b != b):
            print("mismatch at", i, a, b)
            ok = False
    return ok

'''
body = "F = make_F(30, 3)\nF[4, 1] = float(\"nan\")\nF[5, 1] = float(\"nan\")\nF[12, 0] = float(\"nan\")\ndf = pd.DataFrame(F, columns=[\"a\", \"b\", \"c\"])\n" + "\n".join(L) + "\n\n@test\ndef test_ewm():\n" + "\n".join(T) + "\ntest_ewm()\n"
open(os.path.join(D, "test_ewm.py"), "w").write(head + body)
