# Generates test_groupby2.py (groupby cumcount/ngroup/nth/shift/diff/cum*/rank/rolling/quantile/skew) from pandas 3.1.1.
import numpy as np, pandas as pd, os
D = os.path.dirname(__file__)
head = open(os.path.join(D, "test_stats2.py")).read().split("F = List[float]")[0]
G = [((i * 19 + 5) % 17 - 6) * 0.53 for i in range(30)]
H = [((i * 7 + 3) % 13 - 4) * 0.71 for i in range(30)]
K = [i % 4 for i in range(30)]
df = pd.DataFrame({"k": K, "g": G, "h": H}); df.loc[5, "g"] = np.nan
gb = df.groupby("k")
L = []; T = []
def add(name, r, call):
    a = np.asarray(r.to_numpy() if hasattr(r, "to_numpy") else r, dtype=float).ravel().tolist()
    L.append(f"E_{name} = {a!r}")
    T.append(f"    assert close(fvals({call}) if True else [], E_{name})")
def adds(name, r, call):
    a = np.asarray(r.to_numpy(), dtype=float).ravel().tolist()
    L.append(f"E_{name} = {a!r}")
    T.append(f"    assert close(vals({call}), E_{name})")
adds("cumcount", gb.cumcount(), "gb.cumcount()")
adds("cumcount_d", gb.cumcount(ascending=False), "gb.cumcount(False)")
adds("ngroup", gb.ngroup(), "gb.ngroup()")
add("nth", gb.nth(1), "gb.nth(1)")
add("shift", gb[["g", "h"]].shift(1), 'gb[["g", "h"]].shift(1)')
add("diff", gb[["g", "h"]].diff(), 'gb[["g", "h"]].diff()')
add("pct", gb[["h"]].pct_change(), 'gb[["h"]].pct_change()')
add("cumsum", gb[["g", "h"]].cumsum(), 'gb[["g", "h"]].cumsum()')
add("cummax", gb[["g", "h"]].cummax(), 'gb[["g", "h"]].cummax()')
add("cummin", gb[["g", "h"]].cummin(), 'gb[["g", "h"]].cummin()')
for m in ("average", "min", "max", "first", "dense"):
    add("rank"+m, gb[["g", "h"]].rank(method=m), f'gb[["g", "h"]].rank("{m}")')
add("rollmean", gb[["g", "h"]].rolling(3).mean().droplevel(0).sort_index(), 'gb[["g", "h"]].rolling(3).mean()')
add("rollstd", gb[["g", "h"]].rolling(4, min_periods=2).std().droplevel(0).sort_index(), 'gb[["g", "h"]].rolling(4, 2).std()')
add("skew", gb[["g", "h"]].skew(), 'gb[["g", "h"]].skew()')
add("quant", gb[["g", "h"]].quantile(0.4), 'gb[["g", "h"]].quantile(0.4)')
add("sem", gb[["g", "h"]].sem(), 'gb[["g", "h"]].sem()')
setup = ('G = List[float]([float((i * 19 + 5) % 17 - 6) * 0.53 for i in range(30)])\nH = List[float]([float((i * 7 + 3) % 13 - 4) * 0.71 for i in range(30)])\n'
 'K = List[int]([i % 4 for i in range(30)])\nG[5] = nan\ndf = pd.DataFrame({"k": pd.Series(K), "g": pd.Series(G), "h": pd.Series(H)})\ngb = df.groupby("k")\n')
open(os.path.join(D, "test_groupby2.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_groupby2():\n" + "\n".join(T) + "\ntest_groupby2()\n")
