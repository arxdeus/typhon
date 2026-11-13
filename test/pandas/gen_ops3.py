# Generates test_ops3.py (agg lists, describe, nlargest, clip/round/where/mask, combine_first, sample, rank pct) from pandas 3.1.1.
import numpy as np, pandas as pd, os
D = os.path.dirname(__file__)
head = open(os.path.join(D, "test_str.py")).read().split("S = List[Optional[str]]")[0].replace("nan = float(\"nan\")\n", "")
G = [((i * 19 + 5) % 17 - 6) * 0.53 + (i % 4) for i in range(30)]
H = [((i * 7 + 3) % 13 - 4) * 0.71 for i in range(30)]
K = [i % 4 for i in range(30)]
G[4] = np.nan
d = pd.DataFrame({"k": K, "g": G, "h": H, "t": [chr(97 + i % 5) for i in range(30)]})
s = d["g"]
L = []; T = []
def f(a): return np.asarray(a, dtype=float).ravel().tolist()
def add(name, r, call, kind="s"):
    L.append(f"E_{name} = {f(r.to_numpy() if hasattr(r, 'to_numpy') else r)!r}")
    wrap = {"s": f"vals({call})", "f": f"fvals({call})", "x": f"[{call}]"}[kind]
    T.append(f"    assert close({wrap}, E_{name})")
def addl(name, labels, call):
    L.append(f"E_{name} = {[str(x) for x in labels]!r}")
    T.append(f"    assert [str(x) for x in ({call}).tolist()] == E_{name}")
add("agg_s", s.agg(["sum", "mean", "max", "min", "std", "count", "median"]), 's.agg(["sum", "mean", "max", "min", "std", "count", "median"])')
addl("agg_s_i", s.agg(["sum", "mean", "max"]).index, 's.agg(["sum", "mean", "max"]).index')
add("agg_one", s.agg("var"), 's.agg("var")', "x")
num = d[["g", "h"]]
add("agg_d", num.agg(["sum", "mean", "min"]), 'dn.agg(["sum", "mean", "min"])', "f")
add("agg_d1", num.agg("max"), 'dn.agg("max")')
add("desc_p", s.describe(percentiles=[0.1, 0.9]), "s.describe([0.1, 0.9])")
addl("desc_p_i", s.describe(percentiles=[0.1, 0.9]).index, "s.describe([0.1, 0.9]).index")
add("desc_d", d.describe(percentiles=[0.05, 0.5]), "d.describe([0.05, 0.5])", "f")
addl("desc_dc", d.describe().columns, "d.describe().columns")
dall = d.describe(include="all")
L.append(f"E_desc_all = {[str(x) for x in dall.index]!r}\nE_desc_allc = {[str(x) for x in dall.columns]!r}")
T.append('    da = d.describe(None, "all")\n    assert [str(x) for x in da.index.tolist()] == E_desc_all\n    assert [str(x) for x in da.columns.tolist()] == E_desc_allc')
add("desc_all_h", dall["h"], 'd.describe(None, "all")["h"]')
L.append(f"E_desc_all_t = {[str(x) for x in dall['t']]!r}")
T.append('    assert [str(x) for x in d.describe(None, "all")["t"].tolist()] == E_desc_all_t')
add("nlarg", d.nlargest(4, "g"), 'd.nlargest(4, "g")[["g", "h"]]', "f") if False else None
nl = d.nlargest(4, "h")
add("nlarg_h", nl["h"], 'd.nlargest(4, "h")["h"]')
addl("nlarg_i", nl.index, 'd.nlargest(4, "h").index')
ns = d.nsmallest(5, ["k", "h"])
addl("nsmall_i", ns.index, 'd.nsmallest(5, ["k", "h"]).index')
addl("nlarg_s", s.nlargest(3).index, "s.nlargest(3).index")
addl("nsmall_s", s.nsmallest(3).index, "s.nsmallest(3).index")
add("rank_pct", s.rank(pct=True), "s.rank(pct=True)")
add("rank_pct_min", s.rank(method="min", pct=True, ascending=False), 's.rank("min", True, "keep", True)') if False else None
add("rank_pct2", s.rank(method="dense", pct=True), 's.rank("dense", True, "keep", True)')
add("clip", s.clip(0.5, 3.0), "s.clip(0.5, 3.0)")
add("clip_lo", s.clip(lower=1.0), "s.clip(lower=1.0)")
add("round2", s.round(1), "s.round(1)")
add("where", s.where(s > 1.5, -1.0), "s.where(s > 1.5, -1.0)")
add("where_nan", s.where(s > 1.5), "s.where(s > 1.5)")
add("mask", s.mask(s > 1.5, 0.0), "s.mask(s > 1.5, 0.0)")
add("isin", s.isin([0.53, 1.06, 3.18]), "s.isin([0.53, 1.06, 3.18])")
add("between", s.between(0.5, 2.5), "s.between(0.5, 2.5)")
add("between_l", s.between(0.53, 2.5, inclusive="left"), 's.between(0.53, 2.5, "left")')
o = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0], index=[2, 4, 6, 40, 41])
cf = s.combine_first(o)
add("cfirst", cf, "s.combine_first(o)")
addl("cfirst_i", cf.index, "s.combine_first(o).index")
u = s.copy(); u.update(pd.Series([7.0, np.nan, 9.0], index=[0, 1, 2]))
add("update", u, "update_case(s)")
# sample uses numpy RandomState choice
add("samp_s", s.sample(5, random_state=3), "s.sample(5, None, False, 3)")
addl("samp_s_i", s.sample(5, random_state=3).index, "s.sample(5, None, False, 3).index")
addl("samp_f", d.sample(frac=0.2, random_state=11).index, "d.sample(None, 0.2, False, 11).index")
addl("samp_r", d.sample(n=12, replace=True, random_state=7).index, "d.sample(12, None, True, 7).index")
add("samp_d", d.sample(n=4, random_state=1)["h"], "d.sample(4, None, False, 1)[\"h\"]")
setup = ('G = List[float]([float((i * 19 + 5) % 17 - 6) * 0.53 + float(i % 4) for i in range(30)])\nH = List[float]([float((i * 7 + 3) % 13 - 4) * 0.71 for i in range(30)])\n'
 'K = List[int]([i % 4 for i in range(30)])\nG[4] = nan\nTT = List[str]([chr(97 + i % 5) for i in range(30)])\n'
 'd = pd.DataFrame({"k": pd.Series(K), "g": pd.Series(G), "h": pd.Series(H), "t": pd.Series(TT)})\ns = d["g"]\ndn = d[["g", "h"]]\n'
 'o = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0], index=[2, 4, 6, 40, 41])\n'
 'def update_case(x: pd.Series) -> pd.Series:\n    u = x.copy()\n    u.update(pd.Series([7.0, nan, 9.0], index=[0, 1, 2]))\n    return u\n\n')
open(os.path.join(D, "test_ops3.py"), "w").write(head + 'nan = float("nan")\n' + setup + "\n".join(L) + "\n\n@test\ndef test_ops3():\n" + "\n".join(T) + "\ntest_ops3()\n")
