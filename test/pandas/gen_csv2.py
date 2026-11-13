# Generates test_csv2.py (read_csv sep/header/names/index_col/usecols/dtype/na_values/skiprows/nrows/parse_dates/comment, to_csv options) from pandas 3.1.1.
import numpy as np, pandas as pd, os, io
D = os.path.dirname(__file__)
head = open(os.path.join(D, "test_ops3.py")).read().split("G = List[float]")[0]
head += '''def ds(i) -> List[str]:
    return [str(x) for x in i.tolist()]

def cols(d: pd.DataFrame) -> List[str]:
    return [str(c) for c in d.columns.tolist()]

def dts(d: pd.DataFrame) -> List[str]:
    return [str(t) for t in d.dtypes.tolist()]

'''
T1 = "# comment line\nid;val;name;when;flag\n1;2.5;alpha;2024-01-02;True\n2;NA;beta;2024-02-03;False\n3;7,5;?;2024-03-04;True\n4;9.0;delta;2024-04-05;False\n"
T2 = "x,y,z\n1,2.5,a\n2,,b\n3,4.5,\n4,5.5,d\n5,6.5,e\n"
L = []; T = []
def f(a): return np.asarray(a, dtype=float).ravel().tolist()
def addd(name, r, call, cols_too=True):
    L.append(f"E_{name} = {f(r.to_numpy(dtype=float))!r}")
    T.append(f"    assert close(fvals({call}), E_{name})")
def addc(name, r, call):
    L.append(f"E_{name}_c = {[str(c) for c in r.columns]!r}\nE_{name}_t = {[str(t) for t in r.dtypes]!r}")
    T.append(f"    assert cols({call}) == E_{name}_c")
    T.append(f"    assert dts({call}) == E_{name}_t")
def txt(name, s, call):
    L.append(f"E_{name} = {s!r}")
    T.append(f"    assert {call} == E_{name}")
r = pd.read_csv(io.StringIO(T2))
addd("r_plain", r[["x", "y"]], 'rd(T2)[["x", "y"]]'); addc("r_plain", r, "rd(T2)")
r = pd.read_csv(io.StringIO(T2), usecols=["x", "z"])
addc("r_use", r, 'pd.read_csv_string(T2, ",", 0, None, None, ["x", "z"])')
r = pd.read_csv(io.StringIO(T2), nrows=3)
addd("r_nrows", r[["x", "y"]], 'pd.read_csv_string(T2, ",", 0, None, None, None, 3)[["x", "y"]]')
r = pd.read_csv(io.StringIO(T2), skiprows=2, header=None, names=["p", "q", "r"])
addc("r_skip", r, 'pd.read_csv_string(T2, ",", None, ["p", "q", "r"], None, None, None, 2)')
addd("r_skip", r[["p", "q"]], 'pd.read_csv_string(T2, ",", None, ["p", "q", "r"], None, None, None, 2)[["p", "q"]]')
r = pd.read_csv(io.StringIO(T2), index_col="x")
txt("r_idx", str(r), 'str(pd.read_csv_string(T2, ",", 0, None, "x"))')
r = pd.read_csv(io.StringIO(T2), index_col=0)
txt("r_idx0", str(r), 'str(pd.read_csv_string(T2, ",", 0, None, 0))')
r = pd.read_csv(io.StringIO(T2), dtype={"x": "float64", "z": "str"})
addc("r_dtype", r, 'pd.read_csv_string(T2, ",", 0, None, None, None, None, 0, {"x": "float64", "z": "str"})')
r = pd.read_csv(io.StringIO(T2), na_values=["a", "6.5"])
addd("r_na", r[["y"]], 'pd.read_csv_string(T2, ",", 0, None, None, None, None, 0, None, ["a", "6.5"])[["y"]]'); addc("r_na", r, 'pd.read_csv_string(T2, ",", 0, None, None, None, None, 0, None, ["a", "6.5"])')
txt("r_na_z", str(r), 'str(pd.read_csv_string(T2, ",", 0, None, None, None, None, 0, None, ["a", "6.5"]))')
r = pd.read_csv(io.StringIO(T1), sep=";", comment="#", decimal=",") if False else pd.read_csv(io.StringIO(T1), sep=";", comment="#", na_values=["?"], parse_dates=["when"])
addc("r_big", r, 'pd.read_csv_string(T1, ";", 0, None, None, None, None, 0, None, ["?"], ["when"], True, True, "#")')
txt("r_big_s", str(r), 'str(pd.read_csv_string(T1, ";", 0, None, None, None, None, 0, None, ["?"], ["when"], True, True, "#"))')
r = pd.read_csv(io.StringIO(T2), keep_default_na=False)
txt("r_nodef", str(r), 'str(pd.read_csv_string(T2, ",", 0, None, None, None, None, 0, None, None, None, False))')
d = pd.read_csv(io.StringIO(T2))
txt("w_def", d.to_csv(), "rd(T2).to_csv()")
txt("w_noidx", d.to_csv(index=False), "rd(T2).to_csv(None, \",\", \"\", None, None, True, False)")
txt("w_sep", d.to_csv(sep=";", na_rep="NULL", float_format="%.3f", index=False), 'rd(T2).to_csv(None, ";", "NULL", "%.3f", None, True, False)')
txt("w_nohdr", d.to_csv(header=False), "rd(T2).to_csv(None, \",\", \"\", None, None, False)")
txt("w_cols", d.to_csv(columns=["z", "x"], index_label="ID"), 'rd(T2).to_csv(None, ",", "", None, ["z", "x"], True, True, "ID")')
txt("w_crlf", d.to_csv(lineterminator="\r\n"), 'rd(T2).to_csv(None, ",", "", None, None, True, True, None, "\\r\\n")')
q = pd.DataFrame({"s": ['a,b', 'he said "hi"', 'plain', 'x\ny'], "n": [1.5, 2.0, float("nan"), 4.0]})
txt("w_quote", q.to_csv(index=False), "qd.to_csv(None, \",\", \"\", None, None, True, False)")
dd = pd.DataFrame({"t": pd.to_datetime(["2024-01-02", "2024-02-03 05:06:07", "2024-03-04"], format="ISO8601"), "v": [1, 2, 3]})
txt("w_dt", dd.to_csv(index=False), "dtd.to_csv(None, \",\", \"\", None, None, True, False)")
setup = ('T1 = ' + repr(T1) + '\nT2 = ' + repr(T2) + '\n'
 'def rd(t: str) -> pd.DataFrame:\n    return pd.read_csv_string(t)\n\n'
 'qd = pd.DataFrame({"s": pd.Series(["a,b", "he said \\"hi\\"", "plain", "x\\ny"]), "n": pd.Series([1.5, 2.0, float("nan"), 4.0])})\n'
 'dtd = pd.DataFrame({"t": pd.to_datetime(["2024-01-02", "2024-02-03 05:06:07", "2024-03-04"]), "v": pd.Series([1, 2, 3])})\n')
open(os.path.join(D, "test_csv2.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_csv2():\n" + "\n".join(T) + "\ntest_csv2()\n")
