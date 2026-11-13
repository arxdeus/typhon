# Generates test_str.py (Series.str accessor) from pandas 3.1.1.
import numpy as np, pandas as pd, os
D = os.path.dirname(__file__)
head = open(os.path.join(D, "test_stats2.py")).read().split("F = List[float]")[0]
head += '''def strs(s: pd.Series) -> List[str]:
    return [str(x) for x in s.tolist()]

def sdf(d: pd.DataFrame) -> List[str]:
    out = List[str]()
    for j in range(len(d.columns)):
        for x in d.iloc[:, j].tolist():
            out.append(str(x))
    return out

'''
S = ["Ab c", None, "xyz", " p q ", "123", "a.b.c", "Hello World", "z9"]
s = pd.Series(S)
setup = 'S = List[Optional[str]](["Ab c", None, "xyz", " p q ", "123", "a.b.c", "Hello World", "z9"])\ns = pd.Series(S)\n'
L = []; T = []
def lst(r):
    return [("nan" if (isinstance(x, float) and x != x) or x is None else (str(x) if not isinstance(x, (bool, np.bool_)) else str(bool(x)))) for x in r]
def add(name, r, call):
    if isinstance(r, pd.DataFrame):
        out = []
        for j in range(r.shape[1]):
            out += lst(r.iloc[:, j].tolist())
        L.append(f"E_{name} = {out!r}")
        T.append(f"    assert sdf({call}) == E_{name}")
    elif r.dtype == np.float64:
        L.append(f"E_{name} = {[float(x) for x in r]!r}")
        T.append(f"    assert close(vals({call}), E_{name})")
    elif r.dtype == np.int64:
        L.append(f"E_{name} = {[float(x) for x in r]!r}")
        T.append(f"    assert close(vals({call}), E_{name})")
    else:
        L.append(f"E_{name} = {lst(r.tolist())!r}")
        T.append(f"    assert strs({call}) == E_{name}")
for f in ["len", "lower", "upper", "title", "capitalize", "swapcase", "strip", "lstrip", "rstrip",
          "isalpha", "isdigit", "isalnum", "isspace", "islower", "isupper", "isnumeric"]:
    add(f, getattr(s.str, f)(), f"s.str.{f}()")
add("strip_c", s.str.strip("a z"), 's.str.strip("a z")')
add("contains", s.str.contains("b"), 's.str.contains("b", True, False)')
add("contains_re", s.str.contains("[xz]."), 's.str.contains("[xz].")')
add("contains_ci", s.str.contains("HELLO", case=False), 's.str.contains("HELLO", False)')
add("contains_na", s.str.contains("b", na=True), 's.str.contains("b", True, False, True)')
add("match", s.str.match("[A-Z]"), 's.str.match("[A-Z]")')
add("fullmatch", s.str.fullmatch("[0-9]+"), 's.str.fullmatch("[0-9]+")')
add("startswith", s.str.startswith("x"), 's.str.startswith("x")')
add("endswith", s.str.endswith("3"), 's.str.endswith("3")')
add("replace", s.str.replace("b", "Q"), 's.str.replace("b", "Q")')
add("replace_n", s.str.replace(".", "-", n=1, regex=False), 's.str.replace(".", "-", 1)')
add("replace_re", s.str.replace("[xy]", "_", regex=True), 's.str.replace("[xy]", "_", -1, True, True)')
add("replace_ci", s.str.replace("hello", "Bye", case=False, regex=True), 's.str.replace("hello", "Bye", -1, False, True)')
add("find", s.str.find("c"), 's.str.find("c")')
add("rfind", s.str.rfind("."), 's.str.rfind(".")')
add("count", s.str.count("[a-z]"), 's.str.count("[a-z]")')
add("slice", s.str.slice(1, 3), "s.str.slice(1, 3)")
add("slice2", s.str.slice(0, None, 2), "s.str.slice(0, None, 2)")
add("slice_neg", s.str.slice(-2), "s.str.slice(-2)")
add("get", s.str.get(0), "s.str.get(0)")
add("get_neg", s.str.get(-1), "s.str.get(-1)")
add("pad", s.str.pad(6, side="left", fillchar="*"), 's.str.pad(6, "left", "*")')
add("padr", s.str.pad(6, side="right", fillchar="*"), 's.str.pad(6, "right", "*")')
add("center", s.str.center(9, "-"), 's.str.center(9, "-")')
add("zfill", s.str.zfill(6), "s.str.zfill(6)")
add("repeat", s.str.repeat(2), "s.str.repeat(2)")
add("rprefix", s.str.removeprefix("x"), 's.str.removeprefix("x")')
add("rsuffix", s.str.removesuffix("c"), 's.str.removesuffix("c")')
add("cat_o", s.str.cat(list("abcdefgh"), sep="-"), 's.str.cat(list("abcdefgh"), "-")')
add("cat_o_na", s.str.cat(list("abcdefgh"), sep="-", na_rep="?"), 's.str.cat(list("abcdefgh"), "-", "?")')
add("split_x", s.str.split(" ", expand=True), 's.str.split(" ", -1, expand=True)')
add("split_dot", s.str.split(".", expand=True), 's.str.split(".", -1, expand=True)')
add("split_n", s.str.split(".", n=1, expand=True), 's.str.split(".", 1, expand=True)')
add("rsplit_n", s.str.rsplit(".", n=1, expand=True), 's.str.rsplit(".", 1, expand=True)')
add("split_ws", s.str.split(expand=True), "s.str.split(None, -1, expand=True)")
add("extract", s.str.extract(r"([a-z])(\d?)"), 's.str.extract(r"([a-z])(\\d?)")')
add("extract1", s.str.extract(r"(\d+)", expand=False), 's.str.extract(r"(\\d+)", False)')
L.append(f"E_cat = {s.str.cat(sep='|')!r}\nE_cat_na = {s.str.cat(sep='|', na_rep='?')!r}")
T.append('    assert s.str.cat(None, "|") == E_cat\n    assert s.str.cat(None, "|", "?") == E_cat_na')
open(os.path.join(D, "test_str.py"), "w").write(head + "nan = float(\"nan\")\n" + setup + "\n".join(L) + "\n\n@test\ndef test_str():\n" + "\n".join(T) + "\ntest_str()\n")
