# Copyright (C) 2026 the Typhon authors
# torch.load / torch.save in the native torch zip+pickle format.
# Expected values from test/robot/gen_pth_expected.py (torch 2.15.0).
import os
import torch

DIR = __file__[:__file__.rfind("/")] if "/" in __file__ else "."
REPO = DIR + "/../../examples/Robot-BCI-Movement-GUI"
M = (1 << 61) - 1

def bithash(t: torch.Tensor) -> int:
    from torch._tensor import _itemsize
    c = t.contiguous()
    sz = _itemsize(c._dt)
    n = c.numel()
    h = UInt[128](0)
    m = UInt[128](M)
    if sz == 4:
        p = Ptr[u32](c._st.ptr + c._off * 4)
        for i in range(n):
            h = (h + UInt[128](p[i]) * UInt[128](i + 1)) % m
    else:
        p = Ptr[u64](c._st.ptr + c._off * 8)
        for i in range(n):
            h = (h + UInt[128](p[i]) * UInt[128](i + 1)) % m
    return int(h)

def dtname(t: torch.Tensor) -> str:
    if t.dtype == torch.float32: return "f32"
    if t.dtype == torch.float64: return "f64"
    if t.dtype == torch.int64: return "i64"
    return "?"

def describe(sd: Dict[str, torch.Tensor]) -> List[str]:
    out = [str(len(sd))]
    for k, t in sd.items():
        shp = ",".join([str(s) for s in t.shape])
        out.append(k + " " + dtname(t) + " " + (shp if shp != "" else "-") + " " + str(bithash(t)))
    return out

def expected() -> Dict[str, List[str]]:
    d = Dict[str, List[str]]()
    cur = ""
    for ln in open(DIR + "/data/pth_expected.txt"):
        ln = ln.rstrip("\n")
        if ln.startswith("FILE "):
            cur = ln[5:]
            d[cur] = List[str]()
        elif ln != "":
            d[cur].append(ln)
    return d

TINY = ["6", "w f32 3,5 257945861091", "wt f32 6,4 609495450361", "row f32 6 37205891316",
        "d f64 2,3 2023988042853206730", "n i64 - 7", "big i64 4 3264175144997"]

@test
def test_python_saved_fixture():
    sd = torch.load(DIR + "/data/tiny_views.pth", map_location=torch.device("cpu"), weights_only=True)
    assert describe(sd) == TINY
    # views keep sharing storage like torch
    assert sd["wt"].shape == [6, 4] and sd["row"].shape == [6]
    assert float(sd["row"][1]) == float(sd["wt"][1, 2])

@test
def test_roundtrip_typhon():
    sd = torch.load(DIR + "/data/tiny_views.pth")
    path = "/tmp/_typhon_pth_roundtrip.pth"
    torch.save(sd, path)
    r = torch.load(path, weights_only=True)
    assert describe(r) == TINY
    assert list(r.keys()) == list(sd.keys())

@test
def test_user_checkpoints():
    if not os.path.exists(REPO + "/src/models/best_general_both_feet_f1.pth"):
        print("SKIP test_user_checkpoints: Robot-BCI-Movement-GUI not present")
        return
    exp = expected()
    for f, lines in exp.items():
        sd = torch.load(REPO + "/" + f, map_location=torch.device("cpu"), weights_only=True)
        got = describe(sd)
        if got != lines:
            for i in range(min(len(got), len(lines))):
                if got[i] != lines[i]:
                    print(f, "mismatch:", got[i], "!=", lines[i])
                    break
        assert got == lines

test_python_saved_fixture()
test_roundtrip_typhon()
test_user_checkpoints()
