# Run in the reference venv (torch 2.15.0):
#   python test/robot/gen_pth_expected.py
# Writes test/robot/data/pth_expected.txt (per-tensor checksums of the user's
# 6 real checkpoints) and test/robot/data/tiny_views.pth (a small torch.save
# fixture with a non-contiguous view, a storage offset and a shared storage).
import os, glob, collections, torch, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "examples", "Robot-BCI-Movement-GUI")
FILES = ["data/users/S025_test/best_finetuned_patient_01_ensemble.pth",
         "data/users/S050_test/best_finetuned_patient_01_ensemble.pth",
         "src/models/best_general_both_feet_f1.pth",
         "src/models/best_general_both_hands_f1.pth",
         "src/models/best_general_left_hand_f1.pth",
         "src/models/best_general_right_hand_f1.pth"]
M = (1 << 61) - 1

def bithash(t):
    # sum over C-order elements of raw-bits * (i+1) mod 2^61-1 (order exact)
    a = t.contiguous().numpy()
    u = a.view(np.uint32 if a.itemsize == 4 else np.uint64).reshape(-1).astype(object)
    h = 0
    for i, x in enumerate(u.tolist()):
        h = (h + int(x) * (i + 1)) % M
    return h

def lines(sd):
    out = [str(len(sd))]
    for k, t in sd.items():
        dt = {torch.float32: "f32", torch.int64: "i64", torch.float64: "f64"}[t.dtype]
        out.append(f"{k} {dt} {','.join(map(str, t.shape)) or '-'} {bithash(t)}")
    return out

out = []
for f in FILES:
    sd = torch.load(os.path.join(REPO, f), map_location="cpu", weights_only=True)
    out.append("FILE " + f)
    out += lines(sd)
open(os.path.join(HERE, "data", "pth_expected.txt"), "w").write("\n".join(out) + "\n")

torch.manual_seed(0)
base = torch.randn(4, 6)
sd = collections.OrderedDict()
sd["w"] = torch.randn(3, 5)
sd["wt"] = base.t()                  # strided view
sd["row"] = base[2]                  # offset view sharing base's storage
sd["d"] = torch.randn(2, 3, dtype=torch.float64)
sd["n"] = torch.tensor(7)            # 0-d int64
sd["big"] = torch.tensor([3, -7, 2 ** 40, -(2 ** 33)])
torch.save(sd, os.path.join(HERE, "data", "tiny_views.pth"))
print("\n".join(["FILE tiny_views.pth"] + lines(sd)))
