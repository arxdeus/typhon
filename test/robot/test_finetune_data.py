# Copyright (C) 2026 the Typhon authors
# Data pipeline parity of models_finetuning.prepare_patient_loaders_from_edf (user code,
# imported unchanged) on S025_test/S050_test: read_raw_edf, rename/pick, Raw.filter 8-30 Hz
# per EDGE-boundary segment, Epochs(-1, 4, baseline), crop(0.5, 3.5), z-score, masks.
# Bit-exact hashes from gen_finetune_data.py (torch 2.15.0, mne 1.13.2).
import os
import torch

DIR = __file__[:__file__.rfind("/")] if "/" in __file__ else "."
REPO = DIR + "/../../examples/Robot-BCI-Movement-GUI"
M = (1 << 61) - 1

def bithash(t: torch.Tensor) -> int:
    from torch._tensor import _itemsize
    c = t.contiguous()
    n = c.numel()
    h = UInt[128](0)
    m = UInt[128](M)
    if _itemsize(c._dt) == 4:
        p = Ptr[u32](c._st.ptr + c._off * 4)
        for i in range(n):
            h = (h + UInt[128](p[i]) * UInt[128](i + 1)) % m
    else:
        p = Ptr[u64](c._st.ptr + c._off * 8)
        for i in range(n):
            h = (h + UInt[128](p[i]) * UInt[128](i + 1)) % m
    return int(h)

if not os.path.exists(REPO + "/data/users/S025_test/combined_record.edf"):
    print("SKIP: examples/Robot-BCI-Movement-GUI absent")
else:
    from src.models.models_finetuning import prepare_patient_loaders_from_edf
    out = List[str]()
    for s in ["S025_test", "S050_test"]:
        full, y, subs = prepare_patient_loaders_from_edf(REPO + "/data/users/" + s + "/combined_record.edf")
        X = full.dataset.tensors[0]
        out.append(f"{s} n={len(y)} shape={','.join([str(v) for v in X.shape])} X={bithash(X)} y={bithash(y)} batches={len(full)}")
        for k in subs:
            t = subs[k].dataset.tensors
            out.append(f"{s} {k} n={len(t[1])} X={bithash(t[0])} y={bithash(t[1])} batches={len(subs[k])}")
    exp = [l.rstrip("\n") for l in open(DIR + "/data/finetune_data.txt")]
    bad = 0
    for i in range(len(exp)):
        if i >= len(out) or exp[i] != out[i]:
            print("MISMATCH\n  exp", exp[i], "\n  got", out[i] if i < len(out) else "<none>")
            bad += 1
    assert bad == 0 and len(out) == len(exp)
    print("finetune data ok", len(out))
