# Copyright (C) 2026 the Typhon authors
# Generates data/finetune_data.txt (torch 2.15.0, mne 1.13.2): run in the reference venv
# from this directory's parent's parent (typhon/): python test/robot/gen_finetune_data.py
import os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(here, "../../examples/Robot-BCI-Movement-GUI")
sys.path.insert(0, REPO)
from src.models.models_finetuning import prepare_patient_loaders_from_edf
M = (1 << 61) - 1

def bithash(a):
    a = np.ascontiguousarray(a)
    u = a.view(np.uint32 if a.itemsize == 4 else np.uint64).reshape(-1).astype(object)
    return int(sum(int(v) * (i + 1) for i, v in enumerate(u)) % M)

out = []
for s in ["S025_test", "S050_test"]:
    full, y, subs = prepare_patient_loaders_from_edf(os.path.join(REPO, "data/users", s, "combined_record.edf"))
    X = full.dataset.tensors[0].numpy()
    out.append(f"{s} n={len(y)} shape={','.join(map(str, X.shape))} X={bithash(X)} y={bithash(y.numpy())} batches={len(full)}")
    for k in subs:
        t = subs[k].dataset.tensors
        out.append(f"{s} {k} n={len(t[1])} X={bithash(t[0].numpy())} y={bithash(t[1].numpy())} batches={len(subs[k])}")
open(os.path.join(here, "data/finetune_data.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
