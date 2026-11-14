# Copyright (C) 2026 the Typhon authors
# RNG stream parity of the exact random calls models_finetuning.py makes:
# seed_everything(52), nn.Linear init, DataLoader(shuffle=True), torch.rand(1),
# randn_like, rand(n, c, 1), randint, np.random.beta, randperm, Dropout masks.
# Expected output: data/finetune_rng.txt from gen_finetune_rng.py (torch 2.15.0).
DIR = __file__[:__file__.rfind("/")] if "/" in __file__ else "."
OUT = List[str]()
def emit(s: str):
    OUT.append(s)

# --- BODY
import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def h(t):
    return " ".join([f"{v:.8e}" for v in t.reshape(-1).tolist()[:6]]) + f" | {float(t.double().sum().item()):.10e}"

random.seed(52)
os.environ['PYTHONHASHSEED'] = str(52)
np.random.seed(52)
torch.manual_seed(52)
torch.cuda.manual_seed(52)
torch.cuda.manual_seed_all(52)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
emit("random " + str(random.random()))
x = torch.zeros(4, 8, 20)
lin = nn.Linear(4, 32)
emit("linw " + h(lin.weight.detach()))
emit("linb " + h(lin.bias.detach()))
ds = TensorDataset(torch.arange(30).float(), torch.arange(30))
dl = DataLoader(ds, batch_size=8, shuffle=True)
for e in range(2):
    order = []
    for bx, by in dl:
        order.extend([int(v) for v in by.tolist()])
    emit("dl " + str(order))
emit("rand1 " + str(torch.rand(1).item()))
emit("randn_like " + h(torch.randn_like(x)))
emit("rand3 " + h(torch.rand(4, 8, 1, device=x.device)))
emit("randint " + str(int(torch.randint(-8, 9, (1,)).item())))
emit("beta " + str([float(np.random.beta(0.2, 0.2)) for _ in range(5)]))
emit("randperm " + str([int(v) for v in torch.randperm(13).tolist()]))
d = nn.Dropout(p=0.3)
d.train()
emit("dropout " + h(d(torch.ones(5, 32))))
d2 = nn.Dropout(p=0.5)
emit("dropout2 " + h(d2(torch.ones(3, 1920))))
emit("randn " + h(torch.randn(7)))
# --- END BODY

exp = [l.rstrip("\n") for l in open(DIR + "/data/finetune_rng.txt")]
bad = 0
for i in range(max(len(exp), len(OUT))):
    a = exp[i] if i < len(exp) else "<none>"
    b = OUT[i] if i < len(OUT) else "<none>"
    if a != b:
        print("MISMATCH", i, "\n  exp", a, "\n  got", b)
        bad += 1
assert bad == 0
print("finetune rng ok", len(OUT))
