# Copyright (C) 2026 the Typhon authors
# Shortened patient fine-tuning with the user's own helpers from
# models_finetuning(_typhon).py: seed_everything, split_dataloader (train_test_split
# stratify), prepare_for_patient_finetuning, Adam over filter(requires_grad),
# ReduceLROnPlateau(max), class-weighted CrossEntropyLoss, copy.deepcopy(state_dict),
# augment_eeg, mixup_data/mixup_criterion, clip_grad_norm_, f1/precision/recall.
# 3 epochs on the Both Hands submodel + 4 epochs of a MotorControlNet head on S025_test.
# Expected: data/finetune_short.txt from gen_finetune_short.py (torch 2.15.0); metrics
# exact, losses within 2e-4 (float32 summation order in BLAS kernels).
# Build with TYPHON_PATH=examples/Robot-BCI-Movement-GUI until main-dir imports land.
import os
DIR = __file__[:__file__.rfind("/")] if "/" in __file__ else "."
REPO = DIR + "/../../examples/Robot-BCI-Movement-GUI"
OUT = List[str]()
def emit(s: str):
    OUT.append(s)

def npi(t):
    return t.cpu().numpy(np.int64, 1)

if not os.path.exists(REPO + "/data/users/S025_test/combined_record.edf"):
    print("SKIP: examples/Robot-BCI-Movement-GUI absent")
else:
    import numpy as np
    from src.models.models_finetuning_typhon import prepare_patient_loaders_from_edf, split_dataloader, prepare_for_patient_finetuning, seed_everything, augment_eeg, mixup_data, mixup_criterion
# --- BODY
    import copy
    import torch
    import torch.nn as nn
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from src.models.architecture import MotorControlNet, CustomTemporalNet
    full, y, subs = prepare_patient_loaders_from_edf(REPO + "/data/users/S025_test/combined_record.edf", 64, ['F3', 'C3', 'C4', 'Cz', 'T7', 'T8', 'P7', 'O1'])
    seed_everything(52)
    md = REPO + "/src/models/"
    paths = {'fwd': md + 'best_general_both_hands_f1.pth', 'back': md + 'best_general_both_feet_f1.pth',
             'left': md + 'best_general_left_hand_f1.pth', 'right': md + 'best_general_right_hand_f1.pth'}
    model = MotorControlNet(model_class=CustomTemporalNet, weights_paths=paths, freeze_submodels=False).to(torch.device("cpu"))
    tl, vl, yt = split_dataloader(subs['both_hands'], batch_size=64)
    sm = prepare_for_patient_finetuning(model.fwd_model)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, sm.parameters()), lr=0.001, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    w = compute_class_weight(class_weight='balanced', classes=np.arange(2), y=npi(yt))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    best = copy.deepcopy(sm.state_dict())
    best_f1 = 0.0
    for ep in range(1, 4):
        sm.train()
        for bx, by in tl:
            opt.zero_grad()
            loss = crit(sm(bx), by)
            loss.backward()
            opt.step()
            emit(f"sub loss {loss.item():.6f}")
        sm.eval()
        P = []
        T = []
        with torch.no_grad():
            for bx, by in vl:
                P.extend(npi(torch.argmax(sm(bx), dim=1)))
                T.extend(npi(by))
        f1 = f1_score(T, P, average='macro', zero_division=0)
        sch.step(f1)
        if f1 > best_f1:
            best_f1 = f1
            best = copy.deepcopy(sm.state_dict())
        emit(f"sub ep {ep} f1 {f1:.4f} preds {sum([int(v) for v in P])}")
    sm.load_state_dict(best)
    model._freeze_submodels()
    ftl, fvl, fy = split_dataloader(full, batch_size=64)
    opt2 = torch.optim.Adam(model.motor_classifier.parameters(), lr=0.001, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='max', factor=0.5, patience=3)
    yn = npi(fy)
    w2 = compute_class_weight(class_weight='balanced', classes=np.unique(yn), y=yn)
    crit2 = nn.CrossEntropyLoss(weight=torch.tensor(w2, dtype=torch.float32))
    for ep in range(1, 5):
        model.train()
        for bx, by in ftl:
            p = torch.rand(1).item()
            mix = False
            if p < 0.4:
                bx = augment_eeg(bx, noise_std=0.05, channel_drop_p=0.15, time_shift_max=8)
            elif p < 0.7:
                mx, ya, yb, lam = mixup_data(bx, by, alpha=0.2)
                mix = True
            opt2.zero_grad()
            if mix:
                lo = model(mx, mx, mx, mx)
                loss = mixup_criterion(crit2, lo, ya, yb, lam)
            else:
                lo = model(bx, bx, bx, bx)
                loss = crit2(lo, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.motor_classifier.parameters(), max_norm=1.0)
            opt2.step()
            emit(f"fin p {p:.6f} mix {int(mix)} loss {loss.item():.6f}")
        model.eval()
        P = []
        T = []
        with torch.no_grad():
            for bx, by in fvl:
                P.extend(npi(torch.argmax(model(bx, bx, bx, bx), dim=1)))
                T.extend(npi(by))
        f1 = f1_score(T, P, average='macro', zero_division=0)
        pr = precision_score(T, P, average='macro', zero_division=0)
        rc = recall_score(T, P, average='macro', zero_division=0)
        sch2.step(f1)
        emit(f"fin ep {ep} f1 {f1:.4f} P {pr:.2f} R {rc:.2f} lr {sch2.get_last_lr()[0]}")
# --- END BODY
    exp = [l.rstrip("\n") for l in open(DIR + "/data/finetune_short.txt")]
    bad = 0
    maxd = 0.0
    for i in range(max(len(exp), len(OUT))):
        a = exp[i] if i < len(exp) else "<none>"
        b = OUT[i] if i < len(OUT) else "<none>"
        if a == b:
            continue
        ka = a[:a.rfind(" ")]
        kb = b[:b.rfind(" ")]
        if " loss " in a and ka == kb:
            d = abs(float(a[a.rfind(" ") + 1:]) - float(b[b.rfind(" ") + 1:]))
            maxd = max(maxd, d)
            if d <= 2e-4:
                continue
        print("MISMATCH", i, "\n  exp", a, "\n  got", b)
        bad += 1
    assert bad == 0
    print("finetune short ok", len(OUT), "max loss diff", maxd)
