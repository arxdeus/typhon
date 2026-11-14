# Generates data/inference_expected.txt for test_inference.py with the reference
# stack (torch 2.15.0, mne 1.13.2): run from the typhon root with
#   <venv>/bin/python test/robot/gen_inference_expected.py
# Uses the user's verbatim src/core/inference.py (preprocess_chunk, run_inference)
# on the streamer.py windows (5 s = 800 samples, step 80) of S025_test and S050_test.
import os
import queue
import sys

import numpy as np
import torch

DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(DIR, "../../examples/Robot-BCI-Movement-GUI")
sys.path.insert(0, REPO)
torch.set_num_threads(1)

import mne
from src.models.architecture import MotorControlNet, CustomTemporalNet
from src.core.inference import preprocess_chunk, run_inference

PROBE = [0, 1, 500, 1000, 1465]
out = []
for s in ["S025_test", "S050_test"]:
    model_path = f"{REPO}/data/users/{s}/best_finetuned_patient_01_ensemble.pth"
    raw = mne.io.read_raw_edf(f"{REPO}/data/users/{s}/combined_record.edf", preload=True, verbose=False)
    data = raw.get_data()
    chunks = [data[:, a:a + 800] for a in range(0, data.shape[1] - 800, 80)]
    out.append(f"{s} windows {len(chunks)}")
    if s == "S025_test":
        m = MotorControlNet(model_class=CustomTemporalNet)
        m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        m.eval()
        am = []
        for i, c in enumerate(chunks):
            x = preprocess_chunk(c)
            with torch.no_grad():
                o = m(x, x, x, x)[0]
            am.append(str(int(o.argmax())))
            if i in PROBE:
                out.append(f"pre {i} " + " ".join(repr(float(v)) for v in x[0, :, ::97].reshape(-1)))
                out.append(f"logits {i} " + " ".join(repr(float(v)) for v in o))
        out.append("argmax " + "".join(am))
    dq = queue.Queue()
    cq = queue.Queue()
    for c in chunks:
        dq.put(c)
    dq.put(None)
    run_inference(model_path, dq, cq)
    cmds = []
    while not cq.empty():
        cmds.append(cq.get())
    out.append("commands " + "".join(cmds))

with open(os.path.join(DIR, "data/inference_expected.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
print("wrote", len(out), "lines")
