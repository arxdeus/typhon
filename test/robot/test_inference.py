# Copyright (C) 2026 the Typhon authors
# The user's real-time inference, verbatim: src/models/architecture.py
# (MotorControlNet of 4 CustomTemporalNet) and src/core/inference.py
# (preprocess_chunk: mne.filter.filter_data 8-30 Hz firwin, crop, per-channel
# z-score; run_inference: torch.load(.pth, weights_only=True), softmax, EMA,
# threshold/margin -> S/L/R/F/B through queue.Queue) on the streamer.py windows
# (5 s = 800 samples, step 80) of S025_test and S050_test.
# Expected: data/inference_expected.txt from gen_inference_expected.py (torch 2.15.0,
# mne 1.13.2): command sequences exact, argmax of all 1466 S025 windows exact,
# probe logits within 2e-5 (float32 BLAS summation order), preprocessed values 1e-6.
import os
DIR = __file__[:__file__.rfind("/")] if "/" in __file__ else "."
REPO = DIR + "/../../examples/Robot-BCI-Movement-GUI"

if not os.path.exists(REPO + "/data/users/S025_test/combined_record.edf"):
    print("SKIP: examples/Robot-BCI-Movement-GUI absent")
else:
    import queue
    import numpy as np
    import torch
    import mne
    from src.models.architecture import MotorControlNet, CustomTemporalNet
    from src.core.inference import preprocess_chunk, run_inference

    exp = Dict[str, str]()
    for line in open(DIR + "/data/inference_expected.txt"):
        line = line.rstrip()
        sp = line.find(" ")
        key = line[:sp]
        if key == "pre" or key == "logits":
            sp2 = line.find(" ", sp + 1)
            key = key + " " + line[sp + 1:sp2]
            sp = sp2
        elif key.endswith("_test"):
            continue
        if key in exp:
            key = key + " 2"
        exp[key] = line[sp + 1:]

    def floats(s: str) -> List[float]:
        return [float(v) for v in s.split(" ")]

    ok = True
    for si, s in enumerate(["S025_test", "S050_test"]):
        model_path = REPO + "/data/users/" + s + "/best_finetuned_patient_01_ensemble.pth"
        raw = mne.io.read_raw_edf(REPO + "/data/users/" + s + "/combined_record.edf", preload=True, verbose=False)
        data = raw.get_data()
        chunks = [data[:, a:a + 800] for a in range(0, data.shape[1] - 800, 80)]
        if s == "S025_test":
            m = MotorControlNet(model_class=CustomTemporalNet)
            m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            m.eval()
            am = List[str]()
            dpre = 0.0
            dlog = 0.0
            for i in range(len(chunks)):
                x = preprocess_chunk(chunks[i])
                with torch.no_grad():
                    o = m(x, x, x, x)[0]
                am.append(str(int(o.argmax().item())))
                k = "logits " + str(i)
                if k in exp:
                    e = floats(exp[k])
                    for j in range(5):
                        dlog = max(dlog, abs(o[j].item() - e[j]))
                    ep = floats(exp["pre " + str(i)])
                    xs = x[0, :, ::97].reshape(-1)
                    for j in range(len(ep)):
                        dpre = max(dpre, abs(xs[j].item() - ep[j]))
            got = "".join(am)
            flips = sum(1 for i in range(len(got)) if got[i] != exp["argmax"][i])
            print(s, "argmax flips", flips, "max|dlogit|", dlog, "max|dpre|", dpre)
            if flips != 0 or dlog > 2e-5 or dpre > 1e-6:
                ok = False
        # element types declared here (untyped Queue + None sentinel is not inferred);
        # run_inference itself is the user's verbatim code
        dq = queue.Queue[Optional[np.ndarray[float, 2]]]()
        cq = queue.Queue[str]()
        for c in chunks:
            dq.put(c)
        dq.put(None)
        run_inference(model_path, dq, cq)
        cmds = List[str]()
        while not cq.empty():
            cmds.append(cq.get())
        got = "".join(cmds)
        want = exp["commands" + (" 2" if si == 1 else "")]
        diff = sum(1 for i in range(min(len(got), len(want))) if got[i] != want[i])
        print(s, "commands", len(got), "mismatches", diff)
        if got != want:
            ok = False
    print("inference ok" if ok else "inference FAILED")
    if not ok:
        raise ValueError("inference mismatch")
