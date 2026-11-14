# Copyright (C) 2026 the Typhon authors
# Generates data/finetune_short.txt (torch 2.15.0, sklearn 1.9.1, mne 1.13.2) by running
# the BODY of test_finetune_short.py with the ORIGINAL models_finetuning.py helpers:
#   <venv>/bin/python test/robot/gen_finetune_short.py
import os, sys, textwrap
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(here, "../../examples/Robot-BCI-Movement-GUI")
sys.path.insert(0, REPO)
import src.models.models_finetuning as mf
src = open(os.path.join(here, "test_finetune_short.py")).read()
body = textwrap.dedent(src[src.index("# --- BODY\n") + 11:src.index("# --- END BODY\n")])
lines = []
g = dict(vars(mf))
g.update(emit=lines.append, npi=lambda t: t.cpu().numpy(), REPO=REPO, np=np)
exec(body, g)
open(os.path.join(here, "data/finetune_short.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
