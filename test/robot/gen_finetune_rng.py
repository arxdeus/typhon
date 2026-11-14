# Copyright (C) 2026 the Typhon authors
# Generates data/finetune_rng.txt with torch 2.15.0 + numpy: run in the reference venv
#   python test/robot/gen_finetune_rng.py
# The same body is compiled by test_finetune_rng.py (between the BODY markers).
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "test_finetune_rng.py")).read()
body = src[src.index("# --- BODY\n"):src.index("# --- END BODY\n")]
lines = []
exec(body, {"emit": lines.append})
open(os.path.join(here, "data/finetune_rng.txt"), "w").write("\n".join(lines) + "\n")
print(len(lines), "lines")
