# Checks every `*_typhon.py` copy in examples/Robot-BCI-Movement-GUI against its
# original: after the TYPHON header block, only the declared lines (original
# numbering) may differ and the line count must match. Run with CPython.
import os
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../examples/Robot-BCI-Movement-GUI")
END = "# --- END TYPHON HEADER ---\n"
ALLOWED = {
    "src/models/models_finetuning.py": {236, 237, 260, 266, 297, 325, 326, 381, 458, 459},
}

if not os.path.isdir(REPO):
    print("SKIP: examples/Robot-BCI-Movement-GUI absent")
    sys.exit(0)
bad = 0
for orig, allowed in ALLOWED.items():
    copy = orig[:-3] + "_typhon.py"
    a = open(os.path.join(REPO, orig), encoding="utf-8").read().splitlines(True)
    t = open(os.path.join(REPO, copy), encoding="utf-8").read().splitlines(True)
    if END not in t:
        print(f"{copy}: missing header end marker")
        bad += 1
        continue
    t = t[t.index(END) + 1:]
    if len(a) != len(t):
        print(f"{copy}: {len(t)} lines vs {len(a)} in original")
        bad += 1
        continue
    changed = {i + 1 for i, (x, y) in enumerate(zip(a, t)) if x != y}
    extra = sorted(changed - allowed)
    for ln in extra:
        print(f"{copy}:{ln}: undeclared change\n  - {a[ln - 1].rstrip()}\n  + {t[ln - 1].rstrip()}")
    bad += len(extra)
    print(f"{copy}: {len(changed)} declared lines changed, {len(extra)} undeclared")
sys.exit(1 if bad else 0)
