import difflib, os, subprocess, sys
LAB = os.environ.get("BCI_LAB", "/Users/mind/orca/codon/experiments/bci-research-lab")
REL = "research/src/models/log_power_lda_model/preprocessing.py"
B = b"# --- BEGIN verbatim preprocessing.py (bci-research-lab d617e8b) ---\n"
E = b"# --- END verbatim ---\n"
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_preprocessing_ort.py"), "rb").read()
a = src.index(B) + len(B)
got = src[a:src.index(E, a)]
want = subprocess.check_output(["git", "-C", LAB, "show", "HEAD:" + REL])
if got != want:
    sys.stdout.writelines(difflib.unified_diff(want.decode().splitlines(True), got.decode().splitlines(True), "HEAD", "test"))
    sys.exit(1)
print("verbatim OK (%d bytes)" % len(got))
