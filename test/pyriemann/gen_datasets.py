# Generates test_datasets.py (make_matrices, make_masks, make_outliers) from real pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.datasets import simulated as S
D = os.path.dirname(__file__)
src = open(os.path.join(D, "test_potato.py")).read()
head = src.split("X = signals(24, 4, 40)")[0]
head = head.replace("from pyriemann.artifact_detection import Potato, PotatoField\n", "from pyriemann.datasets.simulated import make_matrices, make_matrices_params, make_masks, make_outliers\n")
def lit(a): return repr(np.asarray(a, dtype=float).ravel().tolist())
L = []; T = []
def add(name, arr, call, atol=1e-10):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close({call}, E_{name}, atol={atol})")
for k in ["real", "inv", "orth", "sym", "spd", "spsd"]:
    add("m_" + k, S.make_matrices(3, 4, k, 42), f'make_matrices(3, 4, "{k}", 42)', 1e-9)
add("m_same", S.make_matrices(3, 4, "spd", 7, eigvecs_same=True, evals_low=1.0, evals_high=3.0), 'make_matrices(3, 4, "spd", 7, 1.0, 3.0, True)', 1e-9)
m, ev, _ = S.make_matrices(3, 4, "spd", 5, return_params=True)
add("m_ev", ev, 'make_matrices_params(3, 4, "spd", 5)[1]', 1e-9)
ms = S.make_masks(3, 6, 2, 11)
add("mask0", ms[0], "make_masks(3, 6, 2, 11)[0]", 1e-9)
add("mask2", ms[2], "make_masks(3, 6, 2, 11)[2]", 1e-9)
mean = S.make_matrices(1, 3, "spd", 1)[0]
add("out", S.make_outliers(4, mean, 0.5, 10, 3), "make_outliers(4, mean, 0.5, 10.0, 3)", 1e-8)
setup = '''mean = make_matrices(1, 3, "spd", 1)[0]

'''
open(os.path.join(D, "test_datasets.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_datasets():\n" + "\n".join(T) + "\ntest_datasets()\n")
