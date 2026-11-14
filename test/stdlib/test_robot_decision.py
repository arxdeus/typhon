# Robot-BCI library logic (examples/robot_bci/native/robot_bci.py): robot_stream_step's
# command decision (EMA, sort, threshold 0.3, margin 0.08, inside the API `with lock:`)
# on synthetic probabilities. Expected values: the same rule run in CPython + numpy.
# Guards against branches inside `with` resolving wrongly (silent miscompile seen in
# f6b624c2).
import numpy as np
from threading import Lock

lock = Lock()
alpha = 0.4
threshold = 0.3
smoothed_probs = np.ones(5) / 5.0

def _decide(probs) -> int:
    # Same body as robot_bci.robot_stream_step (EMA + sort + margin), inside a `with` block.
    global smoothed_probs
    with lock:
        try:
            smoothed_probs = alpha * smoothed_probs + (1 - alpha) * probs
            sorted_probs = np.sort(smoothed_probs)[::-1]
            max_prob = sorted_probs[0]
            second_prob = sorted_probs[1]
            max_idx = int(np.argmax(smoothed_probs))
            if max_prob >= threshold and (max_prob - second_prob) > 0.08:
                cmd = max_idx
            else:
                cmd = 0
            return cmd
        except Exception:
            return -1

@test
def test_decision_rule():
    global smoothed_probs
    smoothed_probs = np.ones(5) / 5.0
    seq = [
        [0.05, 0.05, 0.05, 0.05, 0.8],
        [0.05, 0.05, 0.05, 0.05, 0.8],
        [0.1, 0.1, 0.6, 0.1, 0.1],
        [0.1, 0.1, 0.6, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.0, 0.7, 0.1, 0.1, 0.1],
        [0.0, 0.7, 0.1, 0.1, 0.1],
        [0.0, 0.7, 0.1, 0.1, 0.1],
    ]
    got = [_decide(np.array(p)) for p in seq]
    # CPython 3.13 + numpy, same rule (run_inference loop body).
    assert got == [4, 4, 0, 2, 2, 1, 1, 1], got

def _with_branch(x: int) -> int:
    with lock:
        if x > 1:
            cmd = 7
        else:
            cmd = 0
        return cmd

@test
def test_branch_in_with():
    assert _with_branch(5) == 7
    assert _with_branch(0) == 0

test_decision_rule()
test_branch_in_with()
print("robot lib decision OK")
