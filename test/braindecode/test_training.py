# Expected values: braindecode 2.0.0 trial_preds_from_window_preds on np.arange input
import numpy as np
from braindecode.training import trial_preds_from_window_preds

@test
def test_trial_preds():
    p = np.arange(5 * 2 * 6).reshape(5, 2, 6).astype(float)
    r = trial_preds_from_window_preds(p, [0, 1, 2, 0, 1], [6, 9, 12, 6, 10])
    assert len(r) == 2
    assert r[0].shape == (2, 12) and r[1].shape == (2, 10)
    assert r[0].astype(int).tolist() == [[0, 1, 2, 3, 4, 5, 15, 16, 17, 27, 28, 29], [6, 7, 8, 9, 10, 11, 21, 22, 23, 33, 34, 35]]
    assert r[1].astype(int).tolist() == [[36, 37, 38, 39, 40, 41, 50, 51, 52, 53], [42, 43, 44, 45, 46, 47, 56, 57, 58, 59]]
    raised = False
    try:
        trial_preds_from_window_preds(p, [0, 2, 3, 0, 1], [6, 9, 12, 6, 10])
    except AssertionError:
        raised = True
    assert raised

test_trial_preds()
