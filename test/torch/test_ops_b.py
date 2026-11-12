# Expected values from test/torch/gen_ops_b_expected.py (torch 2.15.0).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def close_list(a: List[float], b: List[float], tol: float) -> bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if abs(a[i] - b[i]) > tol * (1.0 + abs(b[i])):
            print("mismatch", i, a[i], b[i])
            return False
    return True

@test
def test_embedding_padding_idx_grad():
    w = torch.tensor([[0.1 * float(i + 1) + 0.01 * float(j) for j in range(3)] for i in range(4)], requires_grad=True)
    idx = torch.tensor([0, 2, 1, 2])
    out = F.embedding(idx, w, padding_idx=2)
    (out * torch.arange(12.0).reshape(4, 3)).sum().backward()
    assert close_list(w.grad.reshape(-1).tolist(),  [0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1e-6)
test_embedding_padding_idx_grad()

@test
def test_mha_key_padding_mask_only():
    torch.manual_seed(0)
    m = nn.MultiheadAttention(4, 2, batch_first=True)
    with torch.no_grad():
        for k, p in m.named_parameters():
            p.copy_((torch.sin(torch.arange(p.numel(), dtype=torch.float32) * 0.3 + float(len(k))) * 0.5).reshape(p.shape))
    x = torch.cos(torch.arange(24, dtype=torch.float32) * 0.2).reshape(2, 3, 4)
    kpm = torch.tensor([[False, False, True], [False, True, False]])
    y, _ = m(x, x, x, key_padding_mask=kpm, need_weights=False)
    assert close_list(y.reshape(-1).tolist(),  [0.04109080135822296, -0.9140654802322388, -0.30648842453956604, 1.2014143466949463, 0.053646668791770935, -0.8924939632415771, -0.30341091752052307, 1.1820731163024902, 0.0784369558095932, -0.8525974750518799, -0.2992875277996063, 1.1451647281646729, 0.8188729882240295, 0.5645117163658142, -0.012721151113510132, -0.06426727771759033, 0.8227145075798035, 0.5725875496864319, -0.010710030794143677, -0.0708855390548706, 0.8108115792274475, 0.5353755950927734, -0.02577519416809082, -0.04459160566329956], 1e-5)
test_mha_key_padding_mask_only()
