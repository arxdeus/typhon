# Expected values for test/torch/test_ops_b.py (torch 2.15.0).
import torch
torch.set_num_threads(1)
w = torch.tensor([[0.1 * (i + 1) + 0.01 * j for j in range(3)] for i in range(4)], requires_grad=True)
idx = torch.tensor([0, 2, 1, 2])
out = torch.nn.functional.embedding(idx, w, padding_idx=2)
(out * torch.arange(12.0).reshape(4, 3)).sum().backward()
print("EMB_GRAD =", w.grad.reshape(-1).tolist())
torch.manual_seed(0)
m = torch.nn.MultiheadAttention(4, 2, batch_first=True)
with torch.no_grad():
    for k, p in m.named_parameters():
        p.copy_(torch.sin(torch.arange(p.numel(), dtype=torch.float32) * 0.3 + len(k)).reshape(p.shape) * 0.5)
x = torch.cos(torch.arange(24, dtype=torch.float32) * 0.2).reshape(2, 3, 4)
kpm = torch.tensor([[False, False, True], [False, True, False]])
y, _ = m(x, x, x, key_padding_mask=kpm, need_weights=False)
print("MHA_KPM =", y.reshape(-1).tolist())
