# Writes test/torch/data/tiny.safetensors with real torch + safetensors and
# prints expected literals for test/torch/test_serialization.py.
import torch, os
from safetensors.torch import save_file
torch.manual_seed(0)
d = {
    "conv.weight": torch.randn(2, 1, 3, 2),
    "conv.bias": torch.randn(2),
    "fc.weight": torch.randn(3, 4, dtype=torch.float64),
    "steps": torch.tensor([3, -7, 2**40], dtype=torch.int64),
    "mask": torch.tensor([True, False, True]),
}
p = os.path.join(os.path.dirname(__file__), "data", "tiny.safetensors")
save_file(d, p, metadata={"format": "pt", "model": "tiny"})
for k in sorted(d):
    print(f"{k.replace('.', '_')} = {[float(v) for v in d[k].reshape(-1).tolist()]!r}")
    print(f"{k.replace('.', '_')}_shape = {list(d[k].shape)}")
