# Generates expected outcomes for test_autograd_semantics.py (torch 2.15.0).
import torch
import torch.nn.functional as F

def X(n, seed):
    return ((torch.arange(n, dtype=torch.float64) * 0.37 + seed).sin() * 2.0 + 3.0)

CASES = {
    "add": lambda x: (x + x).sum(), "sub": lambda x: (x - 1).sum(), "neg": lambda x: (-x).sum(),
    "view": lambda x: x.view(2, 2).sum(), "mean": lambda x: x.mean(), "cat": lambda x: torch.cat([x, x]).sum(),
    "flip": lambda x: x.flip(0).sum(), "slice": lambda x: x[1:].sum(), "expand": lambda x: x.expand(2, 4).sum(),
    "mulscalar": lambda x: (x * 2).sum(), "mul": lambda x: (x * x).sum(), "exp": lambda x: x.exp().sum(),
    "div": lambda x: (x / 2).sum(), "relu": lambda x: x.relu().sum(), "pow": lambda x: (x ** 2).sum(),
    "tanh": lambda x: x.tanh().sum(), "mm": lambda x: (x.view(2, 2) @ x.view(2, 2)).sum(),
    "linear": lambda x: F.linear(x.view(1, 4), x.view(1, 4)).sum(),
}

def two(f):
    x = X(4, 1).requires_grad_()
    y = f(x); y.backward()
    try:
        y.backward(); return "ok"
    except RuntimeError as e:
        return "ERR"

for k, f in CASES.items():
    print(f'    ("{k}", "{two(f)}"),')
x = X(4, 1).requires_grad_()
y = (x * x.exp()).sum(); y.backward(retain_graph=True); y.backward()
print("retain grad", x.grad.tolist())
g = torch.autograd.grad((x * x).sum(), [x], retain_graph=True)
print("autograd.grad", g[0].tolist())
