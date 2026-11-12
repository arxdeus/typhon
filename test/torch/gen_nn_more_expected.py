# Expected values for test/torch/test_nn_more.py (torch 2.15.0).
# Run: /Users/mind/orca/toolchains/bd-venv/bin/python test/torch/gen_nn_more_expected.py
import torch, torch.nn as nn, torch.nn.functional as F, math
def L(t): return "[" + ", ".join(repr(float(v)) for v in t.detach().reshape(-1).double().tolist()) + "]"
def X(shape, seed=1, dtype=torch.float64):
    n = math.prod(shape)
    return (torch.sin(torch.arange(n, dtype=torch.float64) * 0.37 + seed) * 2.0).reshape(shape).to(dtype)

torch.manual_seed(0)
ct = nn.ConvTranspose2d(4, 6, (2, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0), groups=2, dilation=(1, 2)).double()
x = X((2, 4, 3, 5), 1).requires_grad_()
y = ct(x)
print("ct2_w =", L(ct.weight)); print("ct2_shape =", list(y.shape)); print("ct2_y =", L(y))
(y * y).sum().backward()
print("ct2_gx =", L(x.grad)); print("ct2_gw =", L(ct.weight.grad)); print("ct2_gb =", L(ct.bias.grad))
torch.manual_seed(0)
ct1 = nn.ConvTranspose1d(3, 2, 4, stride=3, padding=1).double()
print("ct1_y =", L(ct1(X((2, 3, 5), 2))))

for mode in ["LSTM", "GRU", "RNN"]:
    torch.manual_seed(0)
    m = getattr(nn, mode)(3, 4, num_layers=2, batch_first=True, bidirectional=True).double()
    xi = X((2, 5, 3), 3).requires_grad_()
    out = m(xi)
    o = out[0]; h = out[1][0] if mode == "LSTM" else out[1]
    print(f"{mode.lower()}_keys =", list(m.state_dict().keys()))
    print(f"{mode.lower()}_out =", L(o)); print(f"{mode.lower()}_h =", L(h))
    (o.sum() + h.sum()).backward()
    print(f"{mode.lower()}_gx =", L(xi.grad))

torch.manual_seed(0)
inorm = nn.InstanceNorm1d(3, affine=True).double()
print("inorm =", L(inorm(X((2, 3, 6), 4))))
print("interp_nearest =", L(F.interpolate(X((1, 2, 5), 5), size=8, mode="nearest")))
print("interp_linear =", L(F.interpolate(X((1, 2, 5), 5), scale_factor=2.0, mode="linear")))
print("interp_linear_ac =", L(F.interpolate(X((1, 2, 5), 5), size=7, mode="linear", align_corners=True)))
print("upsample =", L(nn.Upsample(scale_factor=3.0)(X((1, 1, 4), 6))))
print("gnorm_mod =", L(nn.GroupNorm(2, 4).double()(X((2, 4, 3), 7))))
