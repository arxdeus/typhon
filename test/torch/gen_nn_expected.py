# Generates expected values for test/torch/test_nn*.py, test_optim*.py,
# test_data*.py from real torch 2.15.0.
# Run: /Users/mind/orca/toolchains/bd-venv/bin/python test/torch/gen_nn_expected.py
import torch, torch.nn as nn, torch.nn.functional as F, math
torch.set_printoptions(precision=17)

def L(t):
    return "[" + ", ".join(repr(float(v)) for v in t.detach().reshape(-1).double().tolist()) + "]"

def sec(name):
    print(f"\n# ---- {name}")

# deterministic input
def X(shape, seed=1, dtype=torch.float32):
    # X = 2*sin(0.37*k + seed) (k = flat index), identical formula in the Codon tests
    n = math.prod(shape)
    return (torch.sin(torch.arange(n, dtype=torch.float64) * 0.37 + seed) * 2.0).reshape(shape).to(dtype)

sec("init parity: manual_seed(0); Conv2d(2,3,(1,3)); Linear(4,2); BatchNorm2d")
torch.manual_seed(0)
c = nn.Conv2d(2, 3, (1, 3))
l = nn.Linear(4, 2)
print("conv_w =", L(c.weight)); print("conv_b =", L(c.bias))
print("lin_w =", L(l.weight)); print("lin_b =", L(l.bias))
print("after =", L(torch.rand(3)))

sec("conv2d forward/backward, float64, stride/pad/dilation/groups")
x = X((2, 4, 5, 7), 1, torch.float64).requires_grad_()
w = X((6, 2, 2, 3), 2, torch.float64).requires_grad_()
b = X((6,), 3, torch.float64).requires_grad_()
y = F.conv2d(x, w, b, stride=(1, 2), padding=(1, 2), dilation=(2, 1), groups=2)
print("conv_shape =", list(y.shape)); print("conv_y =", L(y))
(y * y).sum().backward()
print("conv_gx =", L(x.grad)); print("conv_gw =", L(w.grad)); print("conv_gb =", L(b.grad))

sec("conv1d same + reflect pad")
x1 = X((2, 3, 9), 4, torch.float64)
w1 = X((4, 3, 4), 5, torch.float64)
print("conv1d_same =", L(F.conv1d(x1, w1, padding="same")))
print("pad_reflect =", L(F.pad(x1, (2, 3), mode="reflect")))
print("pad_replicate =", L(F.pad(x1, (1, 2), mode="replicate")))
print("pad_circular =", L(F.pad(x1, (2, 1), mode="circular")))
print("pad_const =", L(F.pad(x1, (1, 1, 0, 2), value=0.5)))

sec("pooling")
xp = X((2, 3, 6, 9), 6, torch.float64).requires_grad_()
mp = F.max_pool2d(xp, (2, 3), (2, 2), (1, 1), ceil_mode=True)
ap = F.avg_pool2d(xp, (3, 2), (2, 3), (1, 1), count_include_pad=False)
print("maxpool_shape =", list(mp.shape)); print("maxpool =", L(mp))
print("avgpool_shape =", list(ap.shape)); print("avgpool =", L(ap))
(mp.sum() + (ap * ap).sum()).backward()
print("pool_gx =", L(xp.grad))
print("adapt_avg =", L(F.adaptive_avg_pool2d(xp.detach(), (4, 5))))
print("avgpool1d =", L(F.avg_pool1d(x1, 3, 2)))
print("maxpool1d =", L(F.max_pool1d(x1, 2)))

sec("batchnorm train/eval")
torch.manual_seed(0)
bn = nn.BatchNorm2d(3, momentum=0.2).double()
xb = X((4, 3, 2, 5), 7, torch.float64)
yb = bn(xb)
print("bn_train =", L(yb)); print("bn_rm =", L(bn.running_mean)); print("bn_rv =", L(bn.running_var))
bn.eval(); print("bn_eval =", L(bn(xb)))
print("bn_keys =", list(bn.state_dict().keys()))

sec("activations / norms")
xa = X((3, 5), 8, torch.float64)
for n, f in [("elu", F.elu), ("gelu", F.gelu), ("gelu_tanh", lambda t: F.gelu(t, approximate="tanh")),
             ("silu", F.silu), ("lrelu", F.leaky_relu), ("hswish", F.hardswish), ("softmax", lambda t: F.softmax(t, 1)),
             ("logsoftmax", lambda t: F.log_softmax(t, 1)), ("softplus", F.softplus)]:
    print(f"act_{n} =", L(f(xa)))
print("layernorm =", L(F.layer_norm(xa, (5,))))
print("groupnorm =", L(F.group_norm(X((2, 4, 3), 9, torch.float64), 2)))

sec("losses")
logits = X((4, 3), 10, torch.float64).requires_grad_()
tgt = torch.tensor([0, 2, 1, 2])
wt = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
ce = F.cross_entropy(logits, tgt, weight=wt, label_smoothing=0.1)
print("ce =", repr(ce.item()))
ce.backward(); print("ce_grad =", L(logits.grad))
print("ce_plain =", repr(F.cross_entropy(logits, tgt).item()))
print("nll =", repr(F.nll_loss(F.log_softmax(logits, 1), tgt).item()))
print("mse =", repr(F.mse_loss(xa, xa * 0.5).item()))
print("bce_logits =", repr(F.binary_cross_entropy_with_logits(xa, (xa > 0).double()).item()))

sec("dropout masks with manual_seed")
torch.manual_seed(3)
print("dropout =", L(F.dropout(torch.ones(2, 8), 0.3)))
torch.manual_seed(3)
print("dropout2d =", L(F.dropout2d(torch.ones(2, 3, 2, 2), 0.5)))

sec("sequential state_dict keys")
torch.manual_seed(0)
m = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2), nn.ELU(), nn.Flatten(), nn.Linear(2 * 3 * 3, 4))
print("seq_keys =", list(m.state_dict().keys()))
print("seq_out =", L(m(X((2, 1, 5, 5), 11))))

sec("mha + transformer")
torch.manual_seed(0)
mha = nn.MultiheadAttention(8, 2, batch_first=True)
q = X((2, 3, 8), 12)
o, a = mha(q, q, q)
print("mha_out =", L(o)); print("mha_w =", L(a))
torch.manual_seed(0)
tl = nn.TransformerEncoderLayer(8, 2, 16, 0.0, activation="gelu", batch_first=True, norm_first=True)
print("tel_out =", L(tl(q)))
print("tel_keys =", list(tl.state_dict().keys()))

sec("parametrize maxnorm")
import torch.nn.utils.parametrize as P
class MaxNorm(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, X):
        n = X.reshape(X.shape[0], -1).norm(p=2, dim=1, keepdim=True)
        d = n.clamp(min=self.m / 2); num = d.clamp(max=self.m)
        return X * (num / (d + 1e-5)).reshape([-1] + [1] * (X.dim() - 1))
torch.manual_seed(0)
lin = nn.Linear(5, 3)
P.register_parametrization(lin, "weight", MaxNorm(0.5))
print("ptz_keys =", list(lin.state_dict().keys()))
print("ptz_w =", L(lin.weight))
lin(torch.ones(2, 5)).sum().backward()
print("ptz_grad =", L(lin.parametrizations.weight.original.grad))

sec("optim: 3 steps on quadratic loss")
def run_opt(make):
    torch.manual_seed(0)
    lin = nn.Linear(3, 2).double()
    opt = make(lin.parameters())
    xin = X((4, 3), 13, torch.float64)
    for _ in range(3):
        opt.zero_grad()
        loss = (lin(xin) ** 2).mean()
        loss.backward()
        opt.step()
    return lin
print("sgd_w =", L(run_opt(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=0.01, nesterov=True)).weight))
print("adam_w =", L(run_opt(lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.1)).weight))
print("adamw_w =", L(run_opt(lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1)).weight))
print("adamw_ams =", L(run_opt(lambda p: torch.optim.AdamW(p, lr=0.01, amsgrad=True)).weight))
opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5, eta_min=0.001)
lrs = []
for _ in range(12):
    opt.step(); sch.step(); lrs.append(opt.param_groups[0]["lr"])
print("cos_lrs =", repr(lrs))
opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
sch = torch.optim.lr_scheduler.StepLR(opt, 3, 0.5)
lrs = []
for _ in range(7):
    opt.step(); sch.step(); lrs.append(opt.param_groups[0]["lr"])
print("step_lrs =", repr(lrs))

sec("data loader shuffle order")
from torch.utils.data import TensorDataset, DataLoader, random_split
torch.manual_seed(5)
ds = TensorDataset(torch.arange(10.).reshape(10, 1), torch.arange(10))
dl = DataLoader(ds, batch_size=4, shuffle=True)
order = []
for ep in range(2):
    for xb, yb in dl: order.extend(yb.tolist())
print("dl_order =", repr(order))
g = torch.Generator().manual_seed(7)
dl = DataLoader(ds, batch_size=3, shuffle=True, generator=g, drop_last=True)
print("dl_gen_order =", repr([v for xb, yb in dl for v in yb.tolist()]))
print("after_dl =", L(torch.rand(2)))
a, b = random_split(ds, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
print("split_idx =", repr(a.indices + b.indices))

sec("weight_norm")
from torch.nn.utils.parametrizations import weight_norm
torch.manual_seed(0)
c = weight_norm(nn.Conv1d(2, 3, 2))
print("wn_keys =", list(c.state_dict().keys()))
y = c(X((1, 2, 4), 14))
y.sum().backward()
print("wn_y =", L(y))
print("wn_g0 =", L(c.parametrizations.weight.original0.grad))
print("wn_g1 =", L(c.parametrizations.weight.original1.grad))

sec("clip_grad_norm")
torch.manual_seed(0)
lin = nn.Linear(4, 3).double()
(lin(X((2, 4), 15, torch.float64)) ** 2).sum().backward()
tn = torch.nn.utils.clip_grad_norm_(lin.parameters(), 0.5)
print("clip_total =", repr(tn.item()))
print("clip_gw =", L(lin.weight.grad))
