# Generates expected values for test_audit_fixes.py (torch 2.15.0).
import torch
from torch.utils.data import TensorDataset, DataLoader

def X(shape, seed):
    n = 1
    for s in shape: n *= s
    return ((torch.arange(n, dtype=torch.float64) * 0.37 + seed).sin() * 2.0).reshape(shape)

def L(t): return t.detach().double().contiguous().reshape(-1).tolist()

print("# dataloader")
ds = TensorDataset(torch.arange(23).double())
g = torch.Generator(); g.manual_seed(42)
dl = DataLoader(ds, batch_size=5, shuffle=True, generator=g)
print("DL_GEN =", [[int(v) for v in L(b[0])] for ep in range(5) for b in dl])
torch.manual_seed(7)
dl = DataLoader(ds, batch_size=6, shuffle=True)
print("DL_GLOBAL =", [[int(v) for v in L(b[0])] for ep in range(3) for b in dl])
torch.manual_seed(3)
dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
print("DL_DROP =", [[int(v) for v in L(b[0])] for ep in range(3) for b in dl])
print("AFTER =", L(torch.rand(3)))

print("# rnn")
torch.manual_seed(0)
lstm = torch.nn.LSTM(3, 4, num_layers=2, bidirectional=True, batch_first=True).double()
gru = torch.nn.GRU(3, 4).double()
rnn = torch.nn.RNN(3, 4, nonlinearity="relu").double()
x = X([2, 5, 3], 1)
o, (h, c) = lstm(x); print("LSTM_O =", L(o)); print("LSTM_C =", L(c))
o, h = gru(x); print("GRU_H =", L(h))
o, h = rnn(x); print("RNN_O =", L(o))

print("# normal")
torch.manual_seed(11)
print("N_TT =", L(torch.normal(X([2, 3], 2), X([3], 3).abs())))
print("N_TF =", L(torch.normal(X([4], 4), 0.5)))
print("N_FT =", L(torch.normal(1.5, X([4], 5).abs())))
print("N_FF =", L(torch.normal(0.5, 2.0, (5,))))
g = torch.Generator(); g.manual_seed(5)
print("N_G =", L(torch.normal(X([3], 6), X([3], 7).abs(), generator=g)))

print("# cum ops")
a = X([3, 4], 8).requires_grad_()
cp = torch.cumprod(a, 1); print("CP =", L(cp))
(cp * X([3, 4], 9)).sum().backward(); print("CP_G =", L(a.grad))
print("CP0 =", L(a.cumprod(0)))
print("CP_INT =", L(torch.cumprod(torch.tensor([[1, -2, 3], [2, 0, 4]]), 1)), str(torch.cumprod(torch.tensor([1, 2]), 0).dtype))
b = torch.tensor([1., 3., 3., 2., float('nan'), 5., 0.])
r = torch.cummax(b, 0); print("CMAX =", L(r.values), L(r.indices))
r = torch.cummin(b, 0); print("CMIN =", L(r.values), L(r.indices))
r = torch.cummax(torch.tensor([2, 2, 1, 5]), 0); print("CMAX_INT =", L(r.values), L(r.indices))
a.grad = None
r = a.cummax(1); print("CMAX2 =", L(r.values), L(r.indices))
(r.values * X([3, 4], 10)).sum().backward(); print("CMAX2_G =", L(a.grad))
a.grad = None
l = torch.logcumsumexp(a, 0); print("LCSE =", L(l))
(l * X([3, 4], 11)).sum().backward(); print("LCSE_G =", L(a.grad))
print("LCSE_INF =", L(torch.logcumsumexp(torch.tensor([float('-inf'), float('-inf'), 1.0, 1000.0], dtype=torch.float64), 0)))
