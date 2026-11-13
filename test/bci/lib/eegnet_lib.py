# EEGNet (braindecode 2.0.0 defaults) as a native library: inference + on-device training.
# Input trials are float64 (n, 6, 500) C-contiguous, labels int64 in [0, 3).
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from braindecode.models import EEGNet

N_CH = 6
N_T = 500
N_CLS = 3

class Net:
    m: EEGNet
    opt: torch.optim.Adam
    lf: nn.CrossEntropyLoss
    def __init__(self, seed: int):
        torch.manual_seed(seed)
        self.m = EEGNet(n_chans=N_CH, n_outputs=N_CLS, n_times=N_T)
        self.opt = torch.optim.Adam(self.m.parameters(), lr=1e-3)
        self.lf = nn.CrossEntropyLoss()

nets = List[Net]()

def _x(p: Ptr[float], n: int) -> torch.Tensor:
    X = torch.empty(n, N_CH, N_T)
    q = Ptr[float32](X._st.ptr) + X._off
    for i in range(n * N_CH * N_T):
        q[i] = float32(p[i])
    return X

@export
def create(seed: int) -> int:
    nets.append(Net(seed))
    return len(nets) - 1

@export
def load_weights(h: int, cpath: cobj) -> int:
    nets[h].m.load_state_dict(load_file(str.from_ptr(cpath)))
    return 0

@export
def save_weights(h: int, cpath: cobj) -> int:
    sd = nets[h].m.state_dict()
    out = Dict[str, torch.Tensor]()
    for k, v in sd.items():
        out[k] = v.contiguous()
    save_file(out, str.from_ptr(cpath))
    return 0

@export
def set_lr(h: int, lr: float) -> int:
    for g in nets[h].opt.param_groups:
        g["lr"] = lr
    return 0

@export
def predict(h: int, p: Ptr[float], n: int, out: Ptr[float]) -> int:
    """Writes softmax probabilities (n, 3) to out; returns number of classes."""
    m = nets[h].m
    m.eval()
    with torch.no_grad():
        pr = torch.softmax(m(_x(p, n)), dim=1).contiguous()
    v = pr.tolist()
    for i in range(n * N_CLS):
        out[i] = v[i]
    return N_CLS

@export
def train_step(h: int, p: Ptr[float], y: Ptr[int], n: int) -> float:
    """One Adam step on a mini-batch; returns the batch loss."""
    net = nets[h]
    net.m.train()
    yy = torch.empty(n, dtype=torch.int64)
    for i in range(n):
        yy[i] = y[i]
    net.opt.zero_grad()
    l = net.lf(net.m(_x(p, n)), yy)
    l.backward()
    net.opt.step()
    return float(l)
