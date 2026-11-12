# Generates expected values for test/braindecode/test_models*.py using real
# braindecode 2.0.0 / torch 2.15.0 (CPU, float32).
# Run: /path/to/bd-venv/bin/python test/braindecode/gen_models_expected.py [Model ...]
# Weights are set from a deterministic formula (mirrored in the .py tests)
# so forward/gradient checks do not depend on RNG parity. The default
# torch.manual_seed(0) init is also summarised (per-parameter sums) to check
# init parity.
import math
import sys
import warnings

warnings.simplefilter("ignore")
import torch
import braindecode.models as M

torch.set_num_threads(1)


def det_input(b, c, t):
    x = torch.zeros(b, c, t)
    for i in range(b):
        for j in range(c):
            for k in range(t):
                x[i, j, k] = math.sin(0.05 * k * (j + 1) + i) + 0.1 * math.cos(0.3 * k + j)
    return x


def det_value(name, pidx, i):
    v = math.sin(0.37 * i + 1.3 * pidx + 0.1)
    if name.endswith("running_var"):
        return 1.0 + 0.5 * abs(v)
    if name.endswith("running_mean"):
        return 0.05 * v
    return 0.1 * v


def set_det_weights(m):
    sd = m.state_dict()
    new = {}
    for pidx, (k, v) in enumerate(sd.items()):
        if not v.is_floating_point():
            new[k] = v
            continue
        flat = torch.tensor([det_value(k, pidx, i) for i in range(v.numel())], dtype=v.dtype)
        new[k] = flat.reshape(v.shape)
    m.load_state_dict(new)


def lit(t):
    return repr([float(v) for v in t.detach().reshape(-1).tolist()])


CASES = {
    "ShallowFBCSPNet": (lambda: M.ShallowFBCSPNet(n_chans=8, n_outputs=2, n_times=256, sfreq=128.0), (4, 8, 256)),
    "Deep4Net": (lambda: M.Deep4Net(n_chans=8, n_outputs=2, n_times=512), (4, 8, 512)),
    "EEGNet": (lambda: M.EEGNet(n_chans=8, n_outputs=2, n_times=256), (4, 8, 256)),
    "EEGConformer": (lambda: M.EEGConformer(n_chans=8, n_outputs=2, n_times=256), (4, 8, 256)),
    "SleepStagerChambon2018": (lambda: M.SleepStagerChambon2018(n_chans=2, n_outputs=5, n_times=3840, sfreq=128), (2, 2, 3840)),
    "SleepStagerBlanco2020": (lambda: M.SleepStagerBlanco2020(n_chans=2, n_outputs=5, n_times=3000, sfreq=100), (2, 2, 3000)),
    "SCCNet": (lambda: M.SCCNet(n_chans=8, n_outputs=4, n_times=500, sfreq=125), (2, 8, 500)),
    "EEGInceptionERP": (lambda: M.EEGInceptionERP(n_chans=8, n_outputs=2, n_times=128, sfreq=128), (2, 8, 128)),
    "EEGNeX": (lambda: M.EEGNeX(n_chans=8, n_outputs=4, n_times=256), (2, 8, 256)),
    "EEGSimpleConv": (lambda: M.EEGSimpleConv(n_chans=8, n_outputs=4, n_times=160, sfreq=80), (2, 8, 160)),
    "EEGInceptionMI": (lambda: M.EEGInceptionMI(n_chans=4, n_outputs=3, n_times=200, sfreq=100, n_convs=2, n_filters=6), (2, 4, 200)),
    "EEGITNet": (lambda: M.EEGITNet(n_chans=8, n_outputs=4, n_times=500), (2, 8, 500)),
    "BDTCN": (lambda: M.BDTCN(n_chans=4, n_outputs=2, n_times=200, n_blocks=3, n_filters=8, kernel_size=5), (2, 4, 200)),
    "EEGTCNet": (lambda: M.EEGTCNet(n_chans=8, n_outputs=4, n_times=512, kern_length=32), (2, 8, 512)),
    "CTNet": (lambda: M.CTNet(n_chans=8, n_outputs=4, n_times=512, n_filters_time=8, kernel_size=16, num_heads=2, embed_dim=16, num_layers=2), (2, 8, 512)),
    "SincShallowNet": (lambda: M.SincShallowNet(n_chans=8, n_outputs=4, n_times=500, sfreq=250), (2, 8, 500)),
    "ContraWR": (lambda: M.ContraWR(n_chans=2, n_outputs=5, n_times=500, sfreq=100, emb_size=32, res_channels=[8, 16]), (2, 2, 500)),
    "AttentionBaseNet": (lambda: M.AttentionBaseNet(n_chans=8, n_outputs=4, n_times=500), (2, 8, 500)),
    "FBCNet": (lambda: M.FBCNet(n_chans=4, n_outputs=2, n_times=256, sfreq=128), (2, 4, 256)),
    "ATCNet": (lambda: M.ATCNet(n_chans=4, n_outputs=3, n_times=448), (2, 4, 448)),
    "IFNet": (lambda: M.IFNet(n_chans=4, n_outputs=2, n_times=256, sfreq=128), (2, 4, 256)),
    "USleep": (lambda: M.USleep(n_chans=2, n_outputs=5, n_times=3000, sfreq=100, depth=4), (2, 2, 3000)),
    "MSVTNet": (lambda: M.MSVTNet(n_chans=4, n_outputs=3, n_times=448), (2, 4, 448)),
    "TIDNet": (lambda: M.TIDNet(n_chans=4, n_outputs=3, n_times=300), (2, 4, 300)),
    "FBLightConvNet": (lambda: M.FBLightConvNet(n_chans=4, n_outputs=2, n_times=500, sfreq=250), (2, 4, 500)),
    "FBMSNet": (lambda: M.FBMSNet(n_chans=4, n_outputs=2, n_times=250, sfreq=250), (2, 4, 250)),
    "SyncNet": (lambda: M.SyncNet(n_chans=6, n_outputs=3, n_times=400, sfreq=128), (2, 6, 400)),
    "SPARCNet": (lambda: M.SPARCNet(n_chans=4, n_outputs=3, n_times=128, block_layers=2, growth_rate=4, bottleneck_size=2), (2, 4, 128)),
    "TMSANet": (lambda: M.TMSANet(n_chans=6, n_outputs=3, n_times=300), (2, 6, 300)),
    "MSCFormer": (lambda: M.MSCFormer(n_chans=6, n_outputs=3, n_times=300, num_layers=2), (2, 6, 300)),
    "AttnSleep": (lambda: M.AttnSleep(n_chans=1, n_outputs=5, n_times=3000, sfreq=100), (2, 1, 3000)),
    "TCFormer": (lambda: M.TCFormer(n_chans=4, n_outputs=3, n_times=448), (2, 4, 448)),
    "DeepSleepNet": (lambda: M.DeepSleepNet(n_chans=1, n_outputs=5, n_times=3000, sfreq=100, bilstm_hidden_size=32), (2, 1, 3000)),
    "EEGSym": (lambda: M.EEGSym(chs_info=[{"ch_name": n} for n in ["C3", "C4", "Cz", "F3", "F4", "Fz"]], n_outputs=3, n_times=256, sfreq=128), (2, 6, 256)),
}


def run(name):
    ctor, shape = CASES[name]
    torch.manual_seed(0)
    m = ctor()
    n_params = sum(p.numel() for p in m.parameters())
    sd = m.state_dict()
    print(f"# ===== {name}")
    print(f"E_{name}_nparams = {n_params}")
    print(f"E_{name}_keys = {list(sd.keys())!r}")
    print(f"E_{name}_shapes = {[list(v.shape) for v in sd.values()]!r}")
    print(f"E_{name}_init_sums = {[float(v.double().sum()) for v in sd.values()]!r}")
    set_det_weights(m)
    x = det_input(*shape)
    m.eval()
    with torch.no_grad():
        out = m(x)
    print(f"E_{name}_out_shape = {list(out.shape)!r}")
    print(f"E_{name}_out = {lit(out)}")
    # gradients in eval mode (dropout off, BN uses running stats)
    m.zero_grad()
    out = m(x)
    loss = (out * out).sum() + out.sum()
    loss.backward()
    print(f"E_{name}_loss = {float(loss)!r}")
    names = [k for k, _ in m.named_parameters()]
    print(f"E_{name}_pnames = {names!r}")
    print(f"E_{name}_grad_sums = {[float(p.grad.double().sum()) if p.grad is not None else 0.0 for _, p in m.named_parameters()]!r}")
    print(f"E_{name}_grad_abs = {[float(p.grad.double().abs().sum()) if p.grad is not None else 0.0 for _, p in m.named_parameters()]!r}")




# ---------------------------------------------------------------- .codon emitter
# `--emit FILE Model ...` writes a self-contained Codon test file.
CODON_HEADER = '''
# Generated by test/braindecode/gen_models_expected.py (braindecode 2.0.0, torch 2.15.0).
# Do not edit by hand.

import math
import torch
from torch import Tensor
import braindecode.models as M

def det_input(b: int, c: int, t: int) -> Tensor:
    vals = List[float](capacity=b * c * t)
    for i in range(b):
        for j in range(c):
            for k in range(t):
                vals.append(math.sin(0.05 * k * (j + 1) + i) + 0.1 * math.cos(0.3 * k + j))
    return torch.tensor(vals, dtype=torch.float32).reshape(b, c, t)

def det_value(name: str, pidx: int, i: int) -> float:
    v = math.sin(0.37 * i + 1.3 * pidx + 0.1)
    if name.endswith("running_var"):
        return 1.0 + 0.5 * abs(v)
    if name.endswith("running_mean"):
        return 0.05 * v
    return 0.1 * v

def set_det_weights(m):
    sd = m.state_dict()
    new = Dict[str, Tensor]()
    pidx = 0
    for k, v in sd.items():
        if not v.is_floating_point():
            new[k] = v
        else:
            n = v.numel()
            vals = [det_value(k, pidx, i) for i in range(n)]
            new[k] = torch.tensor(vals, dtype=torch.float32).reshape(v.shape)
        pidx += 1
    m.load_state_dict(new)

def close(a: float, b: float, rtol: float, atol: float) -> bool:
    if abs(a - b) > atol + rtol * abs(b):
        print("mismatch", a, b)
        return False
    return True

def check_model(name: str, m, shape: List[int], nparams: int, keys: List[str], shapes: List[List[int]],
                init_sums: List[float], out_shape: List[int], out: List[float], loss: float,
                pnames: List[str], grad_sums: List[float], grad_abs: List[float]) -> bool:
    ok = True
    n = 0
    for p in m.parameters():
        n += p.numel()
    if n != nparams:
        print(name, "nparams", n, nparams); ok = False
    sd = m.state_dict()
    gk = [k for k in sd.keys()]
    if gk != keys:
        print(name, "keys", gk); ok = False
    else:
        for i in range(len(keys)):
            s = [int(d) for d in sd[keys[i]].shape]
            if s != shapes[i]:
                print(name, "shape", keys[i], s, shapes[i]); ok = False
            got = sd[keys[i]].double().sum().item()
            if not close(got, init_sums[i], 1e-4, 1e-4):
                print(name, "init sum", keys[i]); ok = False
    set_det_weights(m)
    x = det_input(shape[0], shape[1], shape[2])
    m.eval()
    y = torch.zeros(1)
    with torch.no_grad():
        y = m(x)
    if [int(d) for d in y.shape] != out_shape:
        print(name, "out shape", y.shape); return False
    yl = y.double().reshape(-1).tolist()
    for i in range(len(out)):
        if not close(yl[i], out[i], 1e-4, 1e-5):
            print(name, "out", i); ok = False
    m.zero_grad()
    y = m(x)
    l = (y * y).sum() + y.sum()
    l.backward()
    if not close(l.item(), loss, 1e-4, 1e-5):
        print(name, "loss"); ok = False
    named = m.named_parameters()
    if [k for k, _ in named] != pnames:
        print(name, "pnames", [k for k, _ in named]); return False
    for i in range(len(named)):
        g = named[i][1].grad
        gs = 0.0
        ga = 0.0
        if g is not None:
            gs = g.double().sum().item()
            ga = g.double().abs().sum().item()
        if not close(ga, grad_abs[i], 1e-3, 1e-5) or not close(gs, grad_sums[i], 1e-3, 1e-4 * grad_abs[i] + 1e-5):
            print(name, "grad", pnames[i]); ok = False
    return ok
'''

CTORS = {
    "ShallowFBCSPNet": "M.ShallowFBCSPNet(n_chans=8, n_outputs=2, n_times=256, sfreq=128.0)",
    "Deep4Net": "M.Deep4Net(n_chans=8, n_outputs=2, n_times=512)",
    "EEGNet": "M.EEGNet(n_chans=8, n_outputs=2, n_times=256)",
    "EEGConformer": "M.EEGConformer(n_chans=8, n_outputs=2, n_times=256)",
    "SleepStagerChambon2018": "M.SleepStagerChambon2018(n_chans=2, n_outputs=5, n_times=3840, sfreq=128.0)",
    "SleepStagerBlanco2020": "M.SleepStagerBlanco2020(n_chans=2, n_outputs=5, n_times=3000, sfreq=100.0)",
    "SCCNet": "M.SCCNet(n_chans=8, n_outputs=4, n_times=500, sfreq=125.0)",
    "EEGInceptionERP": "M.EEGInceptionERP(n_chans=8, n_outputs=2, n_times=128, sfreq=128.0)",
    "EEGNeX": "M.EEGNeX(n_chans=8, n_outputs=4, n_times=256)",
    "EEGSimpleConv": "M.EEGSimpleConv(n_chans=8, n_outputs=4, n_times=160, sfreq=80.0)",
    "EEGInceptionMI": "M.EEGInceptionMI(n_chans=4, n_outputs=3, n_times=200, sfreq=100.0, n_convs=2, n_filters=6)",
    "EEGITNet": "M.EEGITNet(n_chans=8, n_outputs=4, n_times=500)",
    "BDTCN": "M.BDTCN(n_chans=4, n_outputs=2, n_times=200, n_blocks=3, n_filters=8, kernel_size=5)",
    "EEGTCNet": "M.EEGTCNet(n_chans=8, n_outputs=4, n_times=512, kern_length=32)",
    "CTNet": "M.CTNet(n_chans=8, n_outputs=4, n_times=512, n_filters_time=8, kernel_size=16, num_heads=2, embed_dim=16, num_layers=2)",
    "SincShallowNet": "M.SincShallowNet(n_chans=8, n_outputs=4, n_times=500, sfreq=250.0)",
    "ContraWR": "M.ContraWR(n_chans=2, n_outputs=5, n_times=500, sfreq=100.0, emb_size=32, res_channels=[8, 16])",
    "AttentionBaseNet": "M.AttentionBaseNet(n_chans=8, n_outputs=4, n_times=500)",
    "FBCNet": "M.FBCNet(n_chans=4, n_outputs=2, n_times=256, sfreq=128.0)",
    "ATCNet": "M.ATCNet(n_chans=4, n_outputs=3, n_times=448)",
    "IFNet": "M.IFNet(n_chans=4, n_outputs=2, n_times=256, sfreq=128.0)",
    "USleep": "M.USleep(n_chans=2, n_outputs=5, n_times=3000, sfreq=100.0, depth=4)",
    "MSVTNet": "M.MSVTNet(n_chans=4, n_outputs=3, n_times=448)",
    "TIDNet": "M.TIDNet(n_chans=4, n_outputs=3, n_times=300)",
    "FBLightConvNet": "M.FBLightConvNet(n_chans=4, n_outputs=2, n_times=500, sfreq=250.0)",
    "FBMSNet": "M.FBMSNet(n_chans=4, n_outputs=2, n_times=250, sfreq=250.0)",
    "SyncNet": "M.SyncNet(n_chans=6, n_outputs=3, n_times=400, sfreq=128.0)",
    "SPARCNet": "M.SPARCNet(n_chans=4, n_outputs=3, n_times=128, block_layers=2, growth_rate=4, bottleneck_size=2)",
    "TMSANet": "M.TMSANet(n_chans=6, n_outputs=3, n_times=300)",
    "MSCFormer": "M.MSCFormer(n_chans=6, n_outputs=3, n_times=300, num_layers=2)",
    "AttnSleep": "M.AttnSleep(n_chans=1, n_outputs=5, n_times=3000, sfreq=100.0)",
    "TCFormer": "M.TCFormer(n_chans=4, n_outputs=3, n_times=448)",
    "DeepSleepNet": "M.DeepSleepNet(n_chans=1, n_outputs=5, n_times=3000, sfreq=100.0, bilstm_hidden_size=32)",
    "EEGSym": "M.EEGSym(chs_info=[{\"ch_name\": n} for n in [\"C3\", \"C4\", \"Cz\", \"F3\", \"F4\", \"Fz\"]], n_outputs=3, n_times=256, sfreq=128.0)",
}


def emit(path, names):
    import contextlib, io
    out = [CODON_HEADER]
    for n in names:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run(n)
        out.append(buf.getvalue())
        shape = list(CASES[n][1])
        f = ["nparams", "keys", "shapes", "init_sums", "out_shape", "out", "loss", "pnames", "grad_sums", "grad_abs"]
        args = ", ".join(f"E_{n}_{k}" for k in f)
        out.append(f'''
@test
def test_{n}():
    torch.manual_seed(0)
    m = {CTORS[n]}
    ok = check_model("{n}", m, {shape!r}, {args})
    assert ok
test_{n}()
''')
    open(path, "w").write("\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--emit":
        emit(sys.argv[2], sys.argv[3:] or list(CASES))
    else:
        for n in (sys.argv[1:] or list(CASES)):
            run(n)
