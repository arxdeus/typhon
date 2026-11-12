# Expected values for test/einops/test_einops.py from einops 0.8.2 + torch.
# Run: /Users/mind/orca/toolchains/bd-venv/bin/python test/einops/gen_expected.py
import torch, einops
from einops.layers.torch import Rearrange, Reduce
x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
y = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
def L(t): return f"{list(t.shape)} {[float(v) for v in t.reshape(-1).tolist()]}"
cases = [
 ("r1", einops.rearrange(x, "b c t -> b 1 c t")),
 ("r2", einops.rearrange(y, "b k e t -> b (e t) k")),
 ("r3", einops.rearrange(x, "b e (h w) -> b (h w) e", h=2)),
 ("r4", einops.rearrange(x, "b c (t w) -> b t c w", w=2)),
 ("r5", einops.rearrange(y, "... c t -> ... t c")),
 ("r6", einops.rearrange(y, "b c t 1 -> b 1 t c") if False else einops.rearrange(y[..., :1], "batch C T 1 -> batch 1 T C")),
 ("red_mean", einops.reduce(y, "b c (t k) e -> b c t", "mean", k=2)),
 ("red_max", einops.reduce(x, "b c t -> c", "max")),
 ("rep", einops.repeat(x, "b c t -> b c r t", r=2)),
 ("rep2", einops.repeat(x[0, 0], "t -> (2 t) 3")),
 ("layer_r", Rearrange("b c t -> b t c")(x)),
 ("layer_red", Reduce("b c t -> b c", "sum")(x)),
]
for n, t in cases: print(f"{n} = {L(t)}")
