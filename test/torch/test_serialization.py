# Fixture and literals from test/torch/gen_serialization_fixture.py (torch 2.15, safetensors 0.8).

import torch
from safetensors.torch import load_file, save_file, load, save

conv_bias = [-0.8566746115684509, 1.1006041765213013]
conv_bias_shape = [2]
conv_weight = [1.5409960746765137, -0.293428897857666, -2.1787893772125244, 0.5684312582015991, -1.0845223665237427, -1.3985954523086548, 0.40334683656692505, 0.8380263447761536, -0.7192575931549072, -0.40334352850914, -0.5966353416442871, 0.18203648924827576]
conv_weight_shape = [2, 1, 3, 2]
fc_weight = [-1.0711873631091473, 0.12270123916331216, -0.5663174751548545, 0.37311463418249957, -0.8919953148065625, -1.509107744632202, 0.3703935333738153, 1.4565026067903366, 0.9398099079535257, 0.7748488177844168, 0.19186942747902466, 1.2637947253235853]
fc_weight_shape = [3, 4]
mask = [1.0, 0.0, 1.0]
mask_shape = [3]
steps = [3.0, -7.0, 1099511627776.0]
steps_shape = [3]

DIR = __file__[:__file__.rfind("/")] if "/" in __file__ else "."

def same(t: torch.Tensor, vals: List[float], shp: List[int]) -> bool:
    return t.shape == shp and t.tolist() == vals

@test
def test_read_real_safetensors():
    d = load_file(DIR + "/data/tiny.safetensors")
    assert len(d) == 5
    assert d["conv.weight"].dtype == torch.float32 and same(d["conv.weight"], conv_weight, conv_weight_shape)
    assert same(d["conv.bias"], conv_bias, conv_bias_shape)
    assert d["fc.weight"].dtype == torch.float64 and same(d["fc.weight"], fc_weight, fc_weight_shape)
    assert d["steps"].dtype == torch.int64 and int(d["steps"][2]) == 1099511627776
    assert d["mask"].dtype == torch.bool and same(d["mask"], mask, mask_shape)

@test
def test_roundtrip():
    d = load_file(DIR + "/data/tiny.safetensors")
    blob = save(d)
    e = load(blob)
    for k in d.keys():
        assert e[k].dtype == d[k].dtype and torch.equal(e[k], d[k])
    path = "/tmp/_codon_torch_ser_test.pt"
    torch.manual_seed(0)
    sd = {"a": torch.randn(3, 4).t(), "b": torch.arange(5)}
    torch.save(sd, path)
    r = torch.load(path)
    assert torch.equal(r["a"], sd["a"]) and r["a"].shape == [4, 3] and torch.equal(r["b"], sd["b"])

test_read_real_safetensors()
test_roundtrip()
