# torch ops added/fixed for braindecode group-A ports. Expected values from
# torch 2.15.0:
#   x = tensor([1., 2., 3.]); clamp(x, min=2.).sum().backward() -> [0, 0, 1]
#   clamp(x, max=2.) -> [1, 0, 0]; clamp(x, 1., 2.) -> [0, 0, 0]
#   clamp(x, min=tensor(2.), max=tensor(3.)) on [2, 3, 2.5] -> [0.5, 0.5, 1]
#   (tensor bounds split the gradient at ties; not reproduced here)

import torch

def _grad(mn, mx) -> List[float]:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    torch.clamp(x, min=mn, max=mx).sum().backward()
    return x.grad.tolist()

@test
def test_clamp_scalar_boundary_grad():
    assert _grad(2.0, None) == [0.0, 0.0, 1.0]
    assert _grad(None, 2.0) == [1.0, 0.0, 0.0]
    assert _grad(1.0, 2.0) == [0.0, 0.0, 0.0]
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    torch.clamp_min(x, 2.0).sum().backward()
    assert x.grad.tolist() == [0.0, 0.0, 1.0]
test_clamp_scalar_boundary_grad()

# ---- nn.Transformer (torch 2.15.0 reference, d_model=8 nhead=2 2+2 layers ff=16 batch_first)
E_keys = ['encoder.layers.0.self_attn.in_proj_weight', 'encoder.layers.0.self_attn.in_proj_bias', 'encoder.layers.0.self_attn.out_proj.weight', 'encoder.layers.0.self_attn.out_proj.bias', 'encoder.layers.0.linear1.weight', 'encoder.layers.0.linear1.bias', 'encoder.layers.0.linear2.weight', 'encoder.layers.0.linear2.bias', 'encoder.layers.0.norm1.weight', 'encoder.layers.0.norm1.bias', 'encoder.layers.0.norm2.weight', 'encoder.layers.0.norm2.bias', 'encoder.layers.1.self_attn.in_proj_weight', 'encoder.layers.1.self_attn.in_proj_bias', 'encoder.layers.1.self_attn.out_proj.weight', 'encoder.layers.1.self_attn.out_proj.bias', 'encoder.layers.1.linear1.weight', 'encoder.layers.1.linear1.bias', 'encoder.layers.1.linear2.weight', 'encoder.layers.1.linear2.bias', 'encoder.layers.1.norm1.weight', 'encoder.layers.1.norm1.bias', 'encoder.layers.1.norm2.weight', 'encoder.layers.1.norm2.bias', 'encoder.norm.weight', 'encoder.norm.bias', 'decoder.layers.0.self_attn.in_proj_weight', 'decoder.layers.0.self_attn.in_proj_bias', 'decoder.layers.0.self_attn.out_proj.weight', 'decoder.layers.0.self_attn.out_proj.bias', 'decoder.layers.0.multihead_attn.in_proj_weight', 'decoder.layers.0.multihead_attn.in_proj_bias', 'decoder.layers.0.multihead_attn.out_proj.weight', 'decoder.layers.0.multihead_attn.out_proj.bias', 'decoder.layers.0.linear1.weight', 'decoder.layers.0.linear1.bias', 'decoder.layers.0.linear2.weight', 'decoder.layers.0.linear2.bias', 'decoder.layers.0.norm1.weight', 'decoder.layers.0.norm1.bias', 'decoder.layers.0.norm2.weight', 'decoder.layers.0.norm2.bias', 'decoder.layers.0.norm3.weight', 'decoder.layers.0.norm3.bias', 'decoder.layers.1.self_attn.in_proj_weight', 'decoder.layers.1.self_attn.in_proj_bias', 'decoder.layers.1.self_attn.out_proj.weight', 'decoder.layers.1.self_attn.out_proj.bias', 'decoder.layers.1.multihead_attn.in_proj_weight', 'decoder.layers.1.multihead_attn.in_proj_bias', 'decoder.layers.1.multihead_attn.out_proj.weight', 'decoder.layers.1.multihead_attn.out_proj.bias', 'decoder.layers.1.linear1.weight', 'decoder.layers.1.linear1.bias', 'decoder.layers.1.linear2.weight', 'decoder.layers.1.linear2.bias', 'decoder.layers.1.norm1.weight', 'decoder.layers.1.norm1.bias', 'decoder.layers.1.norm2.weight', 'decoder.layers.1.norm2.bias', 'decoder.layers.1.norm3.weight', 'decoder.layers.1.norm3.bias', 'decoder.norm.weight', 'decoder.norm.bias']
E_sums = [2.5407923471648246, 0.0, -2.5887567626778036, 0.0, 2.3673261404037476, 0.7408175598829985, 3.145106077194214, -0.5234945714473724, 8.0, 0.0, 8.0, 0.0, -1.1535563561483286, 0.0, -0.1618956048041582, 0.0, -2.0431997776031494, 0.7408175598829985, 2.284085214138031, -0.5234945714473724, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, -5.997752940282226, 0.0, -0.8586779404431581, 0.0, -0.8017142023454653, 0.0, -2.4544382716412656, 0.0, 4.292617917060852, -1.251676907762885, 4.3716816902160645, 0.2430436909198761, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, -3.5548292273888364, 0.0, -0.7107851002365351, 0.0, -10.081040559860412, 0.0, -3.680395976640284, 0.0, 2.900583505630493, -1.251676907762885, 3.4441646933555603, 0.2430436909198761, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0]
E_out = [0.1007036343216896, 0.15029174089431763, 1.591066837310791, -1.1187560558319092, 1.0787862539291382, 0.5357824563980103, -1.1832704544067383, -1.1546039581298828, 1.0570679903030396, 0.019791308790445328, 1.2795321941375732, -1.3055685758590698, 0.6787829995155334, 0.5955308675765991, -1.2671856880187988, -1.057950735092163, -0.39613279700279236, -1.8699027299880981, -0.5771129131317139, -0.08363641053438187, -0.1625608503818512, 0.5796409249305725, 0.7575143575668335, 1.752190351486206, -1.254025936126709, -1.1396112442016602, 0.507388174533844, -0.5728418231010437, 0.9849531650543213, 1.6460946798324585, 0.5653620362281799, -0.7373191118240356, -0.03176864981651306, -0.09617755562067032, 1.350685715675354, -1.099220871925354, 0.8886099457740784, 1.228776216506958, -0.8496413230895996, -1.3912633657455444, 0.9136690497398376, -0.2568712532520294, 1.2272499799728394, -0.9545996785163879, 0.6448107361793518, 0.9700603485107422, -1.4821189641952515, -1.0621997117996216]
E_gabs = [2.478341457390343e-06, 1.580261709932529e-06, 5.912710321354098e-07, 1.6456367575301556e-06, 3.0195315048331395e-06, 4.434093341387779e-07, 3.5182395095034735e-06, 5.39380778263876e-07, 4.4179907376928895e-07, 6.770954996682121e-07, 4.3206770783399406e-07, 7.165414785959001e-07, 3.735360678547295e-06, 1.0643345831340412e-06, 9.257728379452601e-07, 6.048490490684344e-07, 3.470752290013479e-06, 5.152047037881857e-07, 4.685032763518393e-06, 6.816657105446211e-07, 4.0015964941630955e-07, 7.180493639680208e-07, 4.0871609030546097e-07, 7.180489092206699e-07, 7.195478701760294e-07, 1.0336569857827271e-06, 1.3327638953342102e-05, 1.8943569557450246e-06, 6.613204732275335e-06, 2.522889190004207e-06, 5.882054665562464e-06, 1.1488345990073867e-06, 2.7095161385659594e-06, 8.204663117794553e-07, 9.619250704417937e-06, 9.721361493575387e-07, 8.390488801524043e-06, 8.137676559272222e-07, 1.1606847465372994e-06, 8.50429785259621e-07, 1.1511403954500565e-06, 8.158260129675909e-07, 1.0860832162506995e-06, 1.1250136822127388e-06, 6.8015810938959476e-06, 7.441782372552552e-07, 3.6450244351726724e-06, 9.341033546661492e-07, 6.5843851189129055e-06, 1.0578615956546855e-06, 4.5099386625224724e-06, 9.95115215118858e-07, 7.146409643610241e-06, 9.665451443652273e-07, 5.936130492045777e-06, 1.4163435935188318e-06, 1.2735566770061268e-06, 1.0460539670020808e-06, 1.4182737686496694e-06, 1.2869647889601765e-06, 0.0009594979346729815, 0.0006185732781887054, 95.99903106689453, 61.75035858154297]

@test
def test_transformer():
    import math
    torch.manual_seed(0)
    m = torch.nn.Transformer(d_model=8, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=16, batch_first=True)
    sd = m.state_dict()
    assert [k for k in sd.keys()] == E_keys
    i = 0
    for k, v in sd.items():
        assert abs(v.double().sum().item() - E_sums[i]) < 1e-4
        i += 1
    m.eval()
    src = torch.tensor([math.sin(0.3 * i) for i in range(80)]).reshape(2, 5, 8)
    tgt = torch.tensor([math.cos(0.2 * i) for i in range(48)]).reshape(2, 3, 8)
    y = m(src, tgt)
    l = (y * y).sum()
    l.backward()
    yl = y.reshape(-1).tolist()
    for j in range(len(E_out)):
        assert abs(yl[j] - E_out[j]) < 1e-4
    ps = m.parameters()
    for j in range(len(ps)):
        assert abs(ps[j].grad.abs().sum().item() - E_gabs[j]) < 1e-3 * (1 + E_gabs[j])
test_transformer()

# CircularPad1d/2d (torch 2.15.0 reference values)
E_cp2 = [7.0, 4.0, 5.0, 6.0, 7.0, 11.0, 8.0, 9.0, 10.0, 11.0, 3.0, 0.0, 1.0, 2.0, 3.0, 7.0, 4.0, 5.0, 6.0, 7.0, 11.0, 8.0, 9.0, 10.0, 11.0, 3.0, 0.0, 1.0, 2.0, 3.0, 19.0, 16.0, 17.0, 18.0, 19.0, 23.0, 20.0, 21.0, 22.0, 23.0, 15.0, 12.0, 13.0, 14.0, 15.0, 19.0, 16.0, 17.0, 18.0, 19.0, 23.0, 20.0, 21.0, 22.0, 23.0, 15.0, 12.0, 13.0, 14.0, 15.0]
E_cp1 = [1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0]

@test
def test_circular_pad():
    x = torch.arange(24.0).reshape(1, 2, 3, 4)
    assert torch.nn.CircularPad2d((1, 0, 2, 1))(x).reshape(-1).tolist() == E_cp2
    y = torch.arange(6.0).reshape(1, 2, 3)
    assert torch.nn.CircularPad1d((2, 1))(y).reshape(-1).tolist() == E_cp1
test_circular_pad()
