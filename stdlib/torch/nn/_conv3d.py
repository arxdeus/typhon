# Ported from torch 2.15.0 (BSD-3-Clause)
#
# nn.Conv3d and F.conv3d (im2col/vol2col + BLAS, torch._conv).

from torch._tensor import Tensor, Node, _O, _needs, _attach, _pad, _DType
from torch._conv import conv3d_forward, conv3d_backward
from torch.nn.modules.layers import _ConvNd

def _triple(v) -> List[int]:
    if isinstance(v, int):
        return [v, v, v]
    else:
        return [int(e) for e in v]

class _Conv3dNode(Node):
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    groups: int
    has_b: bool

    def __init__(self, x: Tensor, w: Tensor, b: Optional[Tensor], stride: List[int], padding: List[int], dilation: List[int], groups: int):
        ins = [x, w]
        if b is not None:
            ins.append(b)
        super().__init__("ConvolutionBackward0", ins)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_b = b is not None

    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        x = self.inputs[0]
        w = self.inputs[1]
        gx, gw, gb = conv3d_backward(x.detach(), w.detach(), g, self.stride, self.padding, self.dilation, self.groups,
                                     x.requires_grad, w.requires_grad, self.has_b)
        out = [_O(gx) if x.requires_grad else None, _O(gw) if w.requires_grad else None]
        if self.has_b:
            out.append(_O(gb))
        return out

def _same_pads(size: int, k: int, d: int) -> Tuple[int, int]:
    total = d * (k - 1)
    left = total // 2
    return (left, total - left)

def conv3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride = 1, padding = 0, dilation = 1, groups: int = 1) -> Tensor:
    if input.ndim != 4 and input.ndim != 5:
        raise RuntimeError(f"Expected 4D (unbatched) or 5D (batched) input to conv3d, but got input of size: {input.shape}")
    batched = input.ndim == 5
    x = input if batched else input.unsqueeze(0)
    st = _triple(stride)
    dl = _triple(dilation)
    pd = [0, 0, 0]
    if isinstance(padding, str):
        if padding == "same":
            if st != [1, 1, 1]:
                raise RuntimeError("padding='same' is not supported for strided convolutions")
            pads = List[int]()
            sym = True
            for i in range(3):
                l, r = _same_pads(x._shape[2 + i], weight._shape[2 + i], dl[i])
                pd[i] = l
                if l != r:
                    sym = False
            if not sym:
                flat = List[int]()
                for i in range(2, -1, -1):
                    l, r = _same_pads(x._shape[2 + i], weight._shape[2 + i], dl[i])
                    flat.append(l)
                    flat.append(r)
                x = _pad(x, flat, "constant", 0.0)
                pd = [0, 0, 0]
        elif padding != "valid":
            raise ValueError(f"Invalid padding string '{padding}', should be one of {{'valid', 'same'}}")
    else:
        pd = _triple(padding)
    if x._shape[1] != weight._shape[1] * groups:
        raise RuntimeError(f"Given groups={groups}, weight of size {weight.shape}, expected input{x.shape} to have {weight._shape[1] * groups} channels, but got {x._shape[1]} channels instead")
    out = conv3d_forward(x.detach(), weight.detach(), bias.detach() if bias is not None else None, st, pd, dl, groups)
    if _needs([x, weight] + ([bias.__val__()] if bias is not None else List[Tensor]())):
        _attach(out, _Conv3dNode(x, weight, bias, st, pd, dl, groups))
    return out if batched else out.squeeze(0)

class Conv3d(_ConvNd):
    _f_Conv3d: int

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1,
                 groups: int = 1, bias: bool = True, padding_mode: str = "zeros", device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self._f_Conv3d = 0
        self._setup(3, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, dtype, False, 0)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        if self.padding_mode != "zeros":
            mode = "constant"
            if self.padding_mode == "reflect":
                mode = "reflect"
            elif self.padding_mode == "replicate":
                mode = "replicate"
            elif self.padding_mode == "circular":
                mode = "circular"
            x = _pad(input, self._rev_pad, mode, 0.0)
            return conv3d(x, weight, bias, self.stride, 0, self.dilation, self.groups)
        if self.padding_str == "same":
            return conv3d(input, weight, bias, self.stride, "same", self.dilation, self.groups)
        return conv3d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

    def _get_name(self) -> str:
        return "Conv3d"
