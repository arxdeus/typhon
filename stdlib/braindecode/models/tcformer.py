# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.functional import apply_rotary, rotary_angles, rotary_frequencies, scaled_dot_product_attention
from braindecode.functional.initialization import _weight_leaf, _param_by_name
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import CausalConv1d, Conv1dWithConstraint, DropPath

def _init_weights(m):
    n = m._get_name()
    if n in ("Conv1d", "Conv2d", "Linear", "CausalConv1d", "Conv1dWithConstraint"):
        w = _weight_leaf(m, True)
        if w is not None:
            nn.init.xavier_uniform_(w.__val__())
        b = _param_by_name(m, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())

class _GroupedQueryAttention(nn.Module):
    n_query_heads: int
    n_kv_heads: int
    head_dim: int
    scale: float
    query_proj: nn.Linear
    key_value_proj: nn.Linear
    out_proj: nn.Linear
    drop: nn.Dropout
    p: float

    def __init__(self, d_model: int, q_heads: int, kv_heads: int, drop_prob: float = 0.4):
        super().__init__()
        assert d_model % q_heads == 0, "d_model must be divisible by q_heads"
        assert q_heads % kv_heads == 0, "q_heads must be a multiple of kv_heads"
        self.n_query_heads = q_heads
        self.n_kv_heads = kv_heads
        self.head_dim = d_model // q_heads
        assert self.head_dim % 2 == 0, "head_dim (d_model // q_heads) must be even for rotary embeddings"
        self.scale = self.head_dim ** (-0.5)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.add_module("query_proj", self.query_proj)
        self.key_value_proj = nn.Linear(d_model, 2 * kv_heads * self.head_dim, bias=False)
        self.add_module("key_value_proj", self.key_value_proj)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.add_module("out_proj", self.out_proj)
        self.drop = nn.Dropout(drop_prob)
        self.add_module("drop", self.drop)
        self.p = drop_prob

    def _fwd(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        L = x.shape[1]
        q = self.query_proj(x).unflatten(-1, [self.n_query_heads, -1]).transpose(1, 2)
        kv = self.key_value_proj(x).unflatten(-1, [self.n_kv_heads, 2, -1]).permute(3, 0, 2, 1, 4).unbind(0)
        g = self.n_query_heads // self.n_kv_heads
        k = torch.repeat_interleave(kv[0], g, dim=1)
        v = torch.repeat_interleave(kv[1], g, dim=1)
        q = apply_rotary(q, cos[:L], sin[:L], False)
        k = apply_rotary(k, cos[:L], sin[:L], False)
        ctx = scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0, scale=self.scale, exact=True)
        return self.out_proj(ctx.transpose(1, 2).flatten(2))

    def _get_name(self) -> str:
        return "_GroupedQueryAttention"

class _TransformerBlock(nn.Module):
    norm1: nn.LayerNorm
    attn: _GroupedQueryAttention
    drop_path: DropPath
    norm2: nn.LayerNorm
    mlp: nn.Sequential

    def __init__(self, d_model: int, q_heads: int, kv_heads: int, mlp_ratio: int = 2, drop_prob: float = 0.4,
                 drop_path_rate: float = 0.0, activation = nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.add_module("norm1", self.norm1)
        self.attn = _GroupedQueryAttention(d_model, q_heads, kv_heads, drop_prob)
        self.add_module("attn", self.attn)
        self.drop_path = DropPath(drop_path_rate)
        self.add_module("drop_path", self.drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        self.add_module("norm2", self.norm2)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), activation(),
                                 nn.Linear(mlp_ratio * d_model, d_model), nn.Dropout(drop_prob))
        self.add_module("mlp", self.mlp)

    def _fwd(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn._fwd(self.norm1(x), cos, sin))
        return x + self.drop_path(self.mlp(self.norm2(x)))

    def _get_name(self) -> str:
        return "_TransformerBlock"

class _ChannelGroupAttention(nn.Module):
    num_groups: int
    pool: nn.AdaptiveAvgPool2d
    att_fc1: nn.Conv2d
    att_fc2: nn.Conv2d
    relu: nn.ReLU
    sigmoid: nn.Sigmoid

    def __init__(self, in_channels: int, num_groups: int, reduction: int = 4):
        super().__init__()
        assert in_channels % num_groups == 0
        self.num_groups = num_groups
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.add_module("pool", self.pool)
        self.att_fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, groups=num_groups, bias=False)
        self.add_module("att_fc1", self.att_fc1)
        self.att_fc2 = nn.Conv2d(in_channels // reduction, num_groups, 1, groups=num_groups, bias=False)
        self.add_module("att_fc2", self.att_fc2)
        self.relu = nn.ReLU()
        self.add_module("relu", self.relu)
        self.sigmoid = nn.Sigmoid()
        self.add_module("sigmoid", self.sigmoid)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.sigmoid(self.att_fc2(self.relu(self.att_fc1(self.pool(x)))))
        return (x.unflatten(1, [self.num_groups, -1]) * gate.unsqueeze(2)).flatten(1, 2)

    def _get_name(self) -> str:
        return "_ChannelGroupAttention"

class _MultiKernelConvBlock(nn.Module):
    d_model: int
    temporal_convs: nn.ModuleList
    channel_dw_conv: nn.Sequential
    pool1: nn.AvgPool2d
    drop1: nn.Dropout
    use_channel_reduction: bool
    channel_reduction: Optional[nn.Sequential]
    temporal_conv_2: nn.Sequential
    use_group_attn: bool
    group_attn: Optional[_ChannelGroupAttention]
    pool2: nn.AvgPool2d
    drop2: nn.Dropout

    def __init__(self, n_chans: int, temp_kernel_lengths: List[int], n_filters_time: int, depth_multiplier: int,
                 pool_length_1: int, pool_length_2: int, temp_kernel_length_2: int, drop_prob: float, group_dim: int,
                 se_reduction: int, activation):
        super().__init__()
        ng = len(temp_kernel_lengths)
        self.d_model = group_dim * ng
        self.add_module("rearrange", Rearrange("batch nchans time -> batch 1 nchans time"))
        self.temporal_convs = nn.ModuleList()
        self.add_module("temporal_convs", self.temporal_convs)
        for k in temp_kernel_lengths:
            pad = (k // 2 - 1, k // 2, 0, 0) if k % 2 == 0 else (k // 2, k // 2, 0, 0)
            self.temporal_convs.append(nn.Sequential(nn.ConstantPad2d(pad, 0.0), nn.Conv2d(1, n_filters_time, (1, k), bias=False),
                                                     nn.BatchNorm2d(n_filters_time)))
        in_dw = n_filters_time * ng
        f2 = in_dw * depth_multiplier
        self.channel_dw_conv = nn.Sequential(nn.Conv2d(in_dw, f2, (n_chans, 1), bias=False, groups=in_dw),
                                             nn.BatchNorm2d(f2), activation())
        self.add_module("channel_dw_conv", self.channel_dw_conv)
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.add_module("pool1", self.pool1)
        self.drop1 = nn.Dropout(drop_prob)
        self.add_module("drop1", self.drop1)
        self.use_channel_reduction = self.d_model != f2
        self.channel_reduction = None
        if self.use_channel_reduction:
            cr = nn.Sequential(nn.Conv2d(f2, self.d_model, (1, 1), bias=False, groups=ng), nn.BatchNorm2d(self.d_model))
            self.channel_reduction = cr
            self.add_module("channel_reduction", cr)
        self.temporal_conv_2 = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, (1, temp_kernel_length_2), padding="same",
                                                       bias=False, groups=ng), nn.BatchNorm2d(self.d_model), activation())
        self.add_module("temporal_conv_2", self.temporal_conv_2)
        self.use_group_attn = ng > 1
        self.group_attn = None
        if self.use_group_attn:
            ga = _ChannelGroupAttention(self.d_model, ng, se_reduction)
            self.group_attn = ga
            self.add_module("group_attn", ga)
        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.add_module("pool2", self.pool2)
        self.drop2 = nn.Dropout(drop_prob)
        self.add_module("drop2", self.drop2)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = torch.cat([c(x) for c in self.temporal_convs.children()], dim=1)
        x = self.drop1(self.pool1(self.channel_dw_conv(x)))
        if self.channel_reduction is not None:
            x = self.channel_reduction.__val__()(x)
        x = self.temporal_conv_2(x)
        if self.group_attn is not None:
            x = x + self.group_attn.__val__()(x)
        return self.drop2(self.pool2(x)).squeeze(2)

    def _get_name(self) -> str:
        return "_MultiKernelConvBlock"

class _TCNResidualBlock(nn.Module):
    conv1: CausalConv1d
    bn1: nn.BatchNorm1d
    act1: nn.Module
    drop1: nn.Dropout
    conv2: CausalConv1d
    bn2: nn.BatchNorm1d
    act2: nn.Module
    drop2: nn.Dropout
    act3: nn.Module

    def __init__(self, n_filters: int, kernel_length: int, dilation: int, n_groups: int, drop_prob: float, activation):
        super().__init__()
        self.conv1 = CausalConv1d(n_filters, n_filters, kernel_length, dilation=dilation, groups=n_groups)
        self.add_module("conv1", self.conv1)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.add_module("bn1", self.bn1)
        self.act1 = activation()
        self.add_module("act1", self.act1)
        self.drop1 = nn.Dropout(drop_prob)
        self.add_module("drop1", self.drop1)
        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_length, dilation=dilation, groups=n_groups)
        self.add_module("conv2", self.conv2)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.add_module("bn2", self.bn2)
        self.act2 = activation()
        self.add_module("act2", self.act2)
        self.drop2 = nn.Dropout(drop_prob)
        self.add_module("drop2", self.drop2)
        self.act3 = activation()
        self.add_module("act3", self.act3)

    def forward(self, x: Tensor) -> Tensor:
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return self.act3(x + out)

    def _get_name(self) -> str:
        return "_TCNResidualBlock"

class _TCN(nn.Module):
    blocks: nn.ModuleList
    def __init__(self, depth: int, kernel_length: int, n_filters: int, n_groups: int, drop_prob: float, activation):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.add_module("blocks", self.blocks)
        for i in range(depth):
            self.blocks.append(_TCNResidualBlock(n_filters, kernel_length, 1 << i, n_groups, drop_prob, activation))
    def forward(self, x: Tensor) -> Tensor:
        for b in self.blocks.children():
            x = b(x)
        return x
    def _get_name(self) -> str:
        return "_TCN"

class _ClassificationHead(nn.Module):
    n_groups: int
    n_outputs: int
    conv: Conv1dWithConstraint
    def __init__(self, d_features: int, n_groups: int, n_outputs: int, max_norm: float = 0.25):
        super().__init__()
        self.n_groups = n_groups
        self.n_outputs = n_outputs
        self.conv = Conv1dWithConstraint(d_features, n_outputs * n_groups, kernel_size=1, groups=n_groups, max_norm=max_norm)
        self.add_module("conv", self.conv)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x).squeeze(-1)
        return x.view(x.size(0), self.n_groups, self.n_outputs).mean(dim=1)
    def _get_name(self) -> str:
        return "_ClassificationHead"

class TCFormer(EEGModuleMixin, nn.Module):
    """TCFormer from Altaheri et al (2025)."""
    head_dim: int
    conv_block: _MultiKernelConvBlock
    mix: nn.Sequential
    to_tokens: Rearrange
    transformer: nn.ModuleList
    _blocks: List[_TransformerBlock]
    reduce: nn.Sequential
    tcn: _TCN
    final_layer: _ClassificationHead

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, n_filters_time: int = 32,
                 temp_kernel_lengths = (20, 32, 64), depth_multiplier: int = 2, pool_length_1: int = 8,
                 pool_length_2: int = 7, temp_kernel_length_2: int = 16, group_dim: int = 16, se_reduction: int = 4,
                 n_transformer_layers: int = 2, q_heads: int = 4, kv_heads: int = 2, mlp_ratio: int = 2,
                 drop_path_max: float = 0.25, tcn_depth: int = 2, tcn_kernel_length: int = 4,
                 classifier_max_norm: float = 0.25, drop_prob_conv: float = 0.4, drop_prob_trans: float = 0.4,
                 drop_prob_tcn: float = 0.3, activation = nn.ELU, activation_ffn = nn.GELU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        tkl = [int(k) for k in temp_kernel_lengths]
        ng = len(tkl)
        d_model = group_dim * ng
        d_fused = group_dim * (ng + 1)
        self.conv_block = _MultiKernelConvBlock(self.n_chans, tkl, n_filters_time, depth_multiplier, pool_length_1,
                                                pool_length_2, temp_kernel_length_2, drop_prob_conv, group_dim, se_reduction, activation)
        self.add_module("conv_block", self.conv_block)
        self.mix = nn.Sequential(nn.Conv1d(d_model, d_model, 1, bias=False), nn.BatchNorm1d(d_model), nn.SiLU())
        self.add_module("mix", self.mix)
        self.to_tokens = Rearrange("batch feat time -> batch time feat")
        self.add_module("to_tokens", self.to_tokens)
        dr = (torch.linspace(0.0, 1.0, n_transformer_layers) ** 2 * drop_path_max).tolist()
        self._blocks = [_TransformerBlock(d_model, q_heads, kv_heads, mlp_ratio, drop_prob_trans, float(dr[i]), activation_ffn)
                        for i in range(n_transformer_layers)]
        self.transformer = nn.ModuleList()
        for b in self._blocks:
            self.transformer.append(b)
        self.add_module("transformer", self.transformer)
        self.head_dim = d_model // q_heads
        self.reduce = nn.Sequential(Rearrange("batch time feat -> batch feat time"), nn.Conv1d(d_model, group_dim, 1, bias=False),
                                    nn.BatchNorm1d(group_dim), nn.SiLU())
        self.add_module("reduce", self.reduce)
        self.tcn = _TCN(tcn_depth, tcn_kernel_length, d_fused, ng + 1, drop_prob_tcn, activation)
        self.add_module("tcn", self.tcn)
        self.final_layer = _ClassificationHead(d_fused, ng + 1, self.n_outputs, classifier_max_norm)
        self.add_module("final_layer", self.final_layer)
        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        cf = self.conv_block(x)
        tokens = self.to_tokens(self.mix(cf))
        L = tokens.shape[1]
        theta = rotary_frequencies(self.head_dim)
        emb = rotary_angles(torch.arange(L, dtype=torch.float32), theta, False)
        cos = emb.cos().to(tokens.dtype)
        sin = emb.sin().to(tokens.dtype)
        for b in self._blocks:
            tokens = b._fwd(tokens, cos, sin)
        fused = torch.cat([cf, self.reduce(tokens)], dim=1)
        dec = self.tcn(fused)
        return self.final_layer(dec[:, :, -1:])

    def _get_name(self) -> str:
        return "TCFormer"
