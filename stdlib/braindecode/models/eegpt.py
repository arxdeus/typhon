# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional.initialization import _param_by_name
from braindecode.modules import Conv1dWithConstraint, LinearWithConstraint
from braindecode.modules.timm import TimmAttention, TimmMlp, PreNormBlock

EEGPT_CHANNELS = ['FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6',
                  'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
                  'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5',
                  'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1',
                  'OZ', 'O2']
EEGPT_19_CHANNELS = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ',
                     'P4', 'P8', 'O1', 'O2']

def _channel_index(name: str) -> int:
    for i in range(len(EEGPT_CHANNELS)):
        if EEGPT_CHANNELS[i] == name:
            return i
    return -1

class _LinearConstraintProbe(nn.Module):
    probe1: LinearWithConstraint
    dropout: nn.Dropout
    probe2: LinearWithConstraint

    def __init__(self, n_patches: int, embed_num: int, embed_dim: int, n_outputs: int, probe_hidden_dim: int = 16,
                 dropout_p: float = 0.5, probe1_max_norm: float = 1.0, probe2_max_norm: float = 0.25):
        super().__init__()
        self.probe1 = LinearWithConstraint(embed_num * embed_dim, probe_hidden_dim, max_norm=probe1_max_norm)
        self.add_module("probe1", self.probe1)
        self.dropout = nn.Dropout(dropout_p)
        self.add_module("dropout", self.dropout)
        self.probe2 = LinearWithConstraint(n_patches * probe_hidden_dim, n_outputs, max_norm=probe2_max_norm)
        self.add_module("probe2", self.probe2)

    def forward(self, z: Tensor) -> Tensor:
        h = self.probe1(self.dropout(z.flatten(2)))
        return self.probe2(h.flatten(1))

    def _get_name(self) -> str:
        return "_LinearConstraintProbe"

class _ChannelProjection(nn.Module):
    proj_type: str
    proj: nn.Module

    def __init__(self, in_channels: int, out_channels: int, proj_type: str = "none", max_norm: float = 1.0):
        super().__init__()
        self.proj_type = proj_type
        p: nn.Module = nn.Identity()
        if proj_type == "none":
            if in_channels != out_channels:
                raise ValueError(f"proj_type='none' requires in_channels == out_channels, got {in_channels} != {out_channels}.")
        elif proj_type == "conv1d_constraint":
            p = Conv1dWithConstraint(in_channels, out_channels, kernel_size=1, max_norm=max_norm)
        elif proj_type == "linear":
            p = nn.Linear(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")
        self.proj = p
        self.add_module("proj", p)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj_type == "linear":
            return self.proj(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(x)

    def _get_name(self) -> str:
        return "_ChannelProjection"

class _EEGPTPatchEmbed(nn.Module):
    n_chans: int
    n_times: int
    patch_size: int
    stride: int
    padding_size: int
    n_times_padded: int
    num_patches: Tuple[int, int]
    proj: nn.Conv2d

    def __init__(self, n_chans: int = 64, n_times: int = 1000, patch_size: int = 16, patch_stride: Optional[int] = None,
                 embed_dim: int = 768):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.patch_size = patch_size
        if n_times < patch_size:
            raise ValueError(f"n_times ({n_times}) must be >= patch_size ({patch_size}).")
        if patch_stride is None:
            r = n_times % patch_size
            self.padding_size = patch_size - r if r != 0 else 0
            self.stride = patch_size
        else:
            s = patch_stride.__val__()
            r = (n_times - patch_size) % s
            self.padding_size = s - r if r != 0 else 0
            self.stride = s
        self.n_times_padded = n_times + self.padding_size
        if self.padding_size > 0:
            self.add_module("padding_layer", nn.ConstantPad1d((0, self.padding_size), 0.0))
        else:
            self.add_module("padding_layer", nn.Identity())
        self.num_patches = (n_chans, (self.n_times_padded - patch_size) // self.stride + 1)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1, patch_size), stride=(1, self.stride))
        self.add_module("proj", self.proj)

    def forward(self, x: Tensor) -> Tensor:
        if self.padding_size > 0:
            x = F.pad(x, [0, self.padding_size])
        return self.proj(x.unsqueeze(1)).permute(0, 3, 2, 1)

    def _get_name(self) -> str:
        return "_PatchEmbed"

def _eegpt_init(std: float):
    def f(m):
        n = m._get_name()
        if n in ("Linear", "Conv2d"):
            w = _param_by_name(m, "weight")
            if w is not None:
                nn.init.trunc_normal_(w.__val__(), std=std)
            b = _param_by_name(m, "bias")
            if b is not None:
                nn.init.zeros_(b.__val__())
        elif n == "Embedding":
            nn.init.normal_(_param_by_name(m, "weight").__val__(), mean=0.0, std=0.02)
        elif n == "LayerNorm":
            w = _param_by_name(m, "weight")
            if w is not None:
                nn.init.ones_(w.__val__())
            b = _param_by_name(m, "bias")
            if b is not None:
                nn.init.zeros_(b.__val__())
    return f

class _EEGTransformer(nn.Module):
    embed_dim: int
    embed_num: int
    num_heads: int
    summary_token: Tensor
    patch_embed: _EEGPTPatchEmbed
    num_patches: Tuple[int, int]
    chan_embed: nn.Embedding
    blocks: nn.ModuleList
    _blocks: List[PreNormBlock[TimmAttention, TimmMlp]]
    norm: nn.LayerNorm
    init_std: float

    def __init__(self, n_chans: int = 64, n_times: int = 1000, patch_size: int = 64, patch_stride: Optional[int] = None,
                 embed_dim: int = 768, embed_num: int = 1, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.0, attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0,
                 norm_eps: float = 1e-6, init_std: float = 0.02):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_num = embed_num
        self.num_heads = num_heads
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
        self.register_parameter("summary_token", self.summary_token)
        self.patch_embed = _EEGPTPatchEmbed(n_chans, n_times, patch_size, patch_stride, embed_dim)
        self.add_module("patch_embed", self.patch_embed)
        self.num_patches = self.patch_embed.num_patches
        self.chan_embed = nn.Embedding(max(len(EEGPT_CHANNELS), n_chans), embed_dim)
        self.add_module("chan_embed", self.chan_embed)
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self._blocks = [PreNormBlock(embed_dim, TimmAttention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                              attn_drop=attn_drop_rate, proj_drop=drop_rate),
                                     TimmMlp(embed_dim, int(float(embed_dim) * mlp_ratio), drop=drop_rate),
                                     norm_eps=norm_eps, drop_path=float(dpr[i])) for i in range(depth)]
        self.blocks = nn.ModuleList()
        for b in self._blocks:
            self.blocks.append(b)
        self.add_module("blocks", self.blocks)
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.add_module("norm", self.norm)
        self.init_std = init_std
        nn.init.trunc_normal_(self.summary_token, std=init_std)
        self.apply(_eegpt_init(init_std))
        with torch.no_grad():
            for i in range(len(self._blocks)):
                s = torch.sqrt(torch.tensor(2.0 * float(i + 1)))
                self._blocks[i].attn.proj.weight.div_(s)
                self._blocks[i].mlp.fc2.weight.div_(s)

    def prepare_chan_ids(self, channels: Optional[List[str]]) -> Tensor:
        if channels is None:
            return torch.arange(self.patch_embed.n_chans)
        ids = List[int]()
        chs = channels.__val__()
        for i in range(len(chs)):
            c = chs[i].upper().strip(".")
            j = _channel_index(c)
            ids.append(j if j >= 0 else i)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def forward_ids(self, x: Tensor, chan_ids: Tensor) -> Tensor:
        x = self.patch_embed(x)
        B, NP, NC = x.shape[0], x.shape[1], x.shape[2]
        if NP != self.num_patches[1] or NC != self.num_patches[0]:
            raise ValueError(f"Patch grid ({NC}, {NP}) does not match the configured {self.num_patches}.")
        x = x + F.embedding(chan_ids.long(), self.chan_embed.weight).unsqueeze(0)
        x = x.flatten(0, 1)
        st = self.summary_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([x, st], dim=1)
        for b in self._blocks:
            x = b(x)
        x = self.norm(x[:, -st.shape[1]:, :])
        x = x.reshape(B, -1, x.shape[1] * x.shape[2])
        return x.unflatten(2, [self.embed_num, -1])

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_ids(x, torch.arange(x.shape[1]))

    def _get_name(self) -> str:
        return "_EEGTransformer"

class EEGPT(EEGModuleMixin, nn.Module):
    """EEGPT from Wang et al (2024). Architecture only (no Hub weights).

    ``final_layer`` (custom head class) and ``patch_module`` are not
    configurable; ``norm_layer`` is LayerNorm(eps=layer_norm_eps)."""
    return_encoder_output: bool
    patch_size: int
    patch_stride: int
    embed_num: int
    embed_dim: int
    depth: int
    num_heads: int
    chan_proj_type: str
    n_chans_target: int
    chan_proj: nn.Module
    target_encoder: _EEGTransformer
    chans_id: Tensor
    flattened_encoder_output_dim: int
    final_layer: nn.Module

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, patch_size: int = 64,
                 patch_stride: int = 32, embed_num: int = 4, embed_dim: int = 512, depth: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, drop_prob: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0, init_std: float = 0.02, qkv_bias: bool = True,
                 layer_norm_eps: float = 1e-6, return_encoder_output: bool = False,
                 chan_proj_type: str = "conv1d_constraint", n_chans_target: int = 19, chan_conv_max_norm: float = 1.0):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.return_encoder_output = return_encoder_output
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.chan_proj_type = chan_proj_type
        self.n_chans_target = n_chans_target
        enc_ch = self.n_chans
        cp: nn.Module = nn.Identity()
        if chan_proj_type != "none":
            cp = _ChannelProjection(self.n_chans, n_chans_target, chan_proj_type, chan_conv_max_norm)
            enc_ch = n_chans_target
        self.chan_proj = cp
        self.add_module("chan_proj", cp)
        self.target_encoder = _EEGTransformer(enc_ch, self.n_times, patch_size, patch_stride, embed_dim, embed_num,
                                              depth, num_heads, mlp_ratio, qkv_bias, drop_prob, attn_drop_rate,
                                              drop_path_rate, layer_norm_eps, init_std)
        self.add_module("target_encoder", self.target_encoder)
        names: Optional[List[str]] = None
        if chan_proj_type != "none":
            names = EEGPT_19_CHANNELS
        elif self._chs_info is not None:
            names = [ch["ch_name"] for ch in self.chs_info]
        self.chans_id = self.target_encoder.prepare_chan_ids(names)
        self.register_buffer("chans_id", self.chans_id, persistent=False)
        np_ = self.target_encoder.num_patches[1]
        self.flattened_encoder_output_dim = np_ * embed_num * embed_dim
        fl: nn.Module = nn.Identity()
        if not return_encoder_output:
            fl = _LinearConstraintProbe(np_, embed_num, embed_dim, self.n_outputs)
        self.final_layer = fl
        self.add_module("final_layer", fl)

    @property
    def n_patches(self) -> int:
        return self.target_encoder.num_patches[1]

    def forward_features(self, x: Tensor) -> Tensor:
        return self.target_encoder.forward_ids(self.chan_proj(x), self.chans_id)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x))

    def _get_name(self) -> str:
        return "EEGPT"
