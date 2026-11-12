# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# EEGCLIP with its default Deep4Net EEG encoder. Custom eeg/text encoders are
# not supported in Typhon: encode_text takes precomputed text features
# (batch, text_embedding_dim) or (batch, seq, text_embedding_dim).

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.models.deep4 import Deep4Net
from braindecode.modules import AvgPool2dWithConv

def to_dense_prediction_model(model: nn.Module, axis = (2, 3)):
    """braindecode EEGModuleMixin.to_dense_prediction_model: replace strides
    by dilations (Conv2d, MaxPool2d, AvgPool2dWithConv), in module order."""
    axes = [int(a) - 2 for a in axis]
    so_far = [1, 1]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            c = cast(nn.Conv2d, m)
            if c.dilation != [1, 1]:
                raise AssertionError("Dilation should equal 1 before conversion, maybe the model is already converted?")
            nd = [1, 1]
            for a in axes:
                nd[a] = so_far[a]
            c.dilation = nd
            so_far = [so_far[0] * c.stride[0], so_far[1] * c.stride[1]]
            ns = list(c.stride)
            for a in axes:
                ns[a] = 1
            c.stride = ns
        elif isinstance(m, nn.MaxPool2d):
            p = cast(nn.MaxPool2d, m)
            nd = [1, 1]
            for a in axes:
                nd[a] = so_far[a]
            p.dilation = nd
            so_far = [so_far[0] * p.stride[0], so_far[1] * p.stride[1]]
            ns = list(p.stride)
            for a in axes:
                ns[a] = 1
            p.stride = ns
        elif isinstance(m, AvgPool2dWithConv):
            raise NotImplementedError("to_dense_prediction_model: AvgPool2dWithConv is not supported in Typhon")

def _projection_head(in_dim: int, out_dim: int, n_layers: int, activation, drop_prob: float) -> nn.Sequential:
    s = nn.Sequential()
    d = in_dim
    for _ in range(n_layers - 1):
        s.append(nn.Linear(d, out_dim))
        s.append(nn.BatchNorm1d(out_dim))
        s.append(activation())
        s.append(nn.Dropout(drop_prob))
        d = out_dim
    s.append(nn.Linear(d, out_dim))
    return s

class EEGCLIP(EEGModuleMixin, nn.Module):
    """EEG-CLIP from Ndir et al (2025)."""
    text_pooling: str
    text_embedding_dim: int
    eeg_embedding_dim: int
    projection_layers: int
    drop_prob: float
    eeg_encoder: Deep4Net
    text_projection: nn.Sequential
    final_layer: nn.Sequential
    logit_scale: Tensor

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None, n_times: Optional[int] = None,
                 text_encoder = None, text_embedding_dim: int = 768, eeg_encoder = None, eeg_embedding_dim: int = 128,
                 text_pooling: str = "cls", projection_layers: int = 3, activation = nn.ReLU, drop_prob: float = 0.1,
                 initial_temperature: float = 0.07, chs_info: Optional[List[Dict[str, str]]] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if text_pooling != "cls" and text_pooling != "mean":
            raise ValueError("text_pooling must be 'cls' or 'mean'.")
        if text_encoder is not None or eeg_encoder is not None:
            raise NotImplementedError("EEGCLIP: custom encoders are not supported in Typhon")
        self.text_pooling = text_pooling
        self.text_embedding_dim = text_embedding_dim
        self.eeg_embedding_dim = eeg_embedding_dim
        self.projection_layers = projection_layers
        self.drop_prob = drop_prob
        enc = Deep4Net(n_chans=self.n_chans, n_times=self.n_times, n_outputs=eeg_embedding_dim, final_conv_length=2,
                       stride_before_pool=True)
        to_dense_prediction_model(enc)
        fl = cast(nn.Sequential, enc.get_submodule("final_layer"))
        fl.add_module("log_softmax", nn.LogSoftmax(dim=1))
        self.eeg_encoder = enc
        self.add_module("eeg_encoder", self.eeg_encoder)
        self.text_projection = _projection_head(text_embedding_dim, self.n_outputs, projection_layers, activation, drop_prob)
        self.add_module("text_projection", self.text_projection)
        self.final_layer = _projection_head(eeg_embedding_dim, self.n_outputs, projection_layers, activation, drop_prob)
        self.add_module("final_layer", self.final_layer)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / initial_temperature)))
        self.register_parameter("logit_scale", self.logit_scale)

    def _bn_guard(self, x: Tensor, f) -> Tensor:
        # _disable_batch_norm_training_if_batch_size_one
        if self.training and x.shape[0] == 1:
            bns = [m for m in self.modules() if "BatchNorm" in m._get_name() and m.training]
            for m in bns:
                m.train(False)
            y = f(x)
            for m in bns:
                m.train(True)
            return y
        return f(x)

    def _forward(self, X: Tensor) -> Tensor:
        features = self.eeg_encoder(X)
        if features.ndim == 2:
            features = features.unsqueeze(-1)
        b = features.shape[0]
        per_step = features.transpose(1, 2).reshape(-1, features.shape[1])
        projected = self.final_layer(per_step)
        return projected.reshape(b, features.shape[-1], projected.shape[-1]).mean(dim=1)

    def forward(self, X: Tensor) -> Tensor:
        return self._bn_guard(X, self._forward)

    def _text(self, features: Tensor) -> Tensor:
        if features.ndim == 3:
            if self.text_pooling == "cls":
                features = features[:, 0]
            else:
                features = features.mean(dim=1)
        return self.text_projection(features)

    def encode_text(self, text_inputs: Tensor) -> Tensor:
        return self._bn_guard(text_inputs, self._text)

    def compute_logits(self, eeg_embeds: Tensor, text_embeds: Tensor) -> Tuple[Tensor, Tensor]:
        l = self.logit_scale * torch.matmul(eeg_embeds, text_embeds.T)
        return l, l.T

    def contrastive_loss(self, eeg_embeds: Tensor, text_embeds: Tensor) -> Tensor:
        le, lt = self.compute_logits(eeg_embeds, text_embeds)
        labels = torch.arange(eeg_embeds.shape[0])
        return (F.cross_entropy(le, labels) + F.cross_entropy(lt, labels)) / 2

    def _get_name(self) -> str:
        return "EEGCLIP"
