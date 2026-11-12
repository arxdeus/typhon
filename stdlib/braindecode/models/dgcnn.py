# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# chs_info entries are Dict[str, str] in Codon: the 3-D electrode position is
# read from "loc" given as whitespace separated floats (first 3 used, like
# MNE's 12-element loc). Standard-montage lookup by channel name is not
# available; channels without a usable "loc" raise ValueError.

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin

class _GraphConvolution(nn.Module):
    cheb_order: int
    projection: nn.Linear
    def __init__(self, in_features: int, out_features: int, cheb_order: int):
        super().__init__()
        self.cheb_order = cheb_order
        self.projection = nn.Linear(in_features * cheb_order, out_features, bias=False)
        self.add_module("projection", self.projection)
    def forward(self, x: Tensor, laplacian: Tensor) -> Tensor:
        comps = [x]
        if self.cheb_order > 1:
            comps.append(torch.matmul(laplacian, x))
        for _ in range(2, self.cheb_order):
            comps.append(2 * torch.matmul(laplacian, comps[-1]) - comps[-2])
        return self.projection(torch.cat(comps, dim=-1))
    def _get_name(self) -> str:
        return "_GraphConvolution"

def _chs_locations(chs_info: List[Dict[str, str]], n_chans: int) -> Optional[List[List[float]]]:
    """extract_channel_locations_from_chs_info (float32 coordinates)."""
    locs = List[List[float]]()
    for i in range(n_chans):
        if i >= len(chs_info) or "loc" not in chs_info[i]:
            break
        parts = chs_info[i]["loc"].split()
        if len(parts) < 3:
            break
        locs.append([float(float32(float(parts[k]))) for k in range(3)])
    if len(locs) == 0:
        return None
    allzero = True
    for r in locs:
        for v in r:
            if abs(v) > 1e-8:
                allzero = False
    return None if allzero else locs

def _median(v: List[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])

def _build_initial_adjacency(chs_info: List[Dict[str, str]], n_chans: int, n_neighbors: int = 5) -> List[float]:
    pos = _chs_locations(chs_info, n_chans)
    ok = pos is not None
    if ok:
        p = pos.__val__()
        if len(p) != n_chans:
            ok = False
        for r in p:
            for v in r:
                if math.isnan(v) or math.isinf(v):
                    ok = False
    if not ok:
        raise ValueError("Could not extract valid 3-D channel locations from chs_info. Please ensure that each "
                         "entry in chs_info contains a 'loc' key with 3-D positions.")
    if n_chans < 2:
        raise ValueError("DGCNN needs at least 2 channels to build its channel graph.")
    p = pos.__val__()
    k = min(n_neighbors, n_chans - 1)
    n = n_chans
    d = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for c in range(3):
                t = p[i][c] - p[j][c]
                s += t * t
            d[i * n + j] = math.sqrt(s) if i != j else 0.0
    knn_d = List[float]()
    conn = [0.0] * (n * n)
    for i in range(n):
        row = sorted([d[i * n + j] for j in range(n)])
        for q in range(1, k + 1):
            knn_d.append(row[q])
        # kneighbors_graph (include_self=False): k nearest other points,
        # ties broken by index (stable order)
        order = sorted([j for j in range(n) if j != i], key=lambda j: (d[i * n + j], j))
        for q in range(k):
            conn[i * n + order[q]] = 1.0
    sigma = _median(knn_d) + 1e-08
    adj = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            adj[i * n + j] = conn[i * n + j] * math.exp(-d[i * n + j] ** 2 / (2 * sigma ** 2))
    out = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            out[i * n + j] = float(float32(max(adj[i * n + j], adj[j * n + i])))
    return out

class _LearnableAdjacency(nn.Module):
    n_chans: int
    adjacency: Tensor
    bias: Tensor
    def __init__(self, n_chans: int, chs_info: List[Dict[str, str]], n_neighbors: int = 5):
        super().__init__()
        self.n_chans = n_chans
        init_adj = torch.tensor(_build_initial_adjacency(chs_info, n_chans, n_neighbors)).reshape(n_chans, n_chans)
        self.adjacency = nn.Parameter(init_adj + 0.1 * torch.randn(n_chans, n_chans))
        self.register_parameter("adjacency", self.adjacency)
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.register_parameter("bias", self.bias)
    def laplacian(self) -> Tensor:
        adjacency = F.relu(self.adjacency + self.bias)
        degree = adjacency.sum(dim=1)
        dis = torch.diag(1.0 / (torch.sqrt(degree) + 1e-05))
        identity = torch.eye(self.n_chans, dtype=adjacency.dtype)
        return identity - torch.matmul(torch.matmul(dis, adjacency), dis)
    def forward(self, x: Tensor) -> Tensor:
        return self.laplacian()
    def _get_name(self) -> str:
        return "_LearnableAdjacency"

class DGCNN(EEGModuleMixin, nn.Module):
    """DGCNN from Song et al (2018)."""
    activation: nn.Module
    drop_prob: float
    learnable_adjacency: _LearnableAdjacency
    graph_conv: _GraphConvolution
    graph_conv_bias: Tensor
    classifier: nn.Sequential
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, n_filters: int = 64,
                 cheb_order: int = 2, n_neighbors: int = 5, mlp_dims = (256,), activation = nn.ReLU,
                 drop_prob: float = 0.5):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.drop_prob = drop_prob
        self.learnable_adjacency = _LearnableAdjacency(self.n_chans, self.chs_info, n_neighbors)
        self.add_module("learnable_adjacency", self.learnable_adjacency)
        self.graph_conv = _GraphConvolution(self.n_times, n_filters, cheb_order)
        self.add_module("graph_conv", self.graph_conv)
        self.graph_conv_bias = nn.Parameter(torch.zeros(1, 1, n_filters))
        self.register_parameter("graph_conv_bias", self.graph_conv_bias)
        d = self.n_chans * n_filters
        self.classifier = nn.Sequential()
        for h in mlp_dims:
            self.classifier.append(nn.Linear(d, h))
            self.classifier.append(activation())
            self.classifier.append(nn.Dropout(p=drop_prob))
            d = h
        self.add_module("classifier", self.classifier)
        self.final_layer = nn.Linear(d, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        lap = self.learnable_adjacency.laplacian()
        x = self.activation(self.graph_conv.forward(x, lap) + self.graph_conv_bias)
        x = x.reshape(x.size(0), -1)
        return self.final_layer(self.classifier(x))

    def _get_name(self) -> str:
        return "DGCNN"
