# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# GCNsNet with lazily materialized node biases and final layer, like
# braindecode (materialized on the first forward). The graph (Graclus-style
# coarsening + normalized Laplacians) is built from `adjacency` at
# construction, or from the channel correlation of the first input.

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from braindecode.models.base import EEGModuleMixin
from braindecode.models.dgcnn import _GraphConvolution

class _NodeBias(nn.Module):
    bias: Tensor
    initialized: bool
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(0))
        self.register_parameter("bias", self.bias)
        self.initialized = False
    def forward(self, x: Tensor) -> Tensor:
        if not self.initialized:
            b = nn.Parameter(torch.full([1, x.shape[1], x.shape[2]], 0.1))
            self.bias = b
            self.register_parameter("bias", b)
            self.initialized = True
        return x + self.bias
    def _get_name(self) -> str:
        return "_NodeBias"

class _LazyLinear(nn.Module):
    out_features: int
    weight: Tensor
    bias: Tensor
    initialized: bool
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(0))
        self.register_parameter("weight", self.weight)
        self.bias = nn.Parameter(torch.zeros(0))
        self.register_parameter("bias", self.bias)
        self.initialized = False
    def forward(self, x: Tensor) -> Tensor:
        if not self.initialized:
            fin = x.shape[-1]
            w = nn.Parameter(torch.empty(self.out_features, fin))
            b = nn.Parameter(torch.empty(self.out_features))
            with torch.no_grad():
                init.kaiming_uniform_(w, a=math.sqrt(5))
                bound = 1 / math.sqrt(fin) if fin > 0 else 0.0
                init.uniform_(b, -bound, bound)
            self.weight = w
            self.bias = b
            self.register_parameter("weight", w)
            self.register_parameter("bias", b)
            self.initialized = True
        return F.linear(x, self.weight, self.bias)
    def _get_name(self) -> str:
        return "LazyLinear"

class _LaplacianModule(nn.Module):
    matrix: Optional[Tensor]
    def __init__(self):
        super().__init__()
        self.matrix = None
        self.register_buffer("matrix", None)
    def set(self, m: Tensor):
        self.matrix = m
        self.register_buffer("matrix", m)
    def _get_name(self) -> str:
        return "_LaplacianModule"

# ------------------------------------------------------------ graph coarsening

def _cluster_one_level(a: List[List[float]], node_order: List[int], weights: List[float]) -> List[int]:
    n = len(a)
    marked = [False] * n
    clusters = [0] * n
    cid = 0
    for node in node_order:
        if marked[node]:
            continue
        marked[node] = True
        clusters[node] = cid
        nb = [j for j in range(n) if a[node][j] != 0.0 and not marked[j]]
        if len(nb) > 0:
            best = 0
            bs = -math.inf
            for q in range(len(nb)):
                j = nb[q]
                s = float(float32(a[node][j] * float(float32(1.0 / weights[node] + 1.0 / weights[j]))))
                if s > bs:
                    bs = s
                    best = q
            if bs > 0:
                clusters[nb[best]] = cid
                marked[nb[best]] = True
        cid += 1
    return clusters

def _compute_perm(parents: List[List[int]]) -> List[List[int]]:
    indices = List[List[int]]()
    if len(parents) > 0:
        indices.append(list(range(max(parents[-1]) + 1)))
    for pi in range(len(parents) - 1, -1, -1):
        parent = parents[pi]
        n_nodes = len(parent)
        layer = List[int]()
        for i in indices[-1]:
            children = [c for c in range(len(parent)) if parent[c] == i]
            n_fake = 2 - len(children)
            layer.extend(children)
            layer.extend(list(range(n_nodes, n_nodes + n_fake)))
            n_nodes += n_fake
        indices.append(layer)
    indices.reverse()
    return indices

def _coarsen(adjacency: Tensor, levels: int) -> Tuple[List[List[List[float]]], Optional[List[int]]]:
    n0 = adjacency.shape[0]
    flat = adjacency.detach().double().reshape(-1).tolist()
    a = [[flat[i * n0 + j] for j in range(n0)] for i in range(n0)]
    adjs = [a]
    clusters = List[List[int]]()
    node_order = [int(v) for v in torch.randperm(n0).tolist()]
    degree = [float(float32(sum(a[i][j] for i in range(n0)) - a[j][j])) for j in range(n0)]
    for _ in range(levels):
        cl = _cluster_one_level(a, node_order, degree)
        nc = max(cl) + 1
        n = len(a)
        b = [[0.0] * nc for _ in range(nc)]
        for i in range(n):
            for j in range(n):
                if a[i][j] != 0.0:
                    b[cl[i]][cl[j]] += a[i][j]
        a = [[float(float32(v)) for v in r] for r in b]
        clusters.append(cl)
        adjs.append(a)
        degree = [float(float32(sum(a[i][j] for i in range(nc)))) for j in range(nc)]
        node_order = [int(v) for v in torch.argsort(torch.tensor(degree)).tolist()]
    perms = _compute_perm(clusters)
    for i in range(len(adjs)):
        m = adjs[i]
        n = len(m)
        for k in range(n):
            m[k][k] = 0.0
        if i < levels:
            p = perms[i]
            np_ = len(p)
            adjs[i] = [[(m[p[r]][p[c]] if p[r] < n and p[c] < n else 0.0) for c in range(np_)] for r in range(np_)]
    pr: Optional[List[int]] = None
    if levels > 0:
        pr = perms[0]
    return adjs, pr

def _normed_laplacian_minus_eye(m: List[List[float]]) -> Tensor:
    """scipy.sparse.csgraph.laplacian(normed=True) - I."""
    n = len(m)
    w = [0.0] * n
    for j in range(n):
        s = 0.0
        for i in range(n):
            if i != j:
                s += m[i][j]
        w[j] = 1.0 if s == 0.0 else math.sqrt(s)
    vals = List[float](capacity=n * n)
    for i in range(n):
        for j in range(n):
            if i == j:
                iso = True
                for k in range(n):
                    if k != j and m[k][j] != 0.0:
                        iso = False
                vals.append((0.0 if iso else 1.0) - 1.0)
            else:
                vals.append(-m[i][j] / w[j] / w[i])
    return torch.tensor(vals).reshape(n, n)

class GCNsNet(EEGModuleMixin, nn.Module):
    """GCNs-Net from Lun et al (2023)."""
    n_features: List[int]
    cheb_orders: List[int]
    pool_sizes: List[int]
    graph_initialized: bool
    permutation: Optional[Tensor]
    laplacians: nn.ModuleList
    _laps: List[_LaplacianModule]
    graph_convs: nn.ModuleList
    _gcs: List[_GraphConvolution]
    biases: nn.ModuleList
    _bs: List[_NodeBias]
    batch_norms: nn.ModuleList
    _bns: List[nn.BatchNorm1d]
    activation: nn.Module
    dropout: nn.Dropout
    final_layer: _LazyLinear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 adjacency: Optional[Tensor] = None, n_features = (16, 32, 64, 128, 256, 512),
                 cheb_orders = (2, 2, 2, 2, 2, 2), pool_sizes = (2, 2, 2, 2, 2, 2), activation = nn.Softplus,
                 drop_prob: float = 0.0):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        nf = [int(v) for v in n_features]
        co = [int(v) for v in cheb_orders]
        ps = [int(v) for v in pool_sizes]
        if not (len(nf) == len(co) and len(co) == len(ps) and len(ps) != 0):
            raise ValueError("n_features, cheb_orders and pool_sizes must have the same, non-zero length (one value per layer).")
        for p in ps:
            if p < 1 or (p & (p - 1)) != 0:
                raise ValueError(f"pool_sizes must be 1 or powers of 2, got {pool_sizes}.")
        self.n_features = nf
        self.cheb_orders = co
        self.pool_sizes = ps
        self.graph_initialized = False
        self.permutation = None
        self.register_buffer("permutation", None)
        self._laps = [_LaplacianModule() for _ in nf]
        self.laplacians = nn.ModuleList()
        for l in self._laps:
            self.laplacians.append(l)
        self.add_module("laplacians", self.laplacians)
        if adjacency is not None:
            self._initialize_graph(adjacency.float())
        fin = [1] + nf[:-1]
        self._gcs = [_GraphConvolution(fin[i], nf[i], co[i]) for i in range(len(nf))]
        self.graph_convs = nn.ModuleList()
        for g in self._gcs:
            self.graph_convs.append(g)
        self.add_module("graph_convs", self.graph_convs)
        self._bs = [_NodeBias() for _ in nf]
        self.biases = nn.ModuleList()
        for b in self._bs:
            self.biases.append(b)
        self.add_module("biases", self.biases)
        self._bns = [nn.BatchNorm1d(f) for f in nf]
        self.batch_norms = nn.ModuleList()
        for b in self._bns:
            self.batch_norms.append(b)
        self.add_module("batch_norms", self.batch_norms)
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.final_layer = _LazyLinear(self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    @staticmethod
    def _adjacency(x: Tensor) -> Tensor:
        a = torch.corrcoef(x.transpose(0, 1).flatten(1)).abs()
        n = a.shape[0]
        return a * (1.0 - torch.eye(n))

    def _levels(self) -> List[int]:
        levels = [0]
        for p in self.pool_sizes[:-1]:
            levels.append(levels[-1] + int(round(math.log2(p))))
        return levels

    def _initialize_graph(self, adjacency: Tensor):
        levels = self._levels()
        n_levels = levels[-1]
        adjs, perm = _coarsen(adjacency, n_levels)
        if n_levels > 0:
            pt = torch.tensor(perm.__val__(), dtype=torch.int64)
            self.permutation = pt
            self.register_buffer("permutation", pt)
        else:
            self.permutation = None
            self.register_buffer("permutation", None)
        for i in range(len(self._laps)):
            self._laps[i].set(_normed_laplacian_minus_eye(adjs[levels[i]]))
        self.graph_initialized = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.graph_initialized:
            with torch.no_grad():
                self._initialize_graph(GCNsNet._adjacency(x.detach().float()))
        batch_size, n_chans, n_times = x.shape[0], x.shape[1], x.shape[2]
        x = x.transpose(1, 2).reshape(batch_size * n_times, n_chans)
        if self.permutation is not None:
            perm = self.permutation.__val__()
            x = F.pad(x, [0, perm.numel() - n_chans]).index_select(1, perm)
        x = x.unsqueeze(-1)
        for i in range(len(self._gcs)):
            matrix = self._laps[i].matrix.__val__()
            x = self._bs[i](self._gcs[i].forward(x, matrix))
            x = self.activation(self._bns[i](x.transpose(1, 2)))
            if self.pool_sizes[i] > 1:
                x = F.max_pool1d(x, self.pool_sizes[i], ceil_mode=True)
            x = x.transpose(1, 2)
        x = self.final_layer(self.dropout(x.flatten(1)))
        return x.reshape(batch_size, n_times, self.n_outputs).mean(dim=1)

    def _get_name(self) -> str:
        return "GCNsNet"
