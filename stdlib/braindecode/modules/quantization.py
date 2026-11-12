# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class EMACodebook(nn.Module):
    """EMA-updated vector-quantization codebook (single process).

    k-means initialisation (``kmeans_init=True``) is not ported; the codebook
    then starts from kaiming-uniform values as in braindecode's non-k-means
    path. Dead-code replacement and EMA updates run in training mode."""
    decay: float
    epsilon: float
    threshold_ema_dead_code: int
    codebook_size: int
    inited: Tensor
    cluster_size: Tensor
    embed: Tensor
    embed_avg: Tensor

    def __init__(self, dim: int, codebook_size: int, decay: float = 0.99, epsilon: float = 1e-6,
                 threshold_ema_dead_code: int = 2, kmeans_init: bool = True, kmeans_iters: int = 10):
        super().__init__()
        if dim <= 0 or codebook_size <= 0:
            raise ValueError("dim and codebook_size must be positive.")
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must satisfy 0 <= decay < 1, got {decay}.")
        if kmeans_init:
            raise NotImplementedError("EMACodebook: kmeans_init=True is not ported")
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.codebook_size = codebook_size
        e = torch.zeros(codebook_size, dim)
        nn.init.kaiming_uniform_(e)
        self.inited = torch.tensor([1.0])
        self.register_buffer("inited", self.inited)
        self.cluster_size = torch.zeros(codebook_size)
        self.register_buffer("cluster_size", self.cluster_size)
        self.embed = e
        self.register_buffer("embed", e)
        self.embed_avg = e.clone()
        self.register_buffer("embed_avg", self.embed_avg)

    def quantize(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            et = self.embed.t()
            d = x.pow(2.0).sum(1, keepdim=True) - 2.0 * torch.matmul(x, et) + et.pow(2.0).sum(0, keepdim=True)
            return d.argmin(dim=-1)

    def dequantize(self, ind: Tensor) -> Tensor:
        return F.embedding(ind, self.embed)

    def apply_codebook(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        shape = list(x.shape)
        xf = x.reshape(-1, shape[-1])
        fi = self.quantize(xf)
        ei = fi.view(shape[:-1])
        q = self.dequantize(ei).to(x.dtype)
        if self.training:
            with torch.no_grad():
                xd = xf.detach()
                if self.threshold_ema_dead_code > 0:
                    exp = self.cluster_size < float(self.threshold_ema_dead_code)
                    if bool(exp.any().item()):
                        n = xd.shape[0]
                        idx = torch.randperm(n)[:self.codebook_size] if n >= self.codebook_size else \
                            torch.randint(0, n, [self.codebook_size])
                        self.embed.copy_(torch.where(exp.unsqueeze(-1), xd.index_select(0, idx), self.embed))
                ones = torch.ones(fi.shape[0], dtype=xd.dtype)
                ohs = torch.zeros(self.codebook_size, dtype=xd.dtype).index_add(0, fi, ones)
                self.cluster_size.mul_(self.decay).add_(ohs * (1.0 - self.decay))
                es = torch.zeros(self.codebook_size, xd.shape[1], dtype=xd.dtype).index_add(0, fi, xd)
                self.embed_avg.mul_(self.decay).add_(es * (1.0 - self.decay))
                tot = self.cluster_size.sum()
                sm = (self.cluster_size + self.epsilon) / (tot + self.epsilon * float(self.codebook_size))
                self.embed.copy_(self.embed_avg / (sm * tot).unsqueeze(1))
        return q, ei

    def forward(self, x: Tensor) -> Tensor:
        q, _ = self.apply_codebook(x)
        return q

    def _get_name(self) -> str:
        return "EMACodebook"

def _rotate_to(src: Tensor, tgt: Tensor) -> Tensor:
    # rotation-trick straight-through estimator (Fifty et al. 2024)
    shp = list(src.shape)
    s = src.reshape(-1, shp[-1])
    t = tgt.reshape(-1, shp[-1])
    ns = torch.clamp(torch.linalg.vector_norm(s, 2, dim=-1, keepdim=True), min=1e-6)
    nt = torch.clamp(torch.linalg.vector_norm(t, 2, dim=-1, keepdim=True), min=1e-6)
    u = s / ns
    q = t / nt
    w = F.normalize(u + q, p=2.0, dim=1, eps=1e-6).detach()
    ew = (s * w).sum(dim=1, keepdim=True)
    eu = (s * u.detach()).sum(dim=1, keepdim=True)
    r = s - 2.0 * ew * w + 2.0 * eu * q.detach()
    return (r * (nt / ns).detach()).reshape(shp)

class KMeansCodebook(nn.Module):
    """braindecode EMACodebook with k-means init on the first forward
    (``kmeans_init=True``), as used by VectorQuantizer."""
    decay: float
    epsilon: float
    threshold_ema_dead_code: int
    codebook_size: int
    kmeans_init: bool
    kmeans_iters: int
    inited: Tensor
    cluster_size: Tensor
    embed: Tensor
    embed_avg: Tensor

    def __init__(self, dim: int, codebook_size: int, decay: float = 0.99, epsilon: float = 1e-6,
                 threshold_ema_dead_code: int = 2, kmeans_init: bool = True, kmeans_iters: int = 10):
        super().__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.codebook_size = codebook_size
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        e = torch.zeros(codebook_size, dim)
        if not kmeans_init:
            nn.init.kaiming_uniform_(e)
        self.inited = torch.tensor([0.0 if kmeans_init else 1.0])
        self.register_buffer("inited", self.inited)
        self.cluster_size = torch.zeros(codebook_size)
        self.register_buffer("cluster_size", self.cluster_size)
        self.embed = e
        self.register_buffer("embed", e)
        self.embed_avg = e.clone()
        self.register_buffer("embed_avg", self.embed_avg)

    def _sample(self, s: Tensor, num: int) -> Tensor:
        n = s.shape[0]
        idx = torch.randperm(n)[:num] if n >= num else torch.randint(0, n, [num])
        return s.index_select(0, idx)

    def _kmeans(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        D = s.shape[1]
        if s.shape[0] < self.codebook_size:
            s = torch.cat([s, torch.randn(self.codebook_size - s.shape[0], D, dtype=s.dtype)], dim=0)
        centers = self._sample(s, self.codebook_size)
        bins = torch.ones(self.codebook_size, dtype=torch.long)
        for _ in range(self.kmeans_iters):
            buckets = torch.cdist(s, centers).argmin(dim=-1)
            bins = torch.bincount(buckets, minlength=self.codebook_size)
            empty = bins == 0
            bins = bins.masked_fill(empty, 1)
            nc = torch.zeros(self.codebook_size, D, dtype=s.dtype).index_add(0, buckets, s)
            nc = nc / bins.unsqueeze(-1).to(s.dtype)
            centers = torch.where(empty.unsqueeze(-1), centers, nc)
        return centers, bins

    def init_embed_(self, data: Tensor):
        if not self.kmeans_init or self.inited.item() != 0.0:
            return
        with torch.no_grad():
            smp = self._sample(data.reshape(-1, data.shape[-1]), 4096)
            e, cs = self._kmeans(smp)
            self.embed.copy_(e)
            self.embed_avg.copy_(e)
            self.cluster_size.copy_(cs.to(self.cluster_size.dtype))
            self.inited.fill_(1.0)

    def apply_codebook(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        shp = list(x.shape)
        xf = x.reshape(-1, shp[-1])
        self.init_embed_(xf)
        with torch.no_grad():
            et = self.embed.t()
            d = xf.pow(2.0).sum(1, keepdim=True) - 2.0 * torch.matmul(xf, et) + et.pow(2.0).sum(0, keepdim=True)
            fi = d.argmin(dim=-1)
        ei = fi.view(shp[:-1])
        q = F.embedding(ei, self.embed).to(x.dtype)
        if self.training:
            with torch.no_grad():
                xd = xf.detach()
                if self.threshold_ema_dead_code > 0:
                    exp = self.cluster_size < float(self.threshold_ema_dead_code)
                    if bool(exp.any().item()):
                        self.embed.copy_(torch.where(exp.unsqueeze(-1), self._sample(xd, self.codebook_size), self.embed))
                ones = torch.ones(fi.shape[0], dtype=xd.dtype)
                ohs = torch.zeros(self.codebook_size, dtype=xd.dtype).index_add(0, fi, ones)
                self.cluster_size.mul_(self.decay).add_(ohs * (1.0 - self.decay))
                es = torch.zeros(self.codebook_size, xd.shape[1], dtype=xd.dtype).index_add(0, fi, xd)
                self.embed_avg.mul_(self.decay).add_(es * (1.0 - self.decay))
                tot = self.cluster_size.sum()
                sm = (self.cluster_size + self.epsilon) / (tot + self.epsilon * float(self.codebook_size))
                self.embed.copy_(self.embed_avg / (sm * tot).unsqueeze(1))
        return q, ei

    def forward(self, x: Tensor) -> Tensor:
        q, _ = self.apply_codebook(x)
        return q

    def _get_name(self) -> str:
        return "EMACodebook"

class VectorQuantizer(nn.Module):
    project_in: nn.Module
    project_out: nn.Module
    _codebook: KMeansCodebook
    rotation_trick: bool
    codebook_size: int
    def __init__(self, dim: int, codebook_size: int, codebook_dim: Optional[int] = None, decay: float = 0.99,
                 epsilon: float = 1e-6, threshold_ema_dead_code: int = 2, rotation_trick: bool = True,
                 kmeans_init: bool = True, kmeans_iters: int = 10):
        super().__init__()
        cd = codebook_dim.__val__() if codebook_dim is not None else dim
        pi: nn.Module = nn.Identity()
        po: nn.Module = nn.Identity()
        if cd != dim:
            pi = nn.Linear(dim, cd)
            po = nn.Linear(cd, dim)
        self.project_in = pi
        self.add_module("project_in", pi)
        self.project_out = po
        self.add_module("project_out", po)
        self._codebook = KMeansCodebook(cd, codebook_size, decay, epsilon, threshold_ema_dead_code, kmeans_init, kmeans_iters)
        self.add_module("_codebook", self._codebook)
        self.rotation_trick = rotation_trick
        self.codebook_size = codebook_size
    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.project_in(x)
        q, ei = self._codebook.apply_codebook(x)
        if self.training:
            if self.rotation_trick:
                q = _rotate_to(x, q)
            else:
                q = x + (q - x).detach()
        loss = F.mse_loss(x, q.detach()) * 0.25
        if not self.training:
            loss = loss.detach()
        return self.project_out(q), ei, loss
    def _get_name(self) -> str:
        return "VectorQuantizer"

class ResidualVectorQuantizer(nn.Module):
    num_quantizers: int
    layers: nn.ModuleList
    _layers: List[VectorQuantizer]
    def __init__(self, dim: int, codebook_dim: int, codebook_size: int, num_quantizers: int, rotation_trick: bool = True):
        super().__init__()
        if dim <= 0 or codebook_dim <= 0 or codebook_size <= 0 or num_quantizers <= 0:
            raise ValueError("dim, codebook_dim, codebook_size and num_quantizers must be positive.")
        self.num_quantizers = num_quantizers
        self._layers = [VectorQuantizer(dim, codebook_size, codebook_dim, rotation_trick=rotation_trick)
                        for _ in range(num_quantizers)]
        self.layers = nn.ModuleList()
        for l in self._layers:
            self.layers.append(l)
        self.add_module("layers", self.layers)
    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = F.normalize(x, p=2.0, dim=-1)
        out = torch.zeros(1, dtype=x.dtype).squeeze(0)
        res = x
        losses = List[Tensor]()
        idx = List[Tensor]()
        for vq in self._layers:
            q, i, l = vq.quantize(res)
            res = res - q.detach()
            out = out + q
            losses.append(l)
            idx.append(i)
        return out, torch.stack(idx, dim=-1), torch.stack(losses, dim=-1).mean()
    def forward(self, x: Tensor) -> Tensor:
        q, _, _ = self.quantize(x)
        return q
    def _get_name(self) -> str:
        return "ResidualVectorQuantizer"
