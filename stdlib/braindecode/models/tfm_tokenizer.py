# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.modules.linear_attention import LinearAttentionTransformer
from braindecode.modules.quantization import EMACodebook

@tuple
class TFMTokenizerOutput:
    reconstruction: Tensor
    token_ids: Tensor
    quantized: Tensor
    embeddings: Tensor
    quantization_loss: Tensor
    target_spectrogram: Tensor

def _conv_block(embed_dim: int, kernel_size: int, stride: int, out_dim: int, activation) -> nn.Sequential:
    g = embed_dim // 4
    return nn.Sequential(nn.Conv1d(1, embed_dim, kernel_size=kernel_size, stride=stride), activation(),
                         nn.GroupNorm(g, embed_dim), nn.Conv1d(embed_dim, embed_dim, kernel_size=1), activation(),
                         nn.GroupNorm(g, embed_dim), nn.Conv1d(embed_dim, out_dim, kernel_size=1), activation(),
                         nn.GroupNorm(g, out_dim))

class TFMTokenizer(EEGModuleMixin, nn.Module):
    """TFM-Tokenizer from Pradeepkumar et al (2025). Architecture only.

    ``forward`` returns the reconstructed spectrogram; ``tokenize`` gives
    the full TFMTokenizerOutput. EMACodebook k-means init is not used by
    this model (kmeans_init=False in braindecode too)."""
    window_size: int
    n_freqs: int
    embed_dim: int
    max_seq_len: int
    commitment_cost: float
    frequency_patch_embedding: nn.Sequential
    frequency_encoder: LinearAttentionTransformer
    frequency_attention: nn.Sequential
    frequency_projection: nn.Conv1d
    temporal_patch_embedding: nn.Sequential
    temporal_encoder: LinearAttentionTransformer
    quantizer: EMACodebook
    decoder: LinearAttentionTransformer
    final_layer: nn.Sequential
    stft_window: Tensor

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: float = 200.0, embed_dim: int = 64,
                 codebook_size: int = 8192, freq_patch_size: int = 5, freq_encoder_depth: int = 2,
                 temporal_encoder_depth: int = 2, decoder_depth: int = 8, max_seq_len: int = 1024,
                 commitment_cost: float = 1.0, activation = nn.GELU, drop_prob: float = 0.2):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        sf = self.sfreq
        if sf <= 0.0 or float(int(sf)) != sf or int(sf) % 2:
            raise ValueError(f"sfreq must be a positive even integer, got {sf}.")
        self.window_size = int(sf)
        self.n_freqs = self.window_size // 2
        fp2 = freq_patch_size * freq_patch_size
        ng = self.n_freqs // fp2
        if self.n_freqs % fp2:
            raise ValueError("sfreq // 2 must be divisible by freq_patch_size squared.")
        if embed_dim % 8 or embed_dim % (2 * ng):
            raise ValueError("embed_dim must be a multiple of 8 and of twice the number of frequency groups.")
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.commitment_cost = commitment_cost
        fw = embed_dim // (2 * ng)
        self.frequency_patch_embedding = _conv_block(embed_dim, freq_patch_size, freq_patch_size, embed_dim, activation)
        self.add_module("frequency_patch_embedding", self.frequency_patch_embedding)
        self.frequency_encoder = LinearAttentionTransformer(embed_dim, freq_encoder_depth, 8, drop_prob)
        self.add_module("frequency_encoder", self.frequency_encoder)
        self.frequency_attention = nn.Sequential(nn.Conv1d(embed_dim, fw, kernel_size=freq_patch_size, stride=freq_patch_size),
                                                 nn.Sigmoid())
        self.add_module("frequency_attention", self.frequency_attention)
        self.frequency_projection = nn.Conv1d(embed_dim, fw, kernel_size=freq_patch_size, stride=freq_patch_size)
        self.add_module("frequency_projection", self.frequency_projection)
        self.temporal_patch_embedding = _conv_block(embed_dim, self.window_size, self.window_size // 2, embed_dim // 2, activation)
        self.add_module("temporal_patch_embedding", self.temporal_patch_embedding)
        self.temporal_encoder = LinearAttentionTransformer(embed_dim, temporal_encoder_depth, 8, drop_prob)
        self.add_module("temporal_encoder", self.temporal_encoder)
        self.quantizer = EMACodebook(embed_dim, codebook_size, epsilon=1e-5, threshold_ema_dead_code=0, kmeans_init=False)
        self.add_module("quantizer", self.quantizer)
        nn.init.uniform_(self.quantizer.embed, -1.0 / float(codebook_size), 1.0 / float(codebook_size))
        with torch.no_grad():
            self.quantizer.embed_avg.copy_(self.quantizer.embed)
        self.decoder = LinearAttentionTransformer(embed_dim, decoder_depth, 8, drop_prob)
        self.add_module("decoder", self.decoder)
        self.final_layer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, self.n_freqs))
        self.add_module("final_layer", self.final_layer)
        self.stft_window = torch.hann_window(self.window_size)
        self.register_buffer("stft_window", self.stft_window, persistent=False)

    def compute_spectrogram(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must be a floating-point (batch, channels, samples) tensor.")
        if x.shape[-1] < self.window_size:
            raise ValueError(f"Need at least {self.window_size} samples, got {x.shape[-1]}.")
        B, C, T = x.shape[0], x.shape[1], x.shape[2]
        s = torch.stft(x.reshape(B * C, T), n_fft=self.window_size, hop_length=self.window_size // 2,
                       win_length=self.window_size, window=self.stft_window.to(x.dtype), center=False,
                       return_complex=True)
        return torch.abs(s)[:, :self.n_freqs].to(x.dtype).reshape(B, C, self.n_freqs, -1)

    def encode(self, x: Tensor, spec: Tensor) -> Tensor:
        B, C = x.shape[0], x.shape[1]
        nf = spec.shape[-1]
        fr = spec.transpose(1, 2).reshape(-1, 1, self.n_freqs)
        fr = self.frequency_patch_embedding(fr).transpose(1, 2)
        fr = self.frequency_encoder(fr).transpose(1, 2)
        fr = self.frequency_projection(fr) * self.frequency_attention(fr)
        fr = fr.reshape(B * C, nf, self.embed_dim // 2)
        tm = self.temporal_patch_embedding(x.reshape(B * C, 1, -1)).transpose(1, 2)
        return F.normalize(self.temporal_encoder(torch.cat([fr, tm], dim=-1)), p=2.0, dim=-1)

    def tokenize(self, x: Tensor, spectrogram_mask: Optional[Tensor] = None) -> TFMTokenizerOutput:
        target = self.compute_spectrogram(x)
        B, C, nf = target.shape[0], target.shape[1], target.shape[3]
        if nf > self.max_seq_len:
            raise ValueError(f"{nf} STFT frames exceed max_seq_len={self.max_seq_len}.")
        vis = target
        if spectrogram_mask is not None:
            vis = target * spectrogram_mask.__val__()
        emb = self.encode(x, vis.reshape(B * C, self.n_freqs, -1))
        cb, ids = self.quantizer.apply_codebook(emb)
        q = emb + (cb - emb).detach()
        cl = F.mse_loss(cb, emb)
        rec = self.final_layer(self.decoder(q)).transpose(1, 2)
        return TFMTokenizerOutput(rec.reshape(B, C, self.n_freqs, nf), ids.reshape(B, C, nf), q, emb,
                                  cl.detach() + self.commitment_cost * cl, target)

    def forward(self, x: Tensor) -> Tensor:
        return self.tokenize(x).reconstruction

    def _get_name(self) -> str:
        return "TFMTokenizer"
