# Generates expected values for test/braindecode/test_models_b_*.py
# (foundation / transformer models, group B) using real braindecode 2.0.0 /
# torch 2.15.0 (CPU, float32). Reuses the harness of gen_models_expected.py.
# Run: /path/to/bd-venv/bin/python test/braindecode/gen_models_b_expected.py --emit OUT.py Model ...
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_models_expected as G
import torch
import braindecode.models as M

G.TYPHON_HEADER = G.TYPHON_HEADER.replace("gen_models_expected.py", "gen_models_b_expected.py")
_WEIGHTED_TYPHON = '''
import torch.nn as nn

class _Weighted(nn.Module):
    # fixed non-uniform weight on a standardized output (see generator)
    m: M.NeuroRVQTokenizer
    def __init__(self, m: M.NeuroRVQTokenizer):
        super().__init__()
        self.m = m
        self.add_module("m", m)
    def forward(self, x: Tensor) -> Tensor:
        y = self.m(x)
        w = torch.sin(torch.arange(y.numel(), dtype=torch.float32) * 0.37).reshape(y.shape) + 1.5
        return y * w
'''

def _reve(**kw):
    # REVE's position bank needs the Hub: use explicit deterministic positions
    m = M.REVE(**kw)
    n = kw["n_chans"]
    m.default_pos = torch.tensor([[0.1 * i - 0.1, 0.05 * i, 0.02 * i + 0.1] for i in range(n)])
    return m


_REVE_POS = lambda n: "torch.tensor([" + ", ".join(f"[{0.1 * i - 0.1!r}, {0.05 * i!r}, {0.02 * i + 0.1!r}]" for i in range(n)) + "])"

_AXON_POS = [[0.03 * i - 0.04, 0.02 * i + 0.01, 0.05 - 0.01 * i] for i in range(4)]
_AXON_CHS = [{"ch_name": f"E{i}", "loc": p} for i, p in enumerate(_AXON_POS)]
_AXON_POS_C = "torch.tensor(" + repr(_AXON_POS) + ")"

_DIVER_CHS = [{"ch_name": f"E{i}", "loc": [0.01 * i, 0.02, 0.03 * i + 0.01] + [0] * 9, "kind": "eeg"} for i in range(3)]
_DIVER_MD = "torch.tensor([[0.0, 20.0, 10.0, 0.0, -1.0], [10.0, 20.0, 40.0, 0.0, -1.0], [20.0, 20.0, 70.0, 0.0, -1.0]])"

_ZUNA_POS = [[0.05 * i - 0.06, 0.03 - 0.02 * i, 0.04 + 0.01 * i] for i in range(3)]
_ZUNA_CHS = [{"ch_name": f"E{i}", "loc": p + [0.0] * 9} for i, p in enumerate(_ZUNA_POS)]
_ZUNA_POS_C = "torch.tensor(" + repr(_ZUNA_POS) + ")"

import numpy as _np
_OMNI_POS = [[0.03 * i - 0.04, 0.02 * i + 0.01, 0.05 - 0.01 * i] for i in range(3)]
_OMNI_CHS = [{"ch_name": f"E{i}", "loc": _np.array(p + [0.0] * 9), "kind": 2} for i, p in enumerate(_OMNI_POS)]
_OMNI_SMALL = dict(emb_dim=16, n_neuro=4, window_length=64, n_filters=4, ratios=(2, 2), kernel_size=3, last_kernel_size=3,
                   tokenizer_num_heads=2, codebook_dim=8, codebook_size=16, num_quantizers=2)
_OMNI_SMALL_C = ("emb_dim=16, n_neuro=4, window_length=64, n_filters=4, ratios=[2, 2], kernel_size=3, last_kernel_size=3, "
                 "tokenizer_num_heads=2, codebook_dim=8, codebook_size=16, num_quantizers=2, "
                 "sensor_positions=torch.tensor(" + repr(_OMNI_POS) + "), sensor_types=[0, 0, 0]")

class _Weighted(torch.nn.Module):
    # The standardized reconstruction makes (y*y + y).sum() constant (zero
    # gradient), so the test scales it by a fixed non-uniform weight. Same
    # wrapper (child "m") on the Typhon side, see TYPHON_HEADER below.
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y = self.m(x)[1]
        w = torch.sin(torch.arange(y.numel(), dtype=torch.float32) * 0.37).reshape(y.shape) + 1.5
        return y * w


def _rvq_tok(**kw):
    return _Weighted(M.NeuroRVQTokenizer(**kw))

_DANCE_POS = [[0.03 * (i % 5) - 0.06, 0.02 * (i // 5) - 0.03, 0.05 + 0.002 * i] for i in range(19)]
_DANCE_CHS = [{"ch_name": f"E{i}", "loc": _np.array(p + [0.0] * 9)} for i, p in enumerate(_DANCE_POS)]
_DANCE_POS_C = lambda n: "torch.tensor(" + repr(_DANCE_POS[:n]) + ")"
_DANCE_SMALL = dict(n_queries=5, n_virtual_channels=8, fourier_emb_dim=32, embed_dim=16, conv_hidden=16, conv_depth=3,
                    conv_kernel_size=3, conv_initial_linear=8, num_latents=6, perceiver_depth=1, cross_dim_head=8,
                    latent_dim_head=8, decoder_dim=16, decoder_depth=1)
_DANCE_SMALL_C = ("n_queries=5, n_virtual_channels=8, fourier_emb_dim=32, embed_dim=16, conv_hidden=16, conv_depth=3, "
                  "conv_kernel_size=3, conv_initial_linear=8, num_latents=6, perceiver_depth=1, cross_dim_head=8, "
                  "latent_dim_head=8, decoder_dim=16, decoder_depth=1")

# Default-size configurations: structure only (param count + state_dict keys/shapes),
# name -> (python ctor, typhon ctor)
DEFAULT_CASES = {
    "DANCE": (lambda: M.DANCE(chs_info=_DANCE_CHS, n_outputs=3, n_times=500),
              "M.DANCE(n_chans=19, n_outputs=3, n_times=500, channel_positions=" + _DANCE_POS_C(19) + ")"),
    "SeizureTransformer": (lambda: M.SeizureTransformer(n_chans=19, n_outputs=1, n_times=15360),
                           "M.SeizureTransformer(n_chans=19, n_outputs=1, n_times=15360)"),
}


def emit_defaults(path, names):
    out = []
    for n in names:
        ctor, cctor = DEFAULT_CASES[n]
        m = ctor()
        sd = m.state_dict()
        out.append(f"""
@test
def test_{n}_default_structure():
    m = {cctor}
    n = 0
    for p in m.parameters():
        n += p.numel()
    assert n == {sum(p.numel() for p in m.parameters())}
    sd = m.state_dict()
    assert [k for k in sd.keys()] == {list(sd.keys())!r}
    assert [[int(d) for d in v.shape] for v in sd.values()] == {[list(v.shape) for v in sd.values()]!r}
test_{n}_default_structure()
""")
    with open(path, "a") as f:
        f.write("\n".join(out))

# name -> (python ctor, input shape, typhon ctor)
CASES_B = {
    "BIOT": (lambda: M.BIOT(n_chans=4, n_outputs=3, n_times=400, sfreq=200, embed_dim=32, num_heads=4, num_layers=2),
             (2, 4, 400), "M.BIOT(n_chans=4, n_outputs=3, n_times=400, sfreq=200.0, embed_dim=32, num_heads=4, num_layers=2)"),
    "CBraMod": (lambda: M.CBraMod(n_chans=4, n_outputs=3, n_times=400, patch_size=100, dim_feedforward=64, n_layer=2, nhead=4, emb_dim=16,
                                  channels_kernel_stride_padding_norm=((8, 25, 12, 12, (2, 8)), (8, 3, 1, 1, (2, 8)))),
                (2, 4, 400), "M.CBraMod(n_chans=4, n_outputs=3, n_times=400, patch_size=100, dim_feedforward=64, n_layer=2, nhead=4, emb_dim=16, "
                             "channels_kernel_stride_padding_norm=[(8, 25, 12, 12, (2, 8)), (8, 3, 1, 1, (2, 8))])"),
    "Labram": (lambda: M.Labram(n_chans=128, n_outputs=3, n_times=128, patch_size=64, num_layers=2, num_heads=4),
               (2, 128, 128), "M.Labram(n_chans=128, n_outputs=3, n_times=128, patch_size=64, num_layers=2, num_heads=4)"),
    "LabramDecoder": (lambda: M.Labram(n_chans=128, n_outputs=3, n_times=200, patch_size=64, num_layers=2, num_heads=4, embed_dim=32,
                                       neural_tokenizer=False, qkv_bias=True, conv_in_channels=2, use_mean_pooling=False),
                      (2, 128, 200), "M.Labram(n_chans=128, n_outputs=3, n_times=200, patch_size=64, num_layers=2, num_heads=4, embed_dim=32, "
                                     "neural_tokenizer=False, qkv_bias=True, conv_in_channels=2, use_mean_pooling=False)"),
    "EEGPT": (lambda: M.EEGPT(n_chans=4, n_outputs=3, n_times=256, embed_dim=32, depth=2, num_heads=4, embed_num=2),
              (2, 4, 256), "M.EEGPT(n_chans=4, n_outputs=3, n_times=256, embed_dim=32, depth=2, num_heads=4, embed_num=2)"),
    "EEGPTLinear": (lambda: M.EEGPT(n_chans=4, n_outputs=3, n_times=250, embed_dim=32, depth=1, num_heads=2, embed_num=1,
                                    chan_proj_type="linear", n_chans_target=19, drop_path_rate=0.1),
                    (2, 4, 250), "M.EEGPT(n_chans=4, n_outputs=3, n_times=250, embed_dim=32, depth=1, num_heads=2, embed_num=1, "
                                 "chan_proj_type=\"linear\", n_chans_target=19, drop_path_rate=0.1)"),
    "BENDR": (lambda: M.BENDR(n_chans=4, n_outputs=3, n_times=600, encoder_h=32, contextualizer_hidden=64, transformer_layers=2, transformer_heads=4),
              (2, 4, 600), "M.BENDR(n_chans=4, n_outputs=3, n_times=600, encoder_h=32, contextualizer_hidden=64, transformer_layers=2, transformer_heads=4)"),
    "BENDREnc": (lambda: M.BENDR(n_chans=4, n_outputs=3, n_times=600, encoder_h=32, contextualizer_hidden=64, transformer_layers=1, transformer_heads=4,
                                 encoder_only=True, projection_head=True),
                 (2, 4, 600), "M.BENDR(n_chans=4, n_outputs=3, n_times=600, encoder_h=32, contextualizer_hidden=64, transformer_layers=1, transformer_heads=4, "
                              "encoder_only=True, projection_head=True)"),
    "BrainBERT": (lambda: M.BrainBERT(n_chans=3, n_outputs=2, n_times=1200, sfreq=2048, hidden_dim=32, ffn_dim=64, nperseg=100, noverlap=80,
                                      idx_freq_cutoff=20, stft_clip=2, pool_n_frames=6),
                  (2, 3, 1200), "M.BrainBERT(n_chans=3, n_outputs=2, n_times=1200, sfreq=2048.0, hidden_dim=32, ffn_dim=64, nperseg=100, noverlap=80, "
                                "idx_freq_cutoff=20, stft_clip=2, pool_n_frames=6)"),
    "BrainBERTNoPool": (lambda: M.BrainBERT(n_chans=2, n_outputs=2, n_times=300, sfreq=2048, hidden_dim=16, ffn_dim=32, n_layers=1, n_heads=2,
                                            nperseg=64, noverlap=48, idx_freq_cutoff=12, stft_clip=1, pool_n_frames=None,
                                            stft_zscore_before_clip=False, activation=torch.nn.ReLU),
                        (2, 2, 300), "M.BrainBERT(n_chans=2, n_outputs=2, n_times=300, sfreq=2048.0, hidden_dim=16, ffn_dim=32, n_layers=1, n_heads=2, "
                                     "nperseg=64, noverlap=48, idx_freq_cutoff=12, stft_clip=1, pool_n_frames=None, "
                                     "stft_zscore_before_clip=False, activation=torch.nn.ReLU)"),
    "CSBrain": (lambda: M.CSBrain(chs_info=[{"ch_name": n} for n in ["FP1", "C3", "T7", "P3", "O1", "CZ"]], n_outputs=3, n_times=200,
                                  patch_size=50, nhead=2, dim_feedforward=32, n_layer=2, emb_dim=8),
                (2, 6, 200), "M.CSBrain(chs_info=[{\"ch_name\": n} for n in [\"FP1\", \"C3\", \"T7\", \"P3\", \"O1\", \"CZ\"]], n_outputs=3, n_times=200, "
                             "patch_size=50, nhead=2, dim_feedforward=32, n_layer=2, emb_dim=8)"),
    "CSBrainNoReg": (lambda: M.CSBrain(n_chans=3, n_outputs=2, n_times=180, patch_size=50, nhead=2, dim_feedforward=16, n_layer=1, emb_dim=4,
                                       head_hidden_dim=12),
                     (2, 3, 180), "M.CSBrain(n_chans=3, n_outputs=2, n_times=180, patch_size=50, nhead=2, dim_feedforward=16, n_layer=1, emb_dim=4, "
                                  "head_hidden_dim=12)"),
    "CodeBrain": (lambda: M.CodeBrain(n_chans=3, n_outputs=2, n_times=400, patch_size=100, res_channels=8, skip_channels=8, out_channels=8,
                                      num_res_layers=2, s4_lmax=16, conv_out_chans=5, conv_groups=5, codebook_size_t=16, codebook_size_f=16),
                  (2, 3, 400), "M.CodeBrain(n_chans=3, n_outputs=2, n_times=400, patch_size=100, res_channels=8, skip_channels=8, out_channels=8, "
                               "num_res_layers=2, s4_lmax=16, conv_out_chans=5, conv_groups=5, codebook_size_t=16, codebook_size_f=16)"),
    "CodeBrainSWA": (lambda: M.CodeBrain(n_chans=2, n_outputs=2, n_times=500, patch_size=100, res_channels=8, skip_channels=4, out_channels=4,
                                         num_res_layers=1, s4_lmax=4, s4_bidirectional=False, s4_layernorm=True, swa_window_size=3,
                                         conv_out_chans=5, conv_groups=5, codebook_size_t=8, codebook_size_f=8),
                     (2, 2, 500), "M.CodeBrain(n_chans=2, n_outputs=2, n_times=500, patch_size=100, res_channels=8, skip_channels=4, out_channels=4, "
                                  "num_res_layers=1, s4_lmax=4, s4_bidirectional=False, s4_layernorm=True, swa_window_size=3, "
                                  "conv_out_chans=5, conv_groups=5, codebook_size_t=8, codebook_size_f=8)"),
    "LUNA": (lambda: M.LUNA(n_chans=4, n_outputs=3, n_times=160, embed_dim=16, depth=2, num_heads=2, num_queries=2),
             (2, 4, 160), "M.LUNA(n_chans=4, n_outputs=3, n_times=160, embed_dim=16, depth=2, num_heads=2, num_queries=2)"),
    "REVE": (lambda: _reve(n_chans=3, n_outputs=2, n_times=380, embed_dim=32, depth=2, heads=2, head_dim=16, patch_size=100, patch_overlap=20),
             (2, 3, 380), "M.REVE(n_chans=3, n_outputs=2, n_times=380, embed_dim=32, depth=2, heads=2, head_dim=16, patch_size=100, patch_overlap=20, "
                          "default_pos=" + _REVE_POS(3) + ")"),
    "REVEPool": (lambda: _reve(n_chans=2, n_outputs=2, n_times=350, embed_dim=16, depth=1, heads=2, head_dim=8, patch_size=100, patch_overlap=20,
                               use_geglu=False, attention_pooling=True, on_non_divisible="pad", freqs=3),
                 (2, 2, 350), "M.REVE(n_chans=2, n_outputs=2, n_times=350, embed_dim=16, depth=1, heads=2, head_dim=8, patch_size=100, patch_overlap=20, "
                              "use_geglu=False, attention_pooling=True, on_non_divisible=\"pad\", freqs=3, default_pos=" + _REVE_POS(2) + ")"),
    "STEEGFormer": (lambda: M.STEEGFormer(n_chans=3, n_outputs=2, n_times=70, embed_dim=16, depth=2, num_heads=2, n_chans_pos=8),
                    (2, 3, 70), "M.STEEGFormer(n_chans=3, n_outputs=2, n_times=70, embed_dim=16, depth=2, num_heads=2, n_chans_pos=8)"),
    "STEEGFormerCls": (lambda: M.STEEGFormer(n_chans=3, n_outputs=2, n_times=64, embed_dim=16, depth=1, num_heads=2, n_chans_pos=8,
                                             global_pool="cls", chan_pos_idx=[5, 0, 2], mlp_ratio=2),
                       (2, 3, 64), "M.STEEGFormer(n_chans=3, n_outputs=2, n_times=64, embed_dim=16, depth=1, num_heads=2, n_chans_pos=8, "
                                   "global_pool=\"cls\", chan_pos_idx=[5, 0, 2], mlp_ratio=2)"),
    "MVPFormer": (lambda: M.MVPFormer(n_chans=3, n_outputs=2, n_times=300, segment_len=64, d_model=16, n_layers=2, n_heads=4, n_head_kv=2,
                                      d_inner=32, local_window=2, max_segments=8, max_channels=4),
                  (2, 3, 300), "M.MVPFormer(n_chans=3, n_outputs=2, n_times=300, segment_len=64, d_model=16, n_layers=2, n_heads=4, n_head_kv=2, "
                               "d_inner=32, local_window=2, max_segments=8, max_channels=4)"),
    "MVPFormerConcat": (lambda: M.MVPFormer(n_chans=2, n_outputs=2, n_times=128, segment_len=32, d_model=8, n_layers=1, n_heads=2, n_head_kv=1,
                                            d_inner=16, local_window=1, global_att=False, pooling="concat", max_segments=4, max_channels=2),
                        (2, 2, 128), "M.MVPFormer(n_chans=2, n_outputs=2, n_times=128, segment_len=32, d_model=8, n_layers=1, n_heads=2, n_head_kv=1, "
                                     "d_inner=16, local_window=1, global_att=False, pooling=\"concat\", max_segments=4, max_channels=2)"),
    "EEGDINO": (lambda: M.EEGDINO(n_chans=3, n_outputs=2, n_times=400, sfreq=200, patch_size=100, n_layer=2, nhead=2, dim_feedforward=32,
                                  channels_kernel_stride_padding_norm=((8, 25, 12, 12, (2, 8)), (8, 3, 1, 1, (2, 8)))),
                (2, 3, 400), "M.EEGDINO(n_chans=3, n_outputs=2, n_times=400, sfreq=200.0, patch_size=100, n_layer=2, nhead=2, dim_feedforward=32, "
                             "channels_kernel_stride_padding_norm=[(8, 25, 12, 12, (2, 8)), (8, 3, 1, 1, (2, 8))])"),
    "EpiNT": (lambda: M.EpiNT(n_chans=3, n_outputs=2, n_times=200, patch_size=32, embed_dim=16, n_layers=2, n_heads=2, ffn_dim=32),
              (2, 3, 200), "M.EpiNT(n_chans=3, n_outputs=2, n_times=200, patch_size=32, embed_dim=16, n_layers=2, n_heads=2, ffn_dim=32)"),
    "Brant": (lambda: M.Brant(n_chans=3, n_outputs=2, n_times=250, patch_size=64, embed_dim=16, ffn_dim=32, temporal_n_layers=2,
                              spatial_n_layers=1, n_heads=2),
              (2, 3, 250), "M.Brant(n_chans=3, n_outputs=2, n_times=250, patch_size=64, embed_dim=16, ffn_dim=32, temporal_n_layers=2, "
                           "spatial_n_layers=1, n_heads=2)"),
    "TFMTokenizer": (lambda: M.TFMTokenizer(n_chans=2, n_times=300, sfreq=50, embed_dim=16, codebook_size=32, freq_patch_size=5,
                                            freq_encoder_depth=1, temporal_encoder_depth=1, decoder_depth=1),
                     (2, 2, 300), "M.TFMTokenizer(n_chans=2, n_times=300, sfreq=50.0, embed_dim=16, codebook_size=32, freq_patch_size=5, "
                                  "freq_encoder_depth=1, temporal_encoder_depth=1, decoder_depth=1)"),
    "PBT": (lambda: M.PBT(n_chans=3, n_outputs=2, n_times=200, d_input=32, embed_dim=16, num_layers=2, num_heads=2),
            (2, 3, 200), "M.PBT(n_chans=3, n_outputs=2, n_times=200, d_input=32, embed_dim=16, num_layers=2, num_heads=2)"),
    "PBTBias": (lambda: M.PBT(n_chans=2, n_outputs=2, n_times=128, d_input=16, embed_dim=8, num_layers=1, num_heads=2,
                              bias_transformer=True, learnable_cls=False),
                (2, 2, 128), "M.PBT(n_chans=2, n_outputs=2, n_times=128, d_input=16, embed_dim=8, num_layers=1, num_heads=2, "
                             "bias_transformer=True, learnable_cls=False)"),
    "SleepFM": (lambda: M.SleepFM(n_chans=2, n_outputs=3, n_times=300, sfreq=128, patch_size=64, embed_dim=16, num_heads=2, num_layers=1,
                                  pooling_heads=2, max_seq_length=8),
                (2, 2, 300), "M.SleepFM(n_chans=2, n_outputs=3, n_times=300, sfreq=128.0, patch_size=64, embed_dim=16, num_heads=2, num_layers=1, "
                             "pooling_heads=2, max_seq_length=8)"),
    "SleepFMStager": (lambda: M.SleepFMStager(n_chans=3, n_outputs=5, n_times=448, sfreq=128, patch_size=64, embed_dim=16, encoder_num_heads=2,
                                              encoder_num_layers=1, encoder_pooling_heads=2, encoder_chunk_patches=3, staging_num_heads=2,
                                              staging_pooling_heads=2, max_seq_length=16, channel_modalities=[1, 0, 1]),
                      (2, 3, 448), "M.SleepFMStager(n_chans=3, n_outputs=5, n_times=448, sfreq=128.0, patch_size=64, embed_dim=16, encoder_num_heads=2, "
                                   "encoder_num_layers=1, encoder_pooling_heads=2, encoder_chunk_patches=3, staging_num_heads=2, "
                                   "staging_pooling_heads=2, max_seq_length=16, channel_modalities=[1, 0, 1])"),
    "DuIN": (lambda: M.DuIN(n_chans=4, n_outputs=3, n_times=250, sfreq=1000, patch_size=50, n_neural=4, n_filters=(8, 8, 4),
                            kernel_sizes=(9, 3, 3), strides=(5, 1, 1), n_layers=2, n_heads=2, head_dim=8, ffn_dim=16, head_hidden_dim=8),
             (2, 4, 250), "M.DuIN(n_chans=4, n_outputs=3, n_times=250, sfreq=1000.0, patch_size=50, n_neural=4, n_filters=[8, 8, 4], "
                          "kernel_sizes=[9, 3, 3], strides=[5, 1, 1], n_layers=2, n_heads=2, head_dim=8, ffn_dim=16, head_hidden_dim=8)"),
    "BaRISTA": (lambda: M.BaRISTA(n_chans=3, n_outputs=2, n_times=200, patch_size=64, d_model=16, n_layers=2, num_heads=2, mlp_ratio=2,
                                  cnn_depth=2, cnn_channels=3, coord_bins=10, spatial_indices=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                (2, 3, 200), "M.BaRISTA(n_chans=3, n_outputs=2, n_times=200, patch_size=64, d_model=16, n_layers=2, num_heads=2, mlp_ratio=2, "
                             "cnn_depth=2, cnn_channels=3, coord_bins=10, spatial_indices=torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))"),
    "BaRISTALobes": (lambda: M.BaRISTA(n_chans=3, n_outputs=2, n_times=128, patch_size=32, d_model=8, n_layers=1, num_heads=2, mlp_ratio=2,
                                       cnn_depth=1, cnn_channels=2, spatial_scale="lobes", spatial_indices=[0, 3, 5], pooling="mean"),
                     (2, 3, 128), "M.BaRISTA(n_chans=3, n_outputs=2, n_times=128, patch_size=32, d_model=8, n_layers=1, num_heads=2, mlp_ratio=2, "
                                  "cnn_depth=1, cnn_channels=2, spatial_scale=\"lobes\", spatial_indices=torch.tensor([0, 3, 5]), pooling=\"mean\")"),
    "NeuroRVQ": (lambda: M.NeuroRVQ(n_chans=3, n_outputs=2, n_times=200, sfreq=200, depth=2, num_heads=2, max_patches=4, patch_size=64,
                                    channel_names=["FP1", "cz", "o2"]),
                 (2, 3, 200), "M.NeuroRVQ(n_chans=3, n_outputs=2, n_times=200, sfreq=200.0, depth=2, num_heads=2, max_patches=4, patch_size=64, "
                              "channel_names=[\"FP1\", \"cz\", \"o2\"])"),
    "NeuroRVQEmg": (lambda: M.NeuroRVQ(n_chans=2, n_outputs=2, n_times=300, sfreq=1000, modality="emg", depth=1, num_heads=2, patch_size=80,
                                       max_patches=6, init_values=0.0, qk_norm=False, qkv_bias=False),
                    (2, 2, 300), "M.NeuroRVQ(n_chans=2, n_outputs=2, n_times=300, sfreq=1000.0, modality=\"emg\", depth=1, num_heads=2, patch_size=80, "
                                 "max_patches=6, init_values=0.0, qk_norm=False, qkv_bias=False)"),
    "AXON": (lambda: M.AXON(chs_info=_AXON_CHS, n_outputs=2, n_times=400, sfreq=200, patch_size=50, patch_stride=40, embed_dim=16,
                            depth=2, num_heads=2, temporal_window=2, head_hidden_dim=8),
             (2, 4, 400), "M.AXON(n_chans=4, n_outputs=2, n_times=400, sfreq=200.0, patch_size=50, patch_stride=40, embed_dim=16, "
                          "depth=2, num_heads=2, temporal_window=2, head_hidden_dim=8, channel_positions=" + _AXON_POS_C + ")"),
    "DIVER1": (lambda: M.DIVER1(chs_info=_DIVER_CHS, n_outputs=2, n_times=200, patch_size=50, d_model=32, n_layers=1, cnn_depth=2,
                                cnn_kernel_size=7),
               (2, 3, 200), "M.DIVER1(n_chans=3, n_outputs=2, n_times=200, patch_size=50, d_model=32, n_layers=1, cnn_depth=2, "
                            "cnn_kernel_size=7, chan_metadata=" + _DIVER_MD + ")"),
    "DIVER1Mean": (lambda: M.DIVER1(chs_info=_DIVER_CHS, n_outputs=2, n_times=180, patch_size=64, d_model=32, n_layers=2, cnn_depth=1,
                                    cnn_kernel_size=5, pooling="mean", use_stcpe=False, mup_attention=False, num_heads=2),
                   (2, 3, 180), "M.DIVER1(n_chans=3, n_outputs=2, n_times=180, patch_size=64, d_model=32, n_layers=2, cnn_depth=1, "
                                "cnn_kernel_size=5, pooling=\"mean\", use_stcpe=False, mup_attention=False, num_heads=2, chan_metadata=" + _DIVER_MD + ")"),
    "ZUNA": (lambda: M.ZUNA(chs_info=_ZUNA_CHS, n_outputs=2, n_times=100, sfreq=256, dim=32, n_layers=2, n_heads=2, head_dim=16,
                            fine_time_pts=16, latent_dim=8, multiple_of=16),
             (2, 3, 100), "M.ZUNA(n_chans=3, n_outputs=2, n_times=100, sfreq=256.0, dim=32, n_layers=2, n_heads=2, head_dim=16, "
                          "fine_time_pts=16, latent_dim=8, multiple_of=16, channel_positions=" + _ZUNA_POS_C + ")"),
    "ZUNANoReg": (lambda: M.ZUNA(chs_info=_ZUNA_CHS, n_outputs=2, n_times=96, sfreq=256, dim=16, n_layers=1, n_heads=2, head_dim=8,
                                 fine_time_pts=16, latent_dim=4, multiple_of=8, registers=False, sandwich_norm=False, qk_norm=False),
                  (2, 3, 96), "M.ZUNA(n_chans=3, n_outputs=2, n_times=96, sfreq=256.0, dim=16, n_layers=1, n_heads=2, head_dim=8, "
                              "fine_time_pts=16, latent_dim=4, multiple_of=8, registers=False, sandwich_norm=False, qk_norm=False, "
                              "channel_positions=" + _ZUNA_POS_C + ")"),
    "BrainModule": (lambda: M.BrainModule(n_chans=4, n_outputs=3, n_times=200, hidden_dim=16, depth=4, dilation_period=2),
                    (2, 4, 200), "M.BrainModule(n_chans=4, n_outputs=3, n_times=200, hidden_dim=16, depth=4, dilation_period=2)"),
    "BrainModuleGrow": (lambda: M.BrainModule(n_chans=3, n_outputs=2, n_times=120, hidden_dim=8, depth=3, growth=1.5, glu=1,
                                              batch_norm=False, kernel_size=5),
                        (2, 3, 120), "M.BrainModule(n_chans=3, n_outputs=2, n_times=120, hidden_dim=8, depth=3, growth=1.5, glu=1, "
                                     "batch_norm=False, kernel_size=5)"),
    "MAPA": (lambda: M.MAPA(chs_info=[{"ch_name": n} for n in ["LA1", "LA2", "RB1", "LA3"]], n_outputs=2, n_times=1024, sfreq=2048,
                            d_model=64, mlp_ratio=1, regions=[3, 74, 10, 3]),
             (2, 4, 1024), "M.MAPA(chs_info=[{\"ch_name\": n} for n in [\"LA1\", \"LA2\", \"RB1\", \"LA3\"]], n_outputs=2, n_times=1024, sfreq=2048.0, "
                           "d_model=64, mlp_ratio=1, regions=[3, 74, 10, 3])"),
    "BrainTokenizer": (lambda: M.BrainTokenizer(chs_info=_OMNI_CHS, n_times=128, sfreq=256, **_OMNI_SMALL),
                       (2, 3, 128), "M.BrainTokenizer(n_chans=3, n_times=128, sfreq=256.0, " + _OMNI_SMALL_C + ")"),
    "BrainOmni": (lambda: M.BrainOmni(chs_info=_OMNI_CHS, n_outputs=2, n_times=128, sfreq=256, lm_dim=16, num_heads=4, depth=2,
                                      **_OMNI_SMALL),
                  (2, 3, 128), "M.BrainOmni(n_chans=3, n_outputs=2, n_times=128, sfreq=256.0, lm_dim=16, num_heads=4, depth=2, "
                               + _OMNI_SMALL_C + ")"),
    "NeuroRVQTokenizer": (lambda: _rvq_tok(n_chans=2, n_times=128, sfreq=200, patch_size=64, max_patches=4, encoder_depth=1,
                                            decoder_depth=1, num_heads=2, n_code=16, code_dim=8, num_quantizers=2,
                                            channel_names=["fp1", "cz"]),
                          (2, 2, 128), "_Weighted(M.NeuroRVQTokenizer(n_chans=2, n_times=128, sfreq=200.0, patch_size=64, max_patches=4, encoder_depth=1, "
                                       "decoder_depth=1, num_heads=2, n_code=16, code_dim=8, num_quantizers=2, channel_names=[\"fp1\", \"cz\"]))"),
    "DANCE": (lambda: M.DANCE(chs_info=_DANCE_CHS[:4], n_outputs=3, n_times=200, **_DANCE_SMALL),
              (2, 4, 200), "M.DANCE(n_chans=4, n_outputs=3, n_times=200, " + _DANCE_SMALL_C + ", channel_positions=" + _DANCE_POS_C(4) + ")"),
    "DANCENoMerger": (lambda: M.DANCE(n_chans=3, chs_info=[{"ch_name": f"E{i}", "loc": _np.zeros(12)} for i in range(3)], n_outputs=2,
                                      n_times=100, use_channel_merger=False, **_DANCE_SMALL),
                      (2, 3, 100), "M.DANCE(n_chans=3, n_outputs=2, n_times=100, use_channel_merger=False, " + _DANCE_SMALL_C + ")"),
    "SeizureTransformer": (lambda: M.SeizureTransformer(n_chans=3, n_outputs=1, n_times=200, n_filters=(4, 8, 16), encoder_kernel_sizes=(11, 9, 7),
                                                        decoder_kernel_sizes=(3, 5, 5), res_kernel_sizes=(3, 2), num_layers=2, num_heads=4,
                                                        dim_feedforward=32),
                           (2, 3, 200), "M.SeizureTransformer(n_chans=3, n_outputs=1, n_times=200, n_filters=[4, 8, 16], "
                                        "encoder_kernel_sizes=[11, 9, 7], decoder_kernel_sizes=[3, 5, 5], res_kernel_sizes=[3, 2], "
                                        "num_layers=2, num_heads=4, dim_feedforward=32)"),
}

G.CASES.clear()
G.CTORS.clear()
for k, (ctor, shape, cctor) in CASES_B.items():
    G.CASES[k] = (ctor, shape)
    G.CTORS[k] = cctor

def _fix_inf(path):
    # buffers such as attention masks may sum to +/-inf: emit as math.inf
    import re
    s = open(path).read()
    s = "\n".join(l for l in s.split("\n") if not l.startswith("Using default location"))
    s = re.sub(r"(?<![\w.])(-?)inf(?![\w])", r"\1math.inf", s)
    s = re.sub(r"(?<![\w.])nan(?![\w])", "math.nan", s)
    open(path, "w").write(s)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--emit":
        if "NeuroRVQTokenizer" in sys.argv[3:]:
            G.TYPHON_HEADER += _WEIGHTED_TYPHON
        G.emit(sys.argv[2], sys.argv[3:] or list(G.CASES))
        _fix_inf(sys.argv[2])
        dn = [n for n in sys.argv[3:] if n in DEFAULT_CASES]
        if dn:
            emit_defaults(sys.argv[2], dn)
    else:
        for n in (sys.argv[1:] or list(G.CASES)):
            G.run(n)
