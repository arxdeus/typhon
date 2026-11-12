# Expected values for test/braindecode/test_models_a_*.py (group A ports).
# Reuses the harness of gen_models_expected.py.
# Run: /path/to/bd-venv/bin/python test/braindecode/gen_models_a_expected.py --emit FILE Model ...
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_models_expected as G

M = G.M

import math as _m
import numpy as _np


def _locs(n):
    return [(_m.cos(1.1 * i), _m.sin(1.1 * i), 0.2 * i - 0.3) for i in range(n)]


def _hlocs(n):
    return [(0.09 * _m.cos(1.1 * i), 0.09 * _m.sin(1.1 * i), 0.01 * i) for i in range(n)]


def hchs_py(n):
    return [{"ch_name": f"E{i}", "loc": _np.array(l + (0.0,) * 9)} for i, l in enumerate(_hlocs(n))]


def hchs_typhon(n):
    items = ", ".join('{"ch_name": "E%d", "loc": "%r %r %r"}' % ((i,) + l) for i, l in enumerate(_hlocs(n)))
    return "[" + items + "]"


def chs_py(n):
    return [{"ch_name": f"E{i}", "loc": _np.array(l + (0.0,) * 9)} for i, l in enumerate(_locs(n))]


def chs_typhon(n):
    items = ", ".join('{"ch_name": "E%d", "loc": "%r %r %r"}' % ((i,) + l) for i, l in enumerate(_locs(n)))
    return "[" + items + "]"


def gcn_adj(n):
    v = [abs(_m.sin(0.7 * i * j + 0.3 * (i + j))) if i != j else 0.0 for i in range(n) for j in range(n)]
    a = torch.tensor(v, dtype=torch.float32).reshape(n, n)
    return (a + a.T) / 2


def materialize(m, c, t):
    # lazy modules (GCNsNet): one eval-mode forward to materialize params
    m.eval()
    with torch.no_grad():
        m(G.det_input(1, c, t))
    m.train()
    return m


torch = G.torch
G.TYPHON_HEADER += """
def gcn_adj(n: int) -> Tensor:
    v = List[float]()
    for i in range(n):
        for j in range(n):
            v.append(abs(math.sin(0.7 * i * j + 0.3 * (i + j))) if i != j else 0.0)
    a = torch.tensor(v, dtype=torch.float32).reshape(n, n)
    return (a + a.T) / 2

def materialize(m, c: int, t: int):
    m.eval()
    with torch.no_grad():
        m(det_input(1, c, t))
    m.train()
    return m
"""

# name -> (python ctor, input shape[, typhon ctor])
A_CASES = {
    "TSception": ("M.TSception(n_chans=4, n_outputs=3, n_times=256, sfreq=128.0)", (2, 4, 256)),
    "EEGMiner": ("M.EEGMiner(n_chans=4, n_outputs=3, n_times=256, sfreq=128.0)", (2, 4, 256)),
    "EEGMiner_mag": ("M.EEGMiner(method=\"mag\", n_chans=4, n_outputs=3, n_times=256, sfreq=128.0)", (2, 4, 256)),
    "EEGMiner_corr": ("M.EEGMiner(method=\"corr\", n_chans=4, n_outputs=3, n_times=256, sfreq=128.0)", (2, 4, 256)),
    "DGCNN": ("M.DGCNN(n_chans=6, n_outputs=3, n_times=64, chs_info=chs_py(6), n_filters=8, mlp_dims=(16,))", (2, 6, 64),
              "M.DGCNN(n_chans=6, n_outputs=3, n_times=64, chs_info=%s, n_filters=8, mlp_dims=(16,))" % chs_typhon(6)),
    "DGCNN_k3": ("M.DGCNN(n_chans=6, n_outputs=3, n_times=64, chs_info=chs_py(6), n_filters=8, cheb_order=3, n_neighbors=2)", (2, 6, 64),
                 "M.DGCNN(n_chans=6, n_outputs=3, n_times=64, chs_info=%s, n_filters=8, cheb_order=3, n_neighbors=2)" % chs_typhon(6)),
    "SSTDPN": ("M.SSTDPN(n_chans=4, n_outputs=3, n_times=500)", (2, 4, 500)),
    "SSTDPN_l2": ("M.SSTDPN(n_chans=4, n_outputs=3, n_times=300, n_fused_filters=12, mvp_kernel_sizes=[20, 40], spt_attn_mode=\"l2\", spt_attn_global_context_kernel=100)", (2, 4, 300)),
    "MEDFormer": ("M.MEDFormer(n_chans=4, n_outputs=3, n_times=64, embed_dim=16, num_heads=2, num_layers=2, dim_feedforward=32)", (2, 4, 64)),
    "MEDFormer_sc": ("M.MEDFormer(n_chans=3, n_outputs=2, n_times=48, embed_dim=8, num_heads=2, num_layers=1, dim_feedforward=16, single_channel=True, no_inter_attn=True, patch_len_list=[4, 8])", (2, 3, 48)),
    "SignalJEPA": ("M.SignalJEPA(chs_info=chs_py(3), n_times=256, transformer__d_model=64, transformer__num_encoder_layers=2, transformer__num_decoder_layers=1, transformer__nhead=4, pos_encoder__spat_dim=12, pos_encoder__time_dim=16)", (2, 3, 256),
                   "M.SignalJEPA(chs_info=%s, n_times=256, transformer__d_model=64, transformer__num_encoder_layers=2, transformer__num_decoder_layers=1, transformer__nhead=4, pos_encoder__spat_dim=12, pos_encoder__time_dim=16)" % chs_typhon(3)),
    "SignalJEPA_Contextual": ("M.SignalJEPA_Contextual(n_outputs=2, chs_info=chs_py(3), n_times=256, transformer__d_model=64, transformer__num_encoder_layers=2, transformer__num_decoder_layers=1, transformer__nhead=4, pos_encoder__spat_dim=12, pos_encoder__time_dim=16)", (2, 3, 256),
                              "M.SignalJEPA_Contextual(n_outputs=2, chs_info=%s, n_times=256, transformer__d_model=64, transformer__num_encoder_layers=2, transformer__num_decoder_layers=1, transformer__nhead=4, pos_encoder__spat_dim=12, pos_encoder__time_dim=16)" % chs_typhon(3)),
    "SignalJEPA_PostLocal": ("M.SignalJEPA_PostLocal(n_chans=3, n_outputs=2, n_times=256)", (2, 3, 256)),
    "SignalJEPA_PreLocal": ("M.SignalJEPA_PreLocal(n_chans=3, n_outputs=2, n_times=256, feature_encoder__mode=\"layer_norm\")", (2, 3, 256)),
    "EEGCLIP": ("M.EEGCLIP(n_chans=4, n_outputs=16, n_times=600, eeg_embedding_dim=32, text_embedding_dim=24)", (2, 4, 600)),
    "MIRepNet": ("M.MIRepNet(n_chans=5, n_outputs=3, n_times=300, embed_dim=32, n_filters_time=8, n_filters_spat=16, num_layers=2, num_heads=4)", (2, 5, 300)),
    "Guetschel2026": ("M.Guetschel2026(chs_info=hchs_py(4), n_outputs=3, n_times=440, embed_dim=32, depth=2, num_heads=4, dim_feedforward=40, patch_size=100, patch_overlap=20)", (2, 4, 440),
                      "M.Guetschel2026(chs_info=%s, n_outputs=3, n_times=440, embed_dim=32, depth=2, num_heads=4, dim_feedforward=40, patch_size=100, patch_overlap=20)" % hchs_typhon(4)),
    "Guetschel2026_rp": ("M.Guetschel2026(chs_info=hchs_py(3), n_outputs=3, n_times=440, embed_dim=32, depth=2, num_heads=4, dim_feedforward=40, patch_size=100, patch_overlap=20, random_projection=24)", (2, 3, 440),
                         "M.Guetschel2026(chs_info=%s, n_outputs=3, n_times=440, embed_dim=32, depth=2, num_heads=4, dim_feedforward=40, patch_size=100, patch_overlap=20, random_projection=24)" % hchs_typhon(3)),
    "SensingDynamics": ("M.SensingDynamics(n_chans=16, n_outputs=4, n_times=480, sfreq=2000.0, temporal_channels=16, mid_channels=8, spatial_channels=8, mlp_hidden=16)", (2, 16, 480)),
    "NeuroPose": ("M.NeuroPose(n_chans=16, n_outputs=4, n_times=640, sfreq=2000.0, encoder_channels=(4, 8, 8), decoder_channels=(8, 4, 1), n_res_blocks=1, n_convs_per_block=2)", (2, 16, 640)),
    "EMG2QwertyNet": ("M.EMG2QwertyNet(n_outputs=5, n_chans=8, num_bands=2, electrodes_per_band=4, n_fft=16, hop_length=8, mlp_features=(24,), block_channels=(4, 4), kernel_width=4)", (2, 8, 200)),
    "EMG2QwertyNet_max": ("M.EMG2QwertyNet(n_outputs=5, n_chans=4, num_bands=1, electrodes_per_band=4, n_fft=16, hop_length=8, mlp_features=(16, 12), block_channels=(3,), kernel_width=3, pooling=\"max\", log_softmax=True)", (2, 4, 160)),
    "VEMG2Pose": ("M.VEMG2Pose(n_chans=4, n_outputs=3, n_times=600, sfreq=500.0, encoder_channels=16, feature_dim=8, hidden_size=12, tds_channels=4, num_position_steps=100, stem_kernel_sizes=(5, 3), stem_strides=(2, 2), tds_subsample_kernels=(5, 3), tds_subsample_strides=(2, 2), tds_kernel_widths=(3, 3))", (2, 4, 600)),
    "VEMG2Pose_mlp": ("M.VEMG2Pose(n_chans=4, n_outputs=3, n_times=600, sfreq=500.0, encoder_channels=16, feature_dim=8, hidden_size=12, tds_channels=4, num_position_steps=100, stem_kernel_sizes=(5, 3), stem_strides=(2, 2), tds_subsample_kernels=(5, 3), tds_subsample_strides=(2, 2), tds_kernel_widths=(3, 3), decoder=\"mlp\", decoder_hidden_sizes=(12, 10), parameterization=\"velocity\")", (2, 4, 600)),
    "MetaNeuromotorHand": ("M.MetaNeuromotorHand(n_outputs=5, n_chans=4, mpf_window_length=40, mpf_stride=10, mpf_n_fft=16, mpf_fft_stride=5, mpf_frequency_bins=((0.0, 200.0), (200.0, 600.0), (600.0, 1000.0)), invariance_hidden_dims=(12,), num_adjacent_cov=2, conformer_input_dim=8, conformer_ffn_dim=16, conformer_kernel_size=3, conformer_stride=(1, 2), conformer_num_heads=2, conformer_attn_window_size=(4, 3), conformer_num_layers=2)", (2, 4, 400)),
    "MetaNeuromotorHand_def": ("M.MetaNeuromotorHand(n_outputs=6, conformer_input_dim=16, conformer_ffn_dim=16, conformer_num_layers=3, conformer_stride=(1, 2, 1), conformer_attn_window_size=(8, 4, 0), conformer_kernel_size=3, invariance_hidden_dims=(16,))", (1, 16, 1600)),
    "PopulationTransformer": ("M.PopulationTransformer(chs_info=hchs_py(5), n_outputs=3, n_times=32, hidden_dim=16, ffn_dim=24, n_layers=2, n_heads=2, max_len=300, shift_coords=True)", (2, 5, 32),
                              "M.PopulationTransformer(chs_info=%s, n_outputs=3, n_times=32, hidden_dim=16, ffn_dim=24, n_layers=2, n_heads=2, max_len=300, shift_coords=True)" % hchs_typhon(5)),
    "PopulationTransformer_relu": ("M.PopulationTransformer(n_chans=4, n_outputs=3, n_times=32, hidden_dim=16, ffn_dim=24, n_layers=2, n_heads=2, max_len=300, activation=torch.nn.ReLU)", (2, 4, 32),
                                   "M.PopulationTransformer(n_chans=4, n_outputs=3, n_times=32, hidden_dim=16, ffn_dim=24, n_layers=2, n_heads=2, max_len=300, activation=torch.nn.ReLU)"),
    "GCNsNet": ("materialize(M.GCNsNet(n_chans=8, n_outputs=3, n_times=16, adjacency=gcn_adj(8), n_features=(4, 8, 8), cheb_orders=(2, 3, 2), pool_sizes=(2, 2, 1)), 8, 16)", (2, 8, 16)),
    "GCNsNet_default": ("materialize(M.GCNsNet(n_chans=6, n_outputs=2, n_times=8), 6, 8)", (2, 6, 8)),
}

for _n, _v in A_CASES.items():
    G.CASES[_n] = (eval("lambda: " + _v[0]), _v[1])
    G.CTORS[_n] = _v[2] if len(_v) > 2 else _v[0]

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--emit":
        G.emit(sys.argv[2], sys.argv[3:] or list(A_CASES))
    else:
        for n in (sys.argv[1:] or list(A_CASES)):
            G.run(n)
