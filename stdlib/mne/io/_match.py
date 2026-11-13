# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# mne.io.match_channel_orders (mne/io/base.py).

from .._containers import RawArray


def match_channel_orders(insts: List[RawArray], copy: bool = True) -> List[RawArray]:
    """Make the channel orders of all instances match the first one."""
    out = List[RawArray]()
    for r in insts:
        out.append(r.copy() if copy else r)
    ch_order = list(out[0].info.ch_names)
    for k in range(1, len(out)):
        out[k] = out[k].reorder_channels(ch_order)
    return out
