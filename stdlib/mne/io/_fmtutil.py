# Shared helpers for the format readers (Typhon-specific).

import numpy as np
from .._info import Info


def _civil_days(y: int, m: int, d: int) -> int:
    yy = y - 1 if m <= 2 else y
    era = yy // 400
    yoe = yy - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _utc_secs(y: int, mo: int, d: int, h: int, mi: int, s: int) -> float:
    return float(_civil_days(y, mo, d)) * 86400.0 + float(h * 3600 + mi * 60 + s)


def _sel_idx(sel, names: List[str]) -> List[int]:
    # names or indices -> indices
    out = List[int]()
    for x in sel:
        if isinstance(x, str):
            for i in range(len(names)):
                if names[i] == x:
                    out.append(i)
        else:
            out.append(int(x))
    return out


def _prefix_idx(names: List[str], pre: str) -> List[int]:
    out = List[int]()
    for i in range(len(names)):
        if names[i].upper().startswith(pre):
            out.append(i)
    return out


def _set_kind(info: Info, idx: List[int], kind: str):
    for i in idx:
        info.ch_types[i] = kind
        info.units[i] = Info([kind], 1.0, [kind]).units[0]
        info.coil_types[i] = Info([kind], 1.0, [kind]).coil_types[0]


def _split(s: str) -> List[str]:
    return s.strip().split()
