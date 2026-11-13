# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# Artemis123 .bin/.txt reader (mne/io/artemis123/artemis123.py). Narrower:
# no channel position table, no HPI subsystem, no head localization
# (add_head_trans), no digitized points.

import numpy as np
from ._bytes import _read_file, _read_text, _f32_be
from ._fmtutil import _utc_secs
from .._info import Info
from .._containers import RawArray


def _ends(p: str, n: int) -> str:
    return p[:len(p) - n]


def read_raw_artemis123(input_fname, preload = False, verbose = None, pos_fname = None,
                        add_head_trans: bool = True) -> RawArray:
    fn = str(input_fname)
    if fn.endswith(".txt"):
        fn = _ends(fn, 4) + ".bin"
    elif not fn.endswith(".bin"):
        raise RuntimeError("Valid artemis123 files must end in \"txt\" or \".bin\".")
    hdr = _ends(fn, 4) + ".txt"
    txt = _read_text(hdr)
    main = Dict[str, str]()
    ch_name = List[str]()
    ch_scale = List[float]()
    ch_lock = List[str]()
    comments = ""
    nsamp = 1
    flag = 0
    for line in txt.split("\n"):
        st = line.strip()
        if st == "" or (flag == 2 and line.startswith("DAQ Map")):
            continue
        if line.startswith("<end"):
            flag = 0
        elif line.startswith("<start main header>"):
            flag = 1
        elif line.startswith("<start per channel header>"):
            flag = 2
        elif line.startswith("<start comments>"):
            flag = 3
        elif line.startswith("<start length>"):
            flag = 4
        elif line.startswith("<start filtering history>"):
            flag = 5
        else:
            if flag == 1:
                v = st.split("\t")
                main[v[0]] = v[1] if len(v) > 1 else ""
            elif flag == 2:
                v = st.split("\t")
                if len(v) != 7:
                    raise OSError(f"Error parsing line \n\t:{line}\nfrom file {hdr}")
                ch_name.append(v[0])
                ch_scale.append(float(v[1]))
                ch_lock.append(v[6])
            elif flag == 3:
                comments = comments + st
            elif flag == 4:
                nsamp = int(st)
    sfreq = float(main["DAQ Sample Rate"])
    n = len(ch_name)
    types = List[str]()
    cals = List[float]()
    bads = List[str]()
    ref_mag = ["REF_001", "REF_002", "REF_003", "REF_004", "REF_005", "REF_006"]
    ref_grad = ["REF_007", "REF_008", "REF_009", "REF_010", "REF_011", "REF_012"]
    for i in range(n):
        nm = ch_name[i]
        cal = ch_scale[i]
        if cal == 0.0:
            cal = 4.716e-10
            bads.append(nm)
        if nm.startswith("MEG"):
            types.append("mag")
            cal *= 1e-15
        elif nm in ref_mag or nm in ref_grad:
            types.append("ref_meg")
            cal *= 1e-15
        elif nm.startswith("REF"):
            types.append("misc")
            bads.append(nm)
        elif nm.startswith("AUX") or nm.startswith("MIO"):
            types.append("misc")
        elif nm.startswith("TRG"):
            types.append("stim")
        else:
            raise ValueError(f"Channel does not match expected channel Types:\"{nm}\"")
        cals.append(cal)
        if ch_lock[i] == "TRUE":
            bads.append(nm)
    info = Info(ch_name, sfreq, types)
    for i in range(n):
        info.cals[i] = cals[i]
        info.ranges[i] = 1.0
        t = ch_name[i]
        if types[i] == "mag":
            info.coil_types[i] = 7501
            info.units[i] = 112
        elif types[i] == "ref_meg":
            info.coil_types[i] = 7502 if t in ref_mag else 7503
            info.units[i] = 112
        else:
            info.coil_types[i] = 0
            info.units[i] = 107 if types[i] == "misc" else -1
        info.locs[i] = [0.0 for _ in range(12)]
    # measurement date from the file name Artemis_Data_YYYY-MM-DD-HHh-MMm_*
    base = fn[fn.rfind("/") + 1:]
    parts = base.split("_")
    if len(parts) > 2:
        try:
            dp = parts[2].split("-")
            hh = int(dp[3][:len(dp[3]) - 1])
            mm = int(dp[4][:len(dp[4]) - 1])
            info.meas_date = _utc_secs(int(dp[0]), int(dp[1]), int(dp[2]), hh, mm, 0)
        except:
            info.meas_date = None
    desc = "Purpose : " + main["Purpose"] + "\n"
    desc += "Notes : " + main["Notes"] + "\n"
    desc += "Comments : " + comments
    info.description = desc
    uniq = List[str]()
    for x in bads:
        if x not in uniq:
            uniq.append(x)
    info.bads = uniq
    b = _read_file(fn)
    data = np.zeros((n, nsamp))
    for s in range(nsamp):
        for c in range(n):
            data[c, s] = float(_f32_be(b, (s * n + c) * 4)) * cals[c]
    raw = RawArray(data, info)
    raw._raw_kind = "RawArtemis123"
    return raw
