# Copyright (C) 2026 the Typhon authors
# Ported from pandas 3.1.1 (BSD-3-Clause)

# Window kernels: ports of pandas/_libs/window/aggregations.pyx (ewm, ewmcov,
# roll_cov, roll_corr, roll_quantile). Same floating point operation order.

NAN = float("nan")
_EPS = 2.220446049250313e-16
_INV_COND_TOL = _EPS * 1e3
_MAX_ORIGIN = 1.3407807929942596e+154  # sqrt(float64 max)

def _mono(start: List[int], end: List[int]) -> bool:
    for i in range(1, len(start)):
        if start[i] < start[i - 1] or end[i] < end[i - 1]:
            return False
    return True

def ewm_kernel(vals: List[float], start: List[int], end: List[int], minp: int,
               com: float, adjust: bool, ignore_na: bool, normalize: bool = True) -> List[float]:
    N = len(vals)
    out = [NAN for _ in range(N)]
    if N == 0:
        return out
    alpha = 1.0 / (1.0 + com)
    old_wt_factor = 1.0 - alpha
    new_wt = 1.0 if adjust else alpha
    for j in range(len(start)):
        s = start[j]
        e = end[j]
        win = e - s
        weighted = vals[s]
        is_obs = weighted == weighted
        nobs = 1 if is_obs else 0
        out[s] = weighted if nobs >= minp else NAN
        old_wt = 1.0
        for i in range(1, win):
            cur = vals[s + i]
            is_obs = cur == cur
            if is_obs:
                nobs += 1
            if weighted == weighted:
                if is_obs or not ignore_na:
                    if normalize:
                        old_wt *= old_wt_factor
                    else:
                        weighted = old_wt_factor * weighted
                    if is_obs:
                        if normalize:
                            if weighted != cur:
                                if not adjust and com == 1.0:
                                    new_wt = 1.0 - old_wt
                                weighted = old_wt * weighted + new_wt * cur
                                weighted /= (old_wt + new_wt)
                            if adjust:
                                old_wt += new_wt
                            else:
                                old_wt = 1.0
                        else:
                            weighted += cur
            elif is_obs:
                weighted = cur
            out[s + i] = weighted if nobs >= minp else NAN
    return out

def ewmcov_kernel(x: List[float], start: List[int], end: List[int], minp: int,
                  y: List[float], com: float, adjust: bool, ignore_na: bool,
                  bias: bool) -> List[float]:
    N = len(x)
    out = [NAN for _ in range(N)]
    if N == 0:
        return out
    alpha = 1.0 / (1.0 + com)
    old_wt_factor = 1.0 - alpha
    new_wt = 1.0 if adjust else alpha
    for j in range(len(start)):
        s = start[j]
        e = end[j]
        win = e - s
        mean_x = x[s]
        mean_y = y[s]
        is_obs = (mean_x == mean_x) and (mean_y == mean_y)
        nobs = 1 if is_obs else 0
        if not is_obs:
            mean_x = NAN
            mean_y = NAN
        out[s] = (0.0 if bias else NAN) if nobs >= minp else NAN
        cov = 0.0
        sum_wt = 1.0
        sum_wt2 = 1.0
        old_wt = 1.0
        for i in range(1, win):
            cur_x = x[s + i]
            cur_y = y[s + i]
            is_obs = (cur_x == cur_x) and (cur_y == cur_y)
            if is_obs:
                nobs += 1
            if mean_x == mean_x:
                if is_obs or not ignore_na:
                    sum_wt *= old_wt_factor
                    sum_wt2 *= (old_wt_factor * old_wt_factor)
                    old_wt *= old_wt_factor
                    if is_obs:
                        old_mean_x = mean_x
                        old_mean_y = mean_y
                        if mean_x != cur_x:
                            mean_x = ((old_wt * old_mean_x) + (new_wt * cur_x)) / (old_wt + new_wt)
                        if mean_y != cur_y:
                            mean_y = ((old_wt * old_mean_y) + (new_wt * cur_y)) / (old_wt + new_wt)
                        cov = ((old_wt * (cov + ((old_mean_x - mean_x) * (old_mean_y - mean_y))))
                               + (new_wt * ((cur_x - mean_x) * (cur_y - mean_y)))) / (old_wt + new_wt)
                        sum_wt += new_wt
                        sum_wt2 += (new_wt * new_wt)
                        old_wt += new_wt
                        if not adjust:
                            sum_wt /= old_wt
                            sum_wt2 /= (old_wt * old_wt)
                            old_wt = 1.0
            elif is_obs:
                mean_x = cur_x
                mean_y = cur_y
            if nobs >= minp:
                if not bias:
                    numerator = sum_wt * sum_wt
                    denominator = numerator - sum_wt2
                    if denominator > 0:
                        out[s + i] = (numerator / denominator) * cov
                    else:
                        out[s + i] = NAN
                else:
                    out[s + i] = cov
            else:
                out[s + i] = NAN
    return out

# state: 0 nobs, 1 mean_x, 2 mean_y, 3 origin_x, 4 origin_y, 5 ssqdm_xy,
#        6 ssqdm_x, 7 ssqdm_y, 8 peak_x, 9 peak_y, 10 unstable(0/1)

def _cancel(st: List[float]) -> bool:
    xy = st[5]
    sx = st[6]
    sy = st[7]
    if xy != xy or sx != sx or sy != sy:
        return True
    return (st[8] * st[9] * _INV_COND_TOL > abs(xy) or st[8] * st[8] * _INV_COND_TOL > sx
            or st[9] * st[9] * _INV_COND_TOL > sy)

def _peak(shx: float, shy: float, st: List[float]):
    dx = abs(shx - st[1])
    dy = abs(shy - st[2])
    if dx > st[8]:
        st[8] = dx
    if dy > st[9]:
        st[9] = dy

def _add_xy(vx: float, vy: float, st: List[float]):
    if vx != vx or vy != vy:
        return
    st[0] += 1.0
    if st[0] == 1.0:
        st[3] = vx if abs(vx) < _MAX_ORIGIN else 0.0
        st[4] = vy if abs(vy) < _MAX_ORIGIN else 0.0
    shx = vx - st[3]
    shy = vy - st[4]
    dx = shx - st[1]
    dy = shy - st[2]
    st[1] += dx / st[0]
    st[2] += dy / st[0]
    st[5] += (shx - st[1]) * dy
    st[6] += (shx - st[1]) * dx
    st[7] += (shy - st[2]) * dy
    _peak(shx, shy, st)
    if _cancel(st):
        st[10] = 1.0

def _remove_xy(vx: float, vy: float, st: List[float]):
    if vx != vx or vy != vy:
        return
    shx = vx - st[3]
    shy = vy - st[4]
    dx = shx - st[1]
    dy = shy - st[2]
    st[0] -= 1.0
    if st[0] != 0.0:
        st[1] -= dx / st[0]
        st[2] -= dy / st[0]
        st[5] -= (shx - st[1]) * dy
        st[6] -= (shx - st[1]) * dx
        st[7] -= (shy - st[2]) * dy
        _peak(shx, shy, st)
        if _cancel(st):
            st[10] = 1.0
    else:
        for k in range(10):
            st[k] = 0.0
        st[10] = 0.0

def _roll_xy(x: List[float], y: List[float], start: List[int], end: List[int],
             minp: int, ddof: int, is_corr: bool) -> List[float]:
    N = len(start)
    minp = max(minp, 1)
    mono = _mono(start, end)
    out = [NAN for _ in range(N)]
    st = [0.0 for _ in range(11)]
    for i in range(N):
        s = start[i]
        e = end[i]
        recompute = i == 0 or not mono or s >= end[i - 1]
        if not recompute:
            for j in range(start[i - 1], s):
                _remove_xy(x[j], y[j], st)
            for j in range(end[i - 1], e):
                _add_xy(x[j], y[j], st)
        if recompute or st[10] != 0.0:
            for k in range(11):
                st[k] = 0.0
            for j in range(s, e):
                _add_xy(x[j], y[j], st)
            st[10] = 0.0
        nobs = st[0]
        if nobs < float(minp):
            out[i] = NAN
        elif is_corr:
            if nobs <= 0.0 or st[6] == 0.0 or st[7] == 0.0:
                out[i] = NAN
            else:
                den = (st[6] * st[7]) ** 0.5
                val = st[5] / den
                if val > 1.0:
                    val = 1.0
                elif val < -1.0:
                    val = -1.0
                out[i] = val
        else:
            out[i] = NAN if nobs <= float(ddof) else st[5] / (nobs - float(ddof))
        if not mono:
            for k in range(11):
                st[k] = 0.0
    return out

def roll_cov(x: List[float], y: List[float], start: List[int], end: List[int],
             minp: int, ddof: int = 1) -> List[float]:
    return _roll_xy(x, y, start, end, minp, ddof, False)

def roll_corr(x: List[float], y: List[float], start: List[int], end: List[int],
              minp: int) -> List[float]:
    return _roll_xy(x, y, start, end, minp, 1, True)

def roll_quantile(values: List[float], start: List[int], end: List[int], minp: int,
                  quantile: float, interpolation: str = "linear") -> List[float]:
    N = len(start)
    if quantile <= 0.0 or quantile >= 1.0:
        raise ValueError(f"quantile value {quantile} not in [0, 1]")
    if interpolation not in ("linear", "lower", "higher", "nearest", "midpoint"):
        raise ValueError(f"Interpolation '{interpolation}' is not supported")
    out = [NAN for _ in range(N)]
    for i in range(N):
        w = [values[j] for j in range(start[i], end[i]) if values[j] == values[j]]
        nobs = len(w)
        if nobs < minp:
            continue
        w.sort()
        if nobs == 0:
            continue
        if nobs == 1:
            out[i] = w[0]
            continue
        iwf = quantile * float(nobs - 1)
        idx = int(iwf)
        if iwf == float(idx):
            out[i] = w[idx]
            continue
        if interpolation == "linear":
            out[i] = w[idx] + (w[idx + 1] - w[idx]) * (iwf - float(idx))
        elif interpolation == "lower":
            out[i] = w[idx]
        elif interpolation == "higher":
            out[i] = w[idx + 1]
        elif interpolation == "nearest":
            fr = iwf - float(idx)
            if fr == 0.5:
                out[i] = w[idx] if idx % 2 == 0 else w[idx + 1]
            elif fr < 0.5:
                out[i] = w[idx]
            else:
                out[i] = w[idx + 1]
        else:
            out[i] = (w[idx] + w[idx + 1]) / 2.0
    return out

def get_center_of_mass(comass: Optional[float], span: Optional[float],
                       halflife: Optional[float], alpha: Optional[float]) -> float:
    import math
    cnt = 0
    if comass is not None: cnt += 1
    if span is not None: cnt += 1
    if halflife is not None: cnt += 1
    if alpha is not None: cnt += 1
    if cnt > 1:
        raise ValueError("comass, span, halflife, and alpha are mutually exclusive")
    if comass is not None:
        if comass < 0:
            raise ValueError("comass must satisfy: comass >= 0")
        return comass
    elif span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        return (span - 1.0) / 2.0
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1.0 - math.exp(math.log(0.5) / halflife)
        return 1.0 / decay - 1.0
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        return (1.0 - alpha) / alpha
    raise ValueError("Must pass one of comass, span, halflife, or alpha")
