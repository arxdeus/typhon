# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Signal space separation core: multipolar (spherical harmonic) basis,
# regularization of the internal space and the decomposition operators.
# Typhon Info carries no dev_head_t: the device and head frames coincide.

import math
import numpy as np
from .._info import Info, _picks_to_idx
from ..forward import _meg_coils, _Coils
from ..channels import _auto_origin


def _get_n_moments(order: int) -> int:
    return (order + 2) * order


def _deg_ord_idx(deg: int, order: int) -> int:
    return deg * deg + deg + order - 1


def _integ1(ca: float, sa: float, cp: float, sp: float, br: float, baz: float, bpol: float,
            c0: float, c1: float, c2: float) -> float:
    gx = sp * ca * br + cp * ca * bpol - sa * baz
    gy = sp * sa * br + cp * sa * bpol + ca * baz
    gz = cp * br - sp * bpol
    return gx * c0 + gy * c1 + gz * c2


def _sss_basis(rmags: np.ndarray[float, 2], cosmags: np.ndarray[float, 2], bins: np.ndarray[int, 1],
               n_coils: int, origin: np.ndarray[float, 1], int_order: int, ext_order: int):
    """SSS basis (n_coils, n_in + n_out); cosmags already weighted (mne _sss_basis)."""
    n_in = _get_n_moments(int_order)
    n_out = _get_n_moments(ext_order)
    S = np.zeros((n_coils, n_in + n_out))
    mo = max(int_order, ext_order)
    for p in range(rmags.shape[0]):
        x = rmags[p, 0] - origin[0]
        y = rmags[p, 1] - origin[1]
        z = rmags[p, 2] - origin[2]
        c0 = cosmags[p, 0]
        c1 = cosmags[p, 1]
        c2 = cosmags[p, 2]
        b = int(bins[p])
        r_n = math.sqrt(x * x + y * y + z * z)
        xc = z / r_n
        L = np.zeros((mo + 1, mo + 2))
        L[0, 0] = 1.0
        pnn = 1.0
        fact = 1.0
        sx2 = math.sqrt((1.0 - xc) * (1.0 + xc))
        for degree in range(mo + 1):
            L[degree, degree] = pnn
            pnn *= -fact * sx2
            fact += 2.0
            if degree < mo:
                L[degree + 1, degree] = xc * (2 * degree + 1) * L[degree, degree]
            if degree >= 2:
                for order in range(degree - 1):
                    L[degree, order] = (xc * (2 * degree - 1) * L[degree - 1, order]
                                        - (degree + order - 1) * L[degree - 2, order]) / (degree - order)
        phi = math.atan2(y, x)
        r_xy = math.sqrt(x * x + y * y)
        cos_pol = z / r_n
        sin_pol = math.sqrt(1.0 - cos_pol * cos_pol)
        z_only = r_xy <= 1e-16
        sin_pol_nz = sin_pol
        if z_only:
            sin_pol_nz = 1.0
            r_xy = 1.0
        cos_az = x / r_xy
        sin_az = y / r_xy
        if z_only:
            cos_az = 1.0
            sin_az = 0.0
        r_nn2 = r_n
        r_nn1 = 1.0 / (r_n * r_n)
        for degree in range(mo + 1):
            if degree <= ext_order:
                r_nn1 *= r_n
            if degree <= int_order:
                r_nn2 *= r_n
            mult = 2e-7 * math.sqrt((2 * degree + 1) * math.pi)
            if degree > 0:
                idx = _deg_ord_idx(degree, 0)
                if degree <= int_order:
                    b_r = mult * (degree + 1) * L[degree, 0] / r_nn2
                    b_pol = -mult * L[degree, 1] / r_nn2
                    S[b, idx] += _integ1(cos_az, sin_az, cos_pol, sin_pol, b_r, 0.0, b_pol, c0, c1, c2)
                if degree <= ext_order:
                    b_r = -mult * degree * L[degree, 0] * r_nn1
                    b_pol = -mult * L[degree, 1] * r_nn1
                    S[b, n_in + idx] += _integ1(cos_az, sin_az, cos_pol, sin_pol, b_r, 0.0, b_pol, c0, c1, c2)
            for order in range(1, degree + 1):
                ord_phi = order * phi
                sin_order = math.sin(ord_phi)
                cos_order = math.cos(ord_phi)
                mult /= math.sqrt((degree - order + 1) * (degree + order))
                factor = mult * math.sqrt(2.0)
                lo = L[degree, order]
                dl = L[degree, order + 1] - (degree + order) * (degree - order + 1) * L[degree, order - 1]
                # cosine (positive order) terms
                idx = _deg_ord_idx(degree, order)
                r_fact = factor * lo * cos_order
                az_fact = factor * order * sin_order * lo
                pol_fact = -factor * dl * cos_order
                if degree <= int_order:
                    b_r = (degree + 1) * r_fact / r_nn2
                    b_az = az_fact / (sin_pol_nz * r_nn2)
                    if z_only:
                        b_az = 0.0
                    b_pol = pol_fact / (2 * r_nn2)
                    S[b, idx] += _integ1(cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol, c0, c1, c2)
                if degree <= ext_order:
                    b_r = -degree * r_fact * r_nn1
                    b_az = az_fact * r_nn1 / sin_pol_nz
                    if z_only:
                        b_az = 0.0
                    b_pol = pol_fact * r_nn1 / 2.0
                    S[b, n_in + idx] += _integ1(cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol, c0, c1, c2)
                # sine (negative order) terms
                idx = _deg_ord_idx(degree, -order)
                r_fact = factor * lo * sin_order
                az_fact = factor * order * cos_order * lo
                pol_fact = factor * dl * sin_order
                if degree <= int_order:
                    b_r = -(degree + 1) * r_fact / r_nn2
                    b_az = az_fact / (sin_pol_nz * r_nn2)
                    if z_only:
                        b_az = 0.0
                    b_pol = pol_fact / (2 * r_nn2)
                    S[b, idx] += _integ1(cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol, c0, c1, c2)
                if degree <= ext_order:
                    b_r = degree * r_fact * r_nn1
                    b_az = az_fact * r_nn1 / sin_pol_nz
                    if z_only:
                        b_az = 0.0
                    b_pol = pol_fact * r_nn1 / 2.0
                    S[b, n_in + idx] += _integ1(cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol, c0, c1, c2)
    return S


class _MFCoils:
    """Integration points of the MEG coils with weighted normals (mne _prep_mf_coils)."""
    rmags: np.ndarray[float, 2]
    cosmags: np.ndarray[float, 2]
    bins: np.ndarray[int, 1]
    n_coils: int

    def __init__(self, rmags, cosmags, bins, n_coils: int):
        self.rmags = rmags
        self.cosmags = cosmags
        self.bins = bins
        self.n_coils = n_coils

    def shifted(self, t: np.ndarray[float, 1]):
        r = self.rmags.copy()
        for i in range(r.shape[0]):
            for k in range(3):
                r[i, k] += t[k]
        return _MFCoils(r, self.cosmags, self.bins, self.n_coils)


def _prep_mf_coils(info: Info, picks: List[int]) -> _MFCoils:
    c = _meg_coils(info, picks)
    cm = c.cosmags.copy()
    for i in range(cm.shape[0]):
        w = c.ws[i]
        for k in range(3):
            cm[i, k] *= w
    return _MFCoils(c.rmags, cm, c.bins, len(picks))


def _basis(c: _MFCoils, origin: np.ndarray[float, 1], int_order: int, ext_order: int):
    return _sss_basis(c.rmags, c.cosmags, c.bins, c.n_coils, origin, int_order, ext_order)


def _col_norm_pinv(x: np.ndarray[float, 2]):
    nr = x.shape[1]
    norm = np.empty(nr)
    for j in range(nr):
        s = 0.0
        for i in range(x.shape[0]):
            s += x[i, j] * x[i, j]
        norm[j] = math.sqrt(s)
    for j in range(nr):
        for i in range(x.shape[0]):
            x[i, j] /= norm[j]
    u, s, v = np.linalg.svd(x, full_matrices=False)
    for j in range(nr):
        for i in range(v.shape[0]):
            v[i, j] /= norm[j]
    vs = v.T.copy()
    for j in range(vs.shape[1]):
        for i in range(vs.shape[0]):
            vs[i, j] = vs[i, j] * 1.0 / s[j]
    return np.dot(vs, u.T), s


def _get_degrees_orders(order: int) -> Tuple[List[int], List[int]]:
    n = _get_n_moments(order)
    degs = [0 for _ in range(n)]
    ords = [0 for _ in range(n)]
    for degree in range(1, order + 1):
        for o in range(degree + 1):
            ii = _deg_ord_idx(degree, o)
            degs[ii] = degree
            ords[ii] = o
            ii = _deg_ord_idx(degree, -o)
            degs[ii] = degree
            ords[ii] = -o
    return degs, ords


def _a_power(d: float) -> float:
    r_in = 0.080
    rho_i = 5.91107375632e-07
    return rho_i * rho_i * (d * r_in ** (2.0 * d + 4.0) / ((2.0 * d + 1.0) * (2.0 * d + 1.0) * (d + 1.0)))


def _regularize_out(int_order: int, ext_order: int, any_mag_or_fine: bool) -> List[int]:
    n_in = _get_n_moments(int_order)
    out = List[int]()
    if ext_order > 0 and not any_mag_or_fine:
        for k in range(n_in, n_in + 3):
            out.append(k)
    return out


def _regularize_in(int_order: int, ext_order: int, S_decomp: np.ndarray[float, 2],
                   any_mag_or_fine: bool, argmax: bool):
    n_in = _get_n_moments(int_order)
    degs, ords = _get_degrees_orders(int_order)
    a_lm_sq = [_a_power(float(degs[i])) for i in range(n_in)]
    I_tots = np.zeros(n_in)
    in_keepers = [i for i in range(n_in)]
    out_removes = _regularize_out(int_order, ext_order, any_mag_or_fine)
    out_keepers = List[int]()
    for k in range(n_in, S_decomp.shape[1]):
        if k not in out_removes:
            out_keepers.append(k)
    remove_order = List[int]()
    S = S_decomp.copy()
    nc = S.shape[1]
    use_norm = np.empty(nc)
    for j in range(nc):
        s = 0.0
        for i in range(S.shape[0]):
            s += S[i, j] * S[i, j]
        use_norm[j] = math.sqrt(s)
    for j in range(nc):
        for i in range(S.shape[0]):
            S[i, j] /= use_norm[j]
    noise_lev = 5e-13 * 5e-13
    for ii in range(n_in):
        cols = in_keepers + out_keepers
        this_S = np.empty((S.shape[0], len(cols)))
        for a in range(len(cols)):
            for i in range(S.shape[0]):
                this_S[i, a] = S[i, cols[a]]
        u, s, vt = np.linalg.svd(this_S, full_matrices=False)
        nk = len(in_keepers)
        k = s.shape[0]
        vs = np.empty((nk, k))
        for a in range(nk):
            nrm = use_norm[in_keepers[a]]
            for q in range(k):
                vs[a, q] = vt[q, a] / nrm * 1.0 / s[q]
        E = np.dot(vs, u.T)
        snr_min = float("inf")
        arg = 0
        tot = 0.0
        for a in range(nk):
            ssum = 0.0
            for i in range(E.shape[1]):
                ssum += E[a, i] * E[a, i]
            eta = ssum * noise_lev
            if ords[in_keepers[a]] == 0:
                eta *= 2.0
            eta *= 0.0025
            snr = a_lm_sq[in_keepers[a]] / eta
            tot += math.log2(snr + 1.0)
            if snr < snr_min:
                snr_min = snr
                arg = a
        I_tots[ii] = 0.5 * tot
        remove_order.append(in_keepers[arg])
        del in_keepers[arg]
        if ii > 10:
            mx = float(np.max(I_tots))
            if I_tots[ii - 1] < 0.95 * mx and I_tots[ii] < 0.95 * mx:
                break
    in_removes = List[int]()
    if n_in > 0:
        max_info = float(np.max(I_tots))
        lim_idx = 0
        if argmax:
            lim_idx = int(np.argmax(I_tots))
        else:
            for ii in range(n_in):
                if I_tots[ii] >= 0.98 * max_info:
                    lim_idx = ii
                    break
        for k in range(lim_idx):
            in_removes.append(remove_order[k])
    return in_removes, out_removes
