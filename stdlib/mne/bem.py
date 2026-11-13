# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.bem subset: spherical conductor models (make_sphere_model with the
# Berg-Scherg equivalent-dipole fit) used by sphere-model forward solutions.

import numpy as np

FIFFV_COORD_HEAD: int = 4
HEAD_SIZE_DEFAULT: float = 0.095


class ConductorModel:
    """Spherical conductor model (mne.bem.ConductorModel with is_sphere=True)."""
    is_sphere: bool
    r0: np.ndarray[float, 1]
    coord_frame: int
    layer_rad: List[float]
    layer_rel_rad: List[float]
    layer_sigma: List[float]
    mu: np.ndarray[float, 1]
    lambda_: np.ndarray[float, 1]
    nfit: int

    def __init__(self, r0: np.ndarray[float, 1]):
        self.is_sphere = True
        self.r0 = r0
        self.coord_frame = FIFFV_COORD_HEAD
        self.layer_rad = List[float]()
        self.layer_rel_rad = List[float]()
        self.layer_sigma = List[float]()
        self.mu = np.zeros(0)
        self.lambda_ = np.zeros(0)
        self.nfit = 0

    @property
    def radius(self) -> Optional[float]:
        if len(self.layer_rad) == 0:
            return None
        return self.layer_rad[-1]

    @property
    def nlayers(self) -> int:
        return len(self.layer_rad)

    def copy(self):
        c = ConductorModel(self.r0.copy())
        c.layer_rad = list(self.layer_rad)
        c.layer_rel_rad = list(self.layer_rel_rad)
        c.layer_sigma = list(self.layer_sigma)
        c.mu = self.mu.copy()
        c.lambda_ = self.lambda_.copy()
        c.nfit = self.nfit
        return c

    def __getitem__(self, key: Literal[str]):
        if key == "r0":
            return self.r0
        elif key == "is_sphere":
            return self.is_sphere
        elif key == "mu":
            return self.mu
        elif key == "lambda":
            return self.lambda_
        elif key == "nfit":
            return self.nfit
        elif key == "coord_frame":
            return self.coord_frame
        else:
            compile_error("unsupported ConductorModel key")

    def __repr__(self) -> str:
        if len(self.layer_rad) == 0:
            return f"<ConductorModel | Sphere (no layers): r0=[{self.r0[0]:.1f} {self.r0[1]:.1f} {self.r0[2]:.1f}] mm>"
        return f"<ConductorModel | Sphere ({len(self.layer_rad)} layers): r0=[{1000 * self.r0[0]:.1f} {1000 * self.r0[1]:.1f} {1000 * self.r0[2]:.1f}] R={1000 * self.layer_rad[-1]:.0f} mm>"


def _multi_sphere_coeffs(rel_rads: List[float], sigmas: List[float], n_terms: int):
    """_fwd_eeg_get_multi_sphere_model_coeffs."""
    nlayer = len(rel_rads)
    c1 = np.zeros(nlayer - 1)
    c2 = np.zeros(nlayer - 1)
    cr = np.zeros(nlayer - 1)
    cr_mult = np.zeros(nlayer - 1)
    for k in range(nlayer - 1):
        c1[k] = sigmas[k] / sigmas[k + 1]
        c2[k] = c1[k] - 1.0
        cr_mult[k] = rel_rads[k]
        cr[k] = cr_mult[k]
        cr_mult[k] *= cr_mult[k]
    coeffs = np.zeros(n_terms - 1)
    for n in range(1, n_terms):
        for k in range(nlayer - 1):
            cr[k] *= cr_mult[k]
        M = np.eye(2)
        n1 = n + 1.0
        for k in range(nlayer - 2, -1, -1):
            A = np.array([[n + n1 * c1[k], n1 * c2[k] / cr[k]],
                          [n * c2[k] * cr[k], n1 + n * c1[k]]])
            M = np.dot(A, M)
        num = n * (2.0 * n + 1.0) ** (nlayer - 1)
        coeffs[n - 1] = num / (n * M[1, 1] + n1 * M[1, 0])
    return coeffs


class _BSFit:
    fn: np.ndarray[float, 1]
    w: np.ndarray[float, 1]
    nterms: int
    nfit: int

    def __init__(self, fn: np.ndarray[float, 1], w: np.ndarray[float, 1], nterms: int, nfit: int):
        self.fn = fn
        self.w = w
        self.nterms = nterms
        self.nfit = nfit

    def compose(self, mu):
        k1 = np.arange(1, self.nterms).astype(float)
        mu1ns = mu[0] ** k1
        y = self.w[:-1] * (self.fn[1:] - mu1ns * self.fn[0])
        M = self.w[:-1, None] * (mu[1:][None, :] ** k1[:, None] - mu1ns[:, None])
        uu, sing, vv = np.linalg.svd(M, full_matrices=False)
        ncomp = self.nfit - 1
        return y, uu[:, :ncomp].copy(), sing[:ncomp].copy(), vv[:ncomp].copy()

    def __call__(self, mu) -> float:
        if float(np.max(np.abs(mu))) >= 1.0:
            return 100.0
        y, uu, sing, vv = self.compose(mu)
        resi = y - np.dot(uu, np.dot(y, uu))
        return float(np.dot(resi, resi))


class _BSCons:
    max_: float

    def __init__(self, max_: float):
        self.max_ = max_

    def __call__(self, x):
        return self.max_ - np.abs(x)


def _fit_berg_scherg(rel_rads: List[float], sigmas: List[float], nterms: int, nfit: int):
    from scipy.optimize import fmin_cobyla
    fn = _multi_sphere_coeffs(rel_rads, sigmas, nterms + 1)
    f = min(rel_rads) / max(rel_rads)
    k = np.arange(1, nterms + 1).astype(float)
    w = np.sqrt((2.0 * k + 1) * (3.0 * k + 1.0) / k) * np.power(f, (k - 1.0))
    w[-1] = 0.0
    u = _BSFit(fn, w, nterms, nfit)
    catol = 1e-6
    mu = fmin_cobyla(u, np.zeros(3), [_BSCons(1.0 - 2 * catol)], rhobeg=0.5, rhoend=1e-4,
                     catol=catol)
    y, uu, sing, vv = u.compose(mu)
    vec = np.dot(y, uu)
    resi = y - np.dot(uu, vec)
    vec /= sing
    lam = np.zeros(nfit)
    lam[1:] = np.dot(vec, vv)
    lam[0] = fn[0] - np.sum(lam[1:])
    rv = float(np.dot(resi, resi) / np.dot(y, y))
    order = np.argsort(mu)[::-1]
    return mu[order].copy(), lam[order].copy(), rv


def make_sphere_model(r0 = (0.0, 0.0, 0.04), head_radius = 0.09, info = None,
                      relative_radii = (0.90, 0.92, 0.97, 1.0),
                      sigmas = (0.33, 1.0, 0.004, 0.33), verbose = None) -> ConductorModel:
    """mne.make_sphere_model (r0/head_radius 'auto' is not supported)."""
    if isinstance(r0, str) or isinstance(head_radius, str):
        compile_error("make_sphere_model: r0/head_radius='auto' is not supported in Typhon mne")
    rr = [float(x) for x in relative_radii]
    sg = [float(x) for x in sigmas]
    if len(rr) != len(sg):
        raise ValueError(f"relative_radii length ({len(rr)}) must match that of sigmas ({len(sg)})")
    if len(sg) <= 1 and not isinstance(head_radius, NoneType):
        raise ValueError(f"at least 2 sigmas must be supplied if head_radius is not None, got {len(sg)}")
    sphere = ConductorModel(np.array([float(x) for x in r0]))
    if not isinstance(head_radius, NoneType):
        hr = float(head_radius)
        order = np.argsort(np.array(rr))
        rr2 = [rr[int(order[i])] for i in range(len(rr))]
        sg2 = [sg[int(order[i])] for i in range(len(sg))]
        R = rr2[-1]
        for i in range(len(rr2)):
            rel = rr2[i] / R
            sphere.layer_rel_rad.append(rel)
            sphere.layer_sigma.append(sg2[i])
            sphere.layer_rad.append(hr * rel)
        mu, lam, rv = _fit_berg_scherg(sphere.layer_rel_rad, sphere.layer_sigma, 200, 3)
        sphere.mu = mu
        sphere.lambda_ = lam / sphere.layer_sigma[-1]
        sphere.nfit = 3
    return sphere


def _check_sphere(sphere) -> np.ndarray[float, 1]:
    """mne.utils.check._check_sphere for None | float | 4-tuple | ConductorModel."""
    if isinstance(sphere, NoneType):
        return np.array([0.0, 0.0, 0.0, HEAD_SIZE_DEFAULT])
    elif isinstance(sphere, ConductorModel):
        if len(sphere.layer_rad) == 0:
            raise ValueError("sphere, if a ConductorModel, must be spherical with multiple layers, not a BEM or single-layer sphere")
        return np.array([sphere.r0[0], sphere.r0[1], sphere.r0[2], sphere.layer_rad[0]])
    elif isinstance(sphere, float) or isinstance(sphere, int):
        return np.array([0.0, 0.0, 0.0, float(sphere)])
    else:
        a = np.array([float(x) for x in sphere])
        if a.shape[0] != 4:
            raise ValueError(f"sphere must be float or 1D array of shape (4,), got array-like of shape ({a.shape[0]},)")
        return a
