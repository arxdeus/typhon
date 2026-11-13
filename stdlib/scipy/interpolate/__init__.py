# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Native Typhon implementation of scipy.interpolate.  CubicSpline,
# PchipInterpolator, Akima1DInterpolator and CubicHermiteSpline are PPoly
# subclasses as in scipy.

from _ppoly import PPoly
from _cubic import CubicHermiteSpline, PchipInterpolator, pchip_interpolate, \
    Akima1DInterpolator, CubicSpline
from _bspl import BSpline
from _fitpack import make_interp_spline, splrep, splev, splint, splder, splantider, \
    splprep, sproot, spalde, insert
from _interpolate import interp1d, RegularGridInterpolator, interpn
from _fitpack2 import UnivariateSpline, InterpolatedUnivariateSpline, LSQUnivariateSpline
from _polyint import KroghInterpolator, BarycentricInterpolator, krogh_interpolate, \
    barycentric_interpolate, approximate_taylor_polynomial, lagrange, pade
from _rbfinterp import RBFInterpolator
from _bsplines2 import make_lsq_spline, make_smoothing_spline, make_splrep, make_splprep, \
    generate_knots
from _ndgriddata import LinearNDInterpolator, CloughTocher2DInterpolator, NearestNDInterpolator, griddata
from _bpoly import BPoly
from _fitpack2d import RectBivariateSpline, bisplev, SmoothBivariateSpline, LSQBivariateSpline, bisplrep
from _rbf import Rbf
from _ndppoly import NdPPoly
pchip = PchipInterpolator
BivariateSpline = RectBivariateSpline
