# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Native Typhon implementation of a scipy.integrate subset.

from _quad import quad, dblquad, tplquad, fixed_quad, nquad, newton_cotes, IntegrationWarning, QuadInfo
from _quad_vec import quad_vec
from _quadrature import trapezoid, simpson, cumulative_trapezoid, cumulative_simpson, romb
from _tanhsinh import tanhsinh, nsum
from _ivp import solve_ivp, OdeSolution, OdeResult
from _odepack import odeint, ODEintWarning
from _ode import ode, complex_ode
from _cubature import cubature, CubatureResult, CubatureRegion
from _lebedev import lebedev_rule
from _qmc_quad import qmc_quad, QMCQuadResult
from _bvp import solve_bvp, BVPResult
from _odesolver import OdeSolver, DenseOutput, RK23, RK45, DOP853, Radau, BDF, LSODA
