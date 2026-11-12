# Generates expected literals for test_integrate_ivp.py (scipy 1.18.1).
# Run: /Users/mind/.jcode/scratch/.venv/bin/python test/scipy/gen_integrate_ivp_expected.py
import warnings
warnings.simplefilter("ignore")
import numpy as np
from scipy import integrate as I

def lit(v):
    return np.asarray(v, dtype=float).tolist()

def rob(t, y):
    return [-0.04*y[0] + 1e4*y[1]*y[2], 0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]*y[1], 3e7*y[1]*y[1]]
def robj(t, y):
    return [[-0.04, 1e4*y[2], 1e4*y[1]],
            [0.04, -1e4*y[2] - 6e7*y[1], -1e4*y[1]],
            [0.0, 6e7*y[1], 0.0]]
def vdp(t, y): return [y[1], 1000.0*(1 - y[0]*y[0])*y[1] - y[0]]
def vdpj(t, y): return [[0.0, 1.0], [-2000.0*y[0]*y[1] - 1.0, 1000.0*(1 - y[0]*y[0])]]
def lin(t, y): return [-2.0*y[0] + y[1], y[0] - 2.0*y[1]]
LINJ = [[-2.0, 1.0], [1.0, -2.0]]
def vdecay(t, y): return -0.5 * y   # vectorized-safe
def cannon(t, y): return [y[1], -0.5]
def hit(t, y): return y[0]
hit.terminal = True; hit.direction = -1

def decay(t, y): return -0.5 * y
def osc(t, y): return [y[1], -y[0]]
def lv(t, z, a, b, c, d): return [a*z[0] - b*z[0]*z[1], -c*z[1] + d*z[0]*z[1]]
def apex(t, y): return y[1]
def litr(v): return repr(np.asarray(v, dtype=float).tolist())
print('IVP0 = [')
for m in ['RK45', 'RK23', 'DOP853']:
    s = I.solve_ivp(decay, [0, 10], [2, 4, 8], method=m)
    print(f'    ({m!r}, {s.nfev}, {litr(s.t)}, {litr(s.y)}),')
print(']')
print('IVP1 = [')
for m in ['RK45', 'RK23', 'DOP853']:
    s = I.solve_ivp(osc, (0.0, 20.0), [1.0, 0.0], method=m, rtol=1e-8, atol=1e-10)
    print(f'    ({m!r}, {s.nfev}, {litr(s.t[-3:])}, {litr(s.y[:, -1])}),')
print(']')
s = I.solve_ivp(lv, [0, 15], [10, 5], args=(1.5, 1, 3, 1), dense_output=True)
print('LV =', (s.nfev, s.t.size, s.y[:, -1].tolist(), s.sol(7.3).tolist(), s.sol([0.5, 14.0]).tolist()))
s = I.solve_ivp(decay, [0, 10], [2.0], t_eval=[0, 1, 2.5, 10])
print('TEVAL =', (s.nfev, s.t.tolist(), s.y.tolist()))
s = I.solve_ivp(cannon, [0, 100], [0, 10], events=(hit, apex))
print('EVENTS =', (s.status, s.nfev, float(s.t[-1]), s.y[:, -1].tolist(), s.t_events[0].tolist(), s.t_events[1].tolist(), s.y_events[1].tolist()))
s = I.solve_ivp(decay, [10, 0], [2.0], method='DOP853', dense_output=True, t_eval=[9.0, 3.0])
print('BACK =', (s.nfev, s.t.tolist(), s.y.tolist(), s.sol(4.0).tolist()))
s = I.solve_ivp(osc, [0, 5], [1.0, 0.0], method='RK23', dense_output=True, max_step=0.1, atol=[1e-6, 1e-8], first_step=0.01)
print('OPTS =', (s.nfev, s.t.size, s.y[:, -1].tolist(), s.sol(2.345).tolist()))
s = I.solve_ivp(osc, [0, 5], [1.0, 0.0], method='DOP853', dense_output=True)
print('DOPDENSE =', (s.nfev, s.sol(1.234).tolist(), s.sol(4.9).tolist()))
s = I.solve_ivp(decay, [0, 0], [1.0])
print('ZERO =', (s.nfev, s.t.tolist(), s.y.tolist(), s.status))

print('ROB = [')
for m in ['Radau', 'BDF', 'LSODA']:
    s = I.solve_ivp(rob, [0.0, 1e5], [1.0, 0.0, 0.0], method=m, rtol=1e-6, atol=1e-10)
    print(f'    ({m!r}, {s.nfev}, {s.njev}, {s.nlu}, {s.t.size}, {lit(s.y[:, -1])}),')
print(']')
print('ROBJ = [')
for m in ['Radau', 'BDF', 'LSODA']:
    s = I.solve_ivp(rob, [0.0, 1e5], [1.0, 0.0, 0.0], method=m, rtol=1e-6, atol=1e-10, jac=robj)
    print(f'    ({m!r}, {s.nfev}, {s.njev}, {s.nlu}, {s.t.size}, {lit(s.y[:, -1])}),')
print(']')
print('VDP = [')
for m in ['Radau', 'BDF', 'LSODA']:
    s = I.solve_ivp(vdp, [0.0, 3000.0], [2.0, 0.0], method=m, jac=vdpj, dense_output=True)
    print(f'    ({m!r}, {s.nfev}, {s.njev}, {s.nlu}, {s.t.size}, {lit(s.y[:, -1])}, {lit(s.sol(1234.5))}, {lit(s.sol([10.0, 2999.0]))}),')
print(']')
print('LIN = [')
for m in ['Radau', 'BDF', 'LSODA']:
    s = I.solve_ivp(lin, [0.0, 5.0], [1.0, 0.5], method=m, jac=LINJ, t_eval=[0.5, 1.0, 4.0])
    print(f'    ({m!r}, {s.nfev}, {s.njev}, {s.nlu}, {lit(s.y)}),')
print(']')
print('VEC = [')
for m in ['Radau', 'BDF', 'RK45']:
    s = I.solve_ivp(vdecay, [0.0, 4.0], [1.0, 2.0], method=m, vectorized=True)
    print(f'    ({m!r}, {s.nfev}, {s.njev}, {lit(s.y[:, -1])}),')
print(']')
print('EVI = [')
for m in ['Radau', 'BDF', 'LSODA']:
    s = I.solve_ivp(cannon, [0.0, 100.0], [0.0, 10.0], method=m, events=hit)
    print(f'    ({m!r}, {s.status}, {s.nfev}, {lit(s.t_events[0])}, {lit(s.y[:, -1])}),')
print(']')
s = I.solve_ivp(vdp, [3000.0, 0.0], [2.0, 0.0], method='BDF', jac=vdpj, first_step=1e-4, max_step=50.0)
print('BACKBDF =', (s.nfev, s.njev, s.nlu, s.t.size, s.y[:, -1].tolist()))
s = I.solve_ivp(lin, [0.0, 5.0], [1.0, 0.5], method='LSODA', min_step=1e-3, max_step=0.5, first_step=0.01)
print('LSOPT =', (s.nfev, s.t.size, s.y[:, -1].tolist()))

def rob2(y, t): return rob(t, y)
def robj2(y, t): return robj(t, y)
y, info = I.odeint(rob2, [1.0, 0.0, 0.0], [0.0, 1.0, 10.0, 100.0, 1e4], full_output=1, rtol=1e-6, atol=1e-10)
print('ODE_ROB =', (lit(y), info['nst'].tolist(), info['nfe'].tolist(), info['nje'].tolist(), info['mused'].tolist(), lit(info['hu']), lit(info['tcur']), info['nqu'].tolist(), lit(info['tsw'])))
y, info = I.odeint(rob2, [1.0, 0.0, 0.0], [0.0, 1.0, 1e3], Dfun=robj2, full_output=1, rtol=1e-8, atol=1e-12)
print('ODE_ROBJ =', (lit(y), info['nst'].tolist(), info['nfe'].tolist(), info['nje'].tolist()))
def osc(y, t, w): return [y[1], -w*w*y[0]]
t = np.linspace(0.0, 10.0, 11)
y, info = I.odeint(osc, [1.0, 0.0], t, args=(2.0,), full_output=1)
print('ODE_OSC =', (lit(y), info['nst'].tolist()[-1], info['nfe'].tolist()[-1], info['mused'].tolist()))
def osct(t, y, w): return [y[1], -w*w*y[0]]
y = I.odeint(osct, [1.0, 0.0], [0.0, 1.0, 2.0], args=(2.0,), tfirst=True, rtol=1e-10, atol=1e-12)
print('ODE_TF =', lit(y))
y, info = I.odeint(osc, [1.0, 0.0], [0.0, 2.0, 4.0, 6.0], args=(3.0,), tcrit=[3.0], hmax=0.5, full_output=1)
print('ODE_CRIT =', (lit(y), info['nst'].tolist(), lit(info['tcur'])))
def bandf(y, t): return [-y[0] + y[1], -2.0*y[1] + y[2], -3.0*y[2]]
def bandj(y, t): return [[0.0, 1.0, 1.0], [-1.0, -2.0, -3.0]]   # ml=0, mu=1 packed
y, info = I.odeint(bandf, [1.0, 1.0, 1.0], [0.0, 1.0, 2.0], Dfun=bandj, ml=0, mu=1, full_output=1)
print('ODE_BAND =', (lit(y), info['nst'].tolist(), info['nje'].tolist()))
y, info = I.odeint(bandf, [1.0, 1.0, 1.0], [0.0, 1.0, 2.0], ml=0, mu=1, full_output=1)
print('ODE_BAND5 =', (lit(y), info['nst'].tolist(), info['nfe'].tolist()))
y, info = I.odeint(rob2, [1.0, 0.0, 0.0], [0.0, 1e5], mxstep=50, full_output=1)
print('ODE_MX =', (lit(y), info['message']))

def f(t, y, a): return [y[1], -a*y[0]]
def jac(t, y, a): return [[0.0, 1.0], [-a, 0.0]]
r = I.ode(f, jac).set_integrator('lsoda', rtol=1e-8, atol=1e-10)
r.set_initial_value([1.0, 0.0], 0.0).set_f_params(2.0).set_jac_params(2.0)
out = []
while r.successful() and r.t < 5.0:
    out.append(r.integrate(r.t + 1.0).tolist())
print('ODE_LSODA =', (out, r.t, r.get_return_code()))
r = I.ode(f).set_integrator('lsoda')
r.set_initial_value([1.0, 0.0], 0.0).set_f_params(2.0)
print('ODE_LSODA2 =', r.integrate(3.0).tolist())
