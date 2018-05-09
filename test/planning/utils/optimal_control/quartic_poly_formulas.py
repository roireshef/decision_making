import sympy as sp
from mpmath import findroot
from sympy import symbols
from sympy.matrices import *

T = symbols('T')
t = symbols('t')
Tm = symbols('T_m')  # safety margin in seconds

s0, v0, a0, vT, aT = symbols('s_0 v_0 a_0 v_T a_T')

A = Matrix([
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 2, 0, 0],
    [4*T**3, 3*T**2, 2*T, 1, 0],
    [12*T**2, 6*T, 2, 0, 0]
])

# solve to get solution
# this assumes a0=aT==0 (constant velocity)
[c4, c3, c2, c1, c0] = A.inv() * Matrix([s0, v0, a0, vT, aT])

x_t = (c4 * t**4 + c3 * t**3 + c2 * t**2 + c1 * t + c0).simplify()
v_t = sp.diff(x_t, t).simplify()
a_t = sp.diff(v_t, t).simplify()
j_t = sp.diff(a_t, t).simplify()

J = sp.integrate(j_t ** 2, (t, 0, T)).simplify()

wJ, wT = symbols('w_J w_T')

cost = (wJ * J + wT * T).simplify()

cost = cost.subs(s0, 0).subs(aT, 0).simplify()
cost_diff = sp.diff(cost, T).simplify()

temp_x_t = x_t.subs(s0, 0).subs(aT, 0).simplify()
temp_v_t = v_t.subs(s0, 0).subs(aT, 0).simplify()
temp_a_t = sp.diff(temp_v_t, t).simplify()

cost_desmos = cost.subs(a0, 0).simplify()
cost_diff_desmos = cost_diff.subs(a0, 0).simplify()
x_t_desmos = temp_x_t.subs(a0, 0).simplify()
v_t_desmos = temp_v_t.subs(a0, 0).simplify()
a_t_desmos = temp_a_t.subs(a0, 0).simplify()
