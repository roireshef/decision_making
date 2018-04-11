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

temp_cost = cost.subs(s0, 0).subs(a0, 0).subs(aT, 0)
# temp_optimum = float(findroot(lambda x: sp.diff(temp_cost.subs(wJ, 200).subs(wT, 1), T).subs(T, x), [1, 15]))

temp_v_t = v_t.subs(s0, 0).subs(a0, 0).subs(aT, 0)
temp_a_t = sp.diff(temp_v_t, t)
result = cost.subs(s0, 0).subs(a0, 0).subs(aT, 0)

