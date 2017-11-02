from sympy import *

a0 = Symbol('a(0)')
aT = Symbol('a(T)')
amax = Symbol('a_max')
ti = Symbol('t_i')
t = Symbol('t')
T = Symbol('T')
v0 = Symbol('v(0)')
s0 = Symbol('s(0)')

sa_0 = Symbol('s_a(0)')
va_0 = Symbol('v_a(0)')

#first trapezoid

a1_t = a0 + (amax - a0) / ti * t

v1_t = integrate(a1_t, t) + v0

s1_t = integrate(v1_t, t) + s0

#second trapezoid

a2_t = amax + (aT - amax)/(T-ti) * (t - ti)

v2_t = v1_t.subs(t, ti) + integrate(a2_t, (t, ti, t))

s2_T = s1_t.subs(t, ti) + integrate(v2_t, (t, ti, T))

sa_T = sa_0 + va_0 * T








a0 = Symbol('a0')
a1 = Symbol('a1')
a2 = Symbol('a2')
a3 = Symbol('a3')
a4 = Symbol('a4')
a5 = Symbol('a5')
t = Symbol('t')

x_t = a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3 + a2 * t ** 2 + a1 * t + a0



