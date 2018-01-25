from sympy import *
from sympy.tensor.array import Array

t = symbols('t')
x = Function('x')(t)
y = Function('y')(t)

s = Array([x, y])

ds = simplify(s.diff())

ds_norm = sqrt(ds[0] ** 2 + ds[1] ** 2)

T = ds / ds_norm

dT = simplify(T.diff())

dT_norm = sqrt(dT[0] ** 2 + dT[1] ** 2)

k = simplify(dT_norm / ds_norm)

dk = simplify(k.diff())

k_tag = simplify(dk / ds_norm)

