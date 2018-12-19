from math import pi

import numpy as np
import time

J = np.exp(2j * pi / 3)
Jc = 1 / J


def roots2(a, b, c):
    bp = b / 2
    delta = bp * bp - a * c
    u1 = (-bp - delta ** .5) / a
    u2 = -u1 - b / a
    return u1, u2


def cardan(a, b, c, d):
    # u = np.empty(2, np.complex128)
    z0 = b / 3 / a
    a2, b2 = a * a, b * b
    p = -b2 / 3 / a2 + c / a
    q = (b / 27 * (2 * b2 / a2 - 9 * c / a) + d) / a
    D = -4 * p * p * p - 27 * q * q
    r = np.sqrt(-D / 27 + 0j)
    u = ((-q+r)/2)**(1/3)
    v = -p / (3 * u)
    # v = ((-q + r) / 2) ** 0.33333333333333333333333
    # w = u * v
    # w0 = np.abs(w + p / 3)
    # w1 = np.abs(w * J + p / 3)
    # w2 = np.abs(w * Jc + p / 3)
    # w1_gt_w0_gt_w2 = np.logical_and(w1 > w0, w0 > w2)
    # w2_lt_w1_let_w0 = np.logical_and(w2 < w1, w1 <= w0)
    # v[np.logical_or(w1_gt_w0_gt_w2, w2_lt_w1_let_w0)] *= Jc
    # v[np.logical_and(w2 >= w1, w1 <= w0)] *= J
    return u + v - z0, u * J + v / J - z0, u / J + v * J - z0


# def cardan(a,b,c,d):
#     #"resolve P=ax^3+bx^2+cx+d=0"
#     #"x=z-b/a/3=z-z0 => P=z^3+pz+q"
#     z0=b/3/a
#     a2,b2 = a*a,b*b
#     p=-b2/3/a2 +c/a
#     q=(b/27*(2*b2/a2-9*c/a)+d)/a
#     D=-4*p*p*p-27*q*q+0j
#     r=np.sqrt(-D/27)
#     J=-0.5+0.86602540378443871j # exp(2i*pi/3)
#     u=((-q+r)/2)**(1/3)
#     v=((-q-r)/2)**(1/3)
#     return u+v-z0,u*J+v/J-z0,u/J+v*J-z0


def ferrari(coefs):
    "resolution of P=ax^4+bx^3+cx^2+dx+e=0"
    "CN all coeffs real."
    "First shift : x= z-b/4/a  =>  P=z^4+pz^2+qz+r"
    a, c, d, e = coefs[:, 0], coefs[:, 2], coefs[:, 3], coefs[:, 4]
    p = c / a
    q = d / a
    r = e / a
    "Second find X so P2=AX^3+BX^2+C^X+D=0"
    A = 8
    B = -4 * p
    C = -8 * r
    D = 4 * r * p - q * q
    y0, y1, y2 = cardan(A, B, C, D)
    y0_imag, y1_imag, y2_imag = np.abs(np.imag(y0)), np.abs(np.imag(y1)), np.abs(np.imag(y2))
    y0[y1_imag < y0_imag] = y1[y1_imag < y0_imag]
    y0[y2_imag < y0_imag] = y2[y2_imag < y0_imag]
    a0 = np.sqrt(-p + 2 * y0.real)
    b0 = np.empty(a0.shape)
    a0_zero_pos = (a0 == 0)
    a0_nonzero_pos = np.logical_not(a0_zero_pos)
    b0[a0_zero_pos] = y0[a0_zero_pos] ** 2 - r[a0_zero_pos]
    b0[a0_nonzero_pos] = -q[a0_nonzero_pos] / 2 / a0[a0_nonzero_pos]
    r0, r1 = roots2(1, a0, y0 + b0)
    r2, r3 = roots2(1, -a0, y0 - b0)
    return r0, r1, r2, r3

# start_time = time.time()
# k_numpy = np.sort(np.array(np.roots([1, 0, 1, 1, 1])))
# print('numpy took: %.9f' % (time.time() - start_time))
#
# start_time = time.time()
# k_ferrari = np.sort(np.array(ferrari(1, 0, 1, 1, 1)))
# print('ferrari took: %.9f' % (time.time() - start_time))
#
# print('error is: %f' % (np.power(np.abs(np.sum(k_ferrari - k_numpy)), 2)))
