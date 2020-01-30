from decision_making.src.planning.behavioral.planner.lane_change_planner import LaneChangePlanner
from decision_making.src.planning.utils.optimal_control.poly1d import Poly1D, QuinticPoly1D, QuarticPoly1D
import numpy as np


def test_areDerivativesInLimits_firstDegreePolynomialNotInLimits_returnFalse():
    poly_coefs = np.array([[1, 3]])
    T_vals = np.array([10])
    limits = np.array([0, 3])
    is_in_limits = Poly1D.are_derivatives_in_limits(0, poly_coefs, T_vals, limits=limits)

    assert not is_in_limits


def test_zipPolyvalWithDerivatives_equalsToPolyValWithDerivates():

    poly_coefs = np.array([[0.1, 0.3], [0.2, 0.6]])
    T_vals = np.array([[10, 8], [7, 6]])

    results1 = Poly1D.polyval_with_derivatives(poly_coefs[0:1], T_vals[0])
    results2 = Poly1D.polyval_with_derivatives(poly_coefs[1:2], T_vals[1])

    results = Poly1D.zip_polyval_with_derivatives(poly_coefs, T_vals)
    assert np.array_equal(np.concatenate((results1, results2)), results)


def test_areDerivativesInLimits_firstDegreePolynomialInLimits_returnTrue():
    poly_coefs = np.array([[0.1, 0.3]])
    T_vals = np.array([10])
    limits = np.array([0, 3])
    is_in_limits = Poly1D.are_derivatives_in_limits(0, poly_coefs, T_vals, limits=limits)

    assert is_in_limits


def test_areDerivativesInLimits_constantNotInLimits_returnFalse():
    poly_coefs = np.array([[20]])
    T_vals = np.array([10])
    limits = np.array([0, 3])
    is_in_limits = Poly1D.are_derivatives_in_limits(0, poly_coefs, T_vals, limits=limits)

    assert not is_in_limits


def test_areDerivativesInLimits_constantInLimits_returnTrue():
    poly_coefs = np.array([[0.3]])
    T_vals = np.array([10])
    limits = np.array([0, 3])
    is_in_limits = Poly1D.are_derivatives_in_limits(0, poly_coefs, T_vals, limits=limits)

    assert is_in_limits


def test_inverseTimeConstraintsTensor_compareWithLingAlgInv_isClose():
    T_vals = np.array([1, 10])
    A1_quintic_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T_vals))
    A2_quintic_inv = QuinticPoly1D.inverse_time_constraints_tensor(T_vals)
    assert np.isclose(A1_quintic_inv, A2_quintic_inv).all()
    A1_quartic_inv = np.linalg.inv(QuarticPoly1D.time_constraints_tensor(T_vals))
    A2_quartic_inv = QuarticPoly1D.inverse_time_constraints_tensor(T_vals)
    assert np.isclose(A1_quartic_inv, A2_quartic_inv).all()


def test_create_braking_actions():
    T1, s1 = LaneChangePlanner._create_triple_cubic_actions(10., 0.5, 9., -4., -5., 5.)
    T2, s2 = LaneChangePlanner._create_quartic_actions(10., 0.5, 9., -4., 5.)
    print(T1, T2, 's:', s1, s2)


import sympy as sp
from sympy import symbols
from sympy.matrices import *
from typing import List
import matplotlib.pyplot as plt
from sympy.solvers import solve

def test_():
    T = symbols('T')
    t = symbols('t')
    d = symbols('d')
    D = symbols('D')
    e = symbols('e')
    E = symbols('E')
    c = symbols('c')

    #T = 3
    #d = 3.6
    A = Matrix([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
        [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
        [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0]]
    )
    [c5, c4, c3, c2, c1, c0] = A.inv() * Matrix([0, 0, 0, d, 0, 0])

    x_t = (c5 * t ** 5 + c4 * t ** 4 + c3 * t ** 3 + c2 * t ** 2 + c1 * t + c0).simplify()
    v_t = sp.diff(x_t, t).simplify()
    a_t = sp.diff(v_t, t).simplify()
    j_t = sp.diff(a_t, t).simplify()
    j_max = j_t.subs(t, 0)

    solution = solve(2*E**4 - e*E**3 - c, E)
    solution = solve((D - 0.5)**3 * (2*D - d) - c, D)

    # constraints = np.array([0, 0, 0, d, 0, 0])
    # QuinticPoly1D.solve(A_inv, np.array([constraints]))[0]
    # v_t = np.polyder(x_t)
    # a_t = np.polyder(v_t)
    # j_t = np.polyder(a_t)
    times = np.arange(0, T + 0.001, 0.1)
    a = np.polyval(a_t, times)
    j = np.polyval(j_t, times)
    plt.plot(times, a, times, j)
    a=0