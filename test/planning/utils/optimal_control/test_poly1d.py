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


import matplotlib.pyplot as plt

def test_():
    A_inv = QuinticPoly1D.inverse_time_constraints_matrix(6)
    constraints = np.array([0, 0, 0, 3.6, 0, 0])
    x_t = QuinticPoly1D.solve(A_inv, constraints)
    v_t = np.polyder(x_t)
    a_t = np.polyder(v_t)
    j_t = np.polyder(a_t)
    times = np.arange(0,6.001,0.1)
    a = np.polyval(a_t, times)
    j = np.polyval(j_t, times)
    plt.plot(times, a)
    a=0