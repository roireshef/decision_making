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
