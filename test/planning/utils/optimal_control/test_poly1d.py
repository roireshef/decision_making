from decision_making.src.planning.utils.optimal_control.poly1d import Poly1D
import numpy as np


def test_areDerivativesInLimits_firstDegreePolynomialNotInLimits_returnFalse():
    poly_coefs = np.array([[1, 3]])
    T_vals = np.array([10])
    limits = np.array([0, 3])
    is_in_limits = Poly1D.are_derivatives_in_limits(0, poly_coefs, T_vals, limits=limits)

    assert not is_in_limits


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

