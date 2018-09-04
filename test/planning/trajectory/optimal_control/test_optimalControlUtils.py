import numpy as np

from decision_making.src.global_constants import VELOCITY_LIMITS
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, Poly1D, \
    QuarticPoly1D


def test_optimalControlUtils_validPolynomZeroToZero_returnsTrue():
    """tests if acceleration profile of x_dotdot(t) = -18x^3+35x^2-15x-0.1 in the range [0, 1.3]
    (acceleration goes from zero to negative value, then positive and then to zero) is in the limits
    [-2, 2] which is True """

    acc_poly = np.array([-18, 35, -15, -0.1])
    pos_poly = np.polyint(acc_poly, 2)

    assert QuinticPoly1D.is_acceleration_in_limits(pos_poly, 1.3, np.array([-2, 2]))


def test_optimalControlUtils_invalidPolynomZeroToZero_returnsFalse():
    """tests if acceleration profile of x_dotdot(t) = -18x^3+35x^2-15x-0.1 in the range [0, 1.3]
    (acceleration goes from zero to negative value, then positive and then to zero) is in the limits
    [-1, 1] which is False """

    acc_poly = np.array([-18, 35, -15, -0.1])
    pos_poly = np.polyint(acc_poly, 2)

    assert not QuinticPoly1D.is_acceleration_in_limits(pos_poly, 1.3, np.array([-1, 1]))


def test_optimalControlUtils_validPolynomZeroToPositive_returnsTrue():
    """tests if acceleration profile of x_dotdot(t) = -18x^3+35x^2-15x-0.1 in the range [0, 1]
    (acceleration goes from zero to negative value, then positive) is in the limits
    [-2, 2] which is True """

    acc_poly = np.array([-18, 35, -15, -0.1])
    pos_poly = np.polyint(acc_poly, 2)

    assert QuinticPoly1D.is_acceleration_in_limits(pos_poly, 1.3, np.array([-2, 2]))


def test_optimalControlUtils_invalidPolynomZeroToPositive_returnsFalse():
    """tests if acceleration profile of x_dotdot(t) = -18x^3+35x^2-15x-0.1 in the range [0, 1]
    (acceleration goes from zero to negative value, then positive ) is in the limits
    [-1, 1] which is False """

    acc_poly = np.array([-18, 35, -15, -0.1])
    pos_poly = np.polyint(acc_poly, 2)

    assert not QuinticPoly1D.is_acceleration_in_limits(pos_poly, 1.3, np.array([-1, 1]))


def test_velocitiesInLimits_testQuinticAndQuartic():
    T_vals = np.arange(7, 7.10001, 0.1)
    poly_coefs1 = np.array([np.array([4.5551256270804475e-05, -0.0031261275542158826, 0.079508383655999507,
                                      -0.90949571395743578, 3.8062370283689084, 88.016858456033745]), ]
                          * T_vals.shape[0])
    poly_coefs2 = np.array([np.array([-0.0031261275542158826, 0.079508383655999507,
                                      -0.90949571395743578, 3.8062370283689084, 88.016858456033745]), ]
                           * T_vals.shape[0])
    in_limits2 = Poly1D.are_velocities_in_limits(poly_coefs2, T_vals, VELOCITY_LIMITS)
    assert in_limits2[0] and in_limits2[1]

    in_limits3 = Poly1D.are_velocities_in_limits(poly_coefs1, T_vals, VELOCITY_LIMITS)
    assert in_limits3[0] and not in_limits3[1]
