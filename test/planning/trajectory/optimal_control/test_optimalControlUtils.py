from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import QuinticPoly1D
import numpy as np


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