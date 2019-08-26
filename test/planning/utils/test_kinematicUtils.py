import numpy as np
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, TRAJECTORY_TIME_RESOLUTION

from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D, Poly1D


def test_isMaintainingDistance_safeSettings_returnsTrue():
    """ see https://www.desmos.com/calculator/t5db9ymokk (D_headwaymargin line) """
    v0 = 10
    vT = 0
    delta_s_init = 35
    headway_specify = 1.5
    headway_safe = 0.7
    safe_margin = 3
    specify_margin = 5

    s_specify = delta_s_init - headway_specify*vT - specify_margin
    constraints_s = np.array([0, v0, 0, s_specify, vT, 0])

    T = 10
    A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(T))

    poly_host_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
    poly_target_s = np.array([0, 0, 0, 0, vT, delta_s_init])

    assert KinematicUtils.is_maintaining_distance(poly_host_s, poly_target_s, safe_margin, headway_safe, np.array([0, T]))
    assert KinematicUtils.calc_safety_margin(poly_host_s, poly_target_s, safe_margin, headway_safe, np.array([0, T])) > 0


def test_isMaintainingDistance_unsafeSettings_returnsFalse():
    """ see https://www.desmos.com/calculator/tqyzkbttoq (D_headwaymargin line) """
    v0 = 10
    vT = 0
    delta_s_init = 35
    headway_specify = 1.5
    headway_safe = 0.7
    safe_margin = 3.5
    specify_margin = 5

    s_specify = delta_s_init - headway_specify*vT - specify_margin
    constraints_s = np.array([0, v0, 0, s_specify, vT, 0])

    T = 10
    A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(T))

    poly_host_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
    poly_target_s = np.array([0, 0, 0, 0, vT, delta_s_init])

    assert not KinematicUtils.is_maintaining_distance(poly_host_s, poly_target_s, safe_margin, headway_safe, np.array([0, T]))
    assert KinematicUtils.calc_safety_margin(poly_host_s, poly_target_s, safe_margin, headway_safe, np.array([0, T])) < 0


def test_filterByVelocityLimit_velocityDecreasesTowardLimit_valid():
    """
    initial velocity is above the limit and initial acceleration is positive;
    final velocity is NOT above the limit, so the trajectory is valid
    """
    v0 = 20
    vT = 10
    velocity_limits = np.full(shape=[1, int(BP_ACTION_T_LIMITS[1] / TRAJECTORY_TIME_RESOLUTION)], fill_value=vT)

    constraints_s = np.array([0, v0, 1, vT, 0])  # initial acceleration is positive
    T = np.array([10])
    A_inv = QuarticPoly1D.inverse_time_constraints_tensor(T)

    # create ftrajectories
    poly_host_s = QuarticPoly1D.zip_solve(A_inv, constraints_s[np.newaxis, :])
    time_samples = np.arange(0, BP_ACTION_T_LIMITS[1], TRAJECTORY_TIME_RESOLUTION)
    ftrajectories_s = Poly1D.polyval_with_derivatives(poly_host_s, time_samples)

    # create ctrajectories
    zeros = np.zeros((ftrajectories_s.shape[0], ftrajectories_s.shape[1], 1))
    ctrajectories = np.c_[ftrajectories_s[..., 0:1], zeros, zeros, ftrajectories_s[..., 1:2], ftrajectories_s[..., 2:3], zeros]

    conforms = KinematicUtils.filter_by_velocity_limit(ctrajectories, velocity_limits, T)
    assert conforms[0]


def test_filterByVelocityLimit_violatesLimitAndPositiveInitialJerk_invalid():
    """
    initial and final velocity are under the limit, initial jerk is positive, trajectory violates velocity limit
    """
    v0 = 10
    vT = 10
    velocity_limits = np.full(shape=[1, int(BP_ACTION_T_LIMITS[1] / TRAJECTORY_TIME_RESOLUTION)], fill_value=13)

    T = np.array([10])
    constraints_s = np.array([0, v0, 0, v0 * T[0] + 20, vT, 0])  # initial acceleration is positive
    A_inv = QuinticPoly1D.inverse_time_constraints_tensor(T)

    # create ftrajectories
    poly_host_s = QuinticPoly1D.zip_solve(A_inv, constraints_s[np.newaxis, :])
    time_samples = np.arange(0, BP_ACTION_T_LIMITS[1], TRAJECTORY_TIME_RESOLUTION)
    ftrajectories_s = Poly1D.polyval_with_derivatives(poly_host_s, time_samples)

    # create ctrajectories
    zeros = np.zeros((ftrajectories_s.shape[0], ftrajectories_s.shape[1], 1))
    ctrajectories = np.c_[ftrajectories_s[..., 0:1], zeros, zeros, ftrajectories_s[..., 1:2], ftrajectories_s[..., 2:3], zeros]

    conforms = KinematicUtils.filter_by_velocity_limit(ctrajectories, velocity_limits, T)
    assert not conforms[0]


def test_filterByVelocityLimit_velocityDecreasesAboveLimit_invalid():
    """
    initial velocity is above the limit and initial acceleration is positive;
    final velocity is above the limit
    """
    v0 = 20
    vT = 11
    velocity_limits = np.full(shape=[1, int(BP_ACTION_T_LIMITS[1] / TRAJECTORY_TIME_RESOLUTION)], fill_value=10)

    constraints_s = np.array([0, v0, 1, vT, 0])
    T = np.array([10])
    A_inv = QuarticPoly1D.inverse_time_constraints_tensor(T)

    # create ftrajectories
    poly_host_s = QuarticPoly1D.zip_solve(A_inv, constraints_s[np.newaxis, :])
    time_samples = np.arange(0, BP_ACTION_T_LIMITS[1], TRAJECTORY_TIME_RESOLUTION)
    ftrajectories_s = Poly1D.polyval_with_derivatives(poly_host_s, time_samples)

    # create ctrajectories
    zeros = np.zeros((ftrajectories_s.shape[0], ftrajectories_s.shape[1], 1))
    ctrajectories = np.c_[ftrajectories_s[..., 0:1], zeros, zeros, ftrajectories_s[..., 1:2], ftrajectories_s[..., 2:3], zeros]

    conforms = KinematicUtils.filter_by_velocity_limit(ctrajectories, velocity_limits, T)
    assert not conforms[0]
