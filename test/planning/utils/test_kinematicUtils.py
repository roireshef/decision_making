from unittest.mock import patch

import numpy as np
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, TRAJECTORY_TIME_RESOLUTION, SAFETY_HEADWAY, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.evaluators.single_lane_action_spec_evaluator import \
    SingleLaneActionSpecEvaluator
from decision_making.src.planning.types import C_X, C_V, C_A, FS_SX, FS_SV, FS_SA, FS_2D_LEN

from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D, Poly1D
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel



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
    assert SingleLaneActionSpecEvaluator.calc_minimal_headway_over_trajectory(poly_host_s, poly_target_s, safe_margin, np.array([0, T])) > SAFETY_HEADWAY


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
    assert SingleLaneActionSpecEvaluator.calc_minimal_headway_over_trajectory(poly_host_s, poly_target_s, safe_margin, np.array([0, T])) < SAFETY_HEADWAY + 0.1


def test_calcMinimalHeadwayOverTrajectory_constantSpeedRvFaster_returnsHeadwayAtStart():
    vT = 10
    v1 = vT - 1
    delta_s_init = 35
    safe_margin = 3
    T = 10

    poly_host_s = np.array([0, 0, 0, 0, v1, 0])
    poly_target_s = np.array([0, 0, 0, 0, vT, delta_s_init])

    minimal_headway = SingleLaneActionSpecEvaluator.calc_minimal_headway_over_trajectory(poly_host_s, poly_target_s, safe_margin, np.array([0, T]))
    expected_headway = (delta_s_init - safe_margin) / v1
    assert abs(minimal_headway - expected_headway) < 0.01


def test_calcMinimalHeadwayOverTrajectory_constantSpeedRvSlower_returnsHeadwayAtEnd():
    vT = 10
    v1 = vT + 1
    delta_s_init = 35
    safe_margin = 3
    T = 10

    poly_host_s = np.array([0, 0, 0, 0, v1, 0])
    poly_target_s = np.array([0, 0, 0, 0, vT, delta_s_init])

    minimal_headway = SingleLaneActionSpecEvaluator.calc_minimal_headway_over_trajectory(poly_host_s, poly_target_s, safe_margin, np.array([0, T]))
    expected_headway = (delta_s_init - safe_margin - T * (v1 - vT)) / v1
    assert abs(minimal_headway - expected_headway) < 0.01


def test_calcMinimalHeadwayOverTrajectory_constantAcceleration_returnsHeadwayAtEnd():
    vT = 10
    v1 = vT
    a1 = 2
    delta_s_init = 150
    safe_margin = 3
    T = 10

    poly_host_s = np.array([0, 0, 0, a1 / 2, v1, 0])
    poly_target_s = np.array([0, 0, 0, 0, vT, delta_s_init])

    minimal_headway = SingleLaneActionSpecEvaluator.calc_minimal_headway_over_trajectory(poly_host_s, poly_target_s, safe_margin, np.array([0, T]))
    expected_headway = (delta_s_init - safe_margin - a1 * T * T / 2) / (v1 + a1 * T)
    assert abs(minimal_headway - expected_headway) < 0.01


def test_calcMinimalHeadwayOverTrajectory_RvDeceleratesHvAccelerates_returnsHeadwayAtEnd():
    vT = 10
    v1 = vT
    a1 = 2
    aT = -2
    delta_s_init = 300
    safe_margin = 3
    T = 10

    poly_host_s = np.array([0, 0, 0, a1 / 2, v1, 0])
    poly_target_s = np.array([0, 0, 0, aT / 2, vT, delta_s_init])

    minimal_headway = SingleLaneActionSpecEvaluator.calc_minimal_headway_over_trajectory(poly_host_s, poly_target_s, safe_margin, np.array([0, T]))
    expected_headway = (delta_s_init - safe_margin - a1 * T * T / 2 + aT * T * T / 2) / (v1 + a1 *T)
    assert abs(minimal_headway - expected_headway) < 0.01


def test_calcMinimalHeadwayOverTrajectory_varyingAcceleration_returnsHeadwayAtMiddle():
    vT = 10
    v1 = vT
    a1 = 2
    j1 = -1
    delta_s_init = 150
    safe_margin = 3
    T = 6  # need a smaller value fot T, as otherwise the velocity becomes negative, which we do not support

    poly_host_s = np.array([0, 0, j1 / 6, a1 / 2, v1, 0])
    poly_target_s = np.array([0, 0, 0, 0, vT, delta_s_init])

    minimal_headway = SingleLaneActionSpecEvaluator.calc_minimal_headway_over_trajectory(poly_host_s, poly_target_s, safe_margin, np.array([0, T]))
    headway_on_the_way = np.array([((delta_s_init - safe_margin -(v1 - vT) * t - a1 * t * t / 2 - j1 * t ** 3 / 6) / (v1 + a1 * t + j1 * t * t / 2))
                                   for t in np.arange(0, T, 0.1)])
    min_headway = np.min(headway_on_the_way)
    assert abs(minimal_headway - min_headway) < 0.1 and \
           minimal_headway < headway_on_the_way[0] - 0.1 and \
           minimal_headway < headway_on_the_way[-1] - 0.1


def test_filterByVelocityLimit_velocityDecreasesTowardLimit_valid():
    """
    initial velocity is above the limit and initial acceleration is positive;
    final velocity is NOT above the limit, so the trajectory is valid
    """
    v0 = 10.1
    vT = 10
    # limit value selected as value at 3 seconds. Values prior to it are a little higher, but should ne valid
    velocity_limits = np.full(shape=[1, int(BP_ACTION_T_LIMITS[1] / TRAJECTORY_TIME_RESOLUTION)], fill_value=10.21)

    constraints_s = np.array([0, v0, 0.2, vT, 0])  # initial acceleration is positive
    T = np.array([6])
    last_idx = int(T/TRAJECTORY_TIME_RESOLUTION)
    A_inv = QuarticPoly1D.inverse_time_constraints_tensor(T)

    # create ftrajectories
    poly_host_s = QuarticPoly1D.zip_solve(A_inv, constraints_s[np.newaxis, :])
    time_samples = np.arange(0, BP_ACTION_T_LIMITS[1], TRAJECTORY_TIME_RESOLUTION)
    ftrajectories_s = Poly1D.polyval_with_derivatives(poly_host_s, time_samples)

    # create ctrajectories - make sure everything beyond T is set to 0.
    ctrajectories = np.zeros((ftrajectories_s.shape[0], ftrajectories_s.shape[1], FS_2D_LEN))
    ctrajectories[:, 0:last_idx, C_X] = ftrajectories_s[:, 0:last_idx, FS_SX]
    ctrajectories[:, 0:last_idx, C_V] = ftrajectories_s[:, 0:last_idx, FS_SV]
    ctrajectories[:, 0:last_idx, C_A] = ftrajectories_s[:, 0:last_idx, FS_SA]

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


# test KinematicUtils.specify_quartic_actions for different scenarios
def test_specifyQuarticActions_differentAggressivenessLevels_validActionsNumberIsCorrect():

    # both velocities and weights are scalars
    w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.STANDARD.value]
    ds, T = KinematicUtils.specify_quartic_actions(w_T, w_J, v_0=25., v_T=0.)
    assert np.sum(np.isfinite(T)) == 1

    # velocities are scalars
    w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS.T
    ds, T = KinematicUtils.specify_quartic_actions(w_T, w_J, v_0=40., v_T=0.)
    assert np.sum(np.isfinite(T)) == 1

    # don't limit the acceleration
    w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS.T
    ds, T = KinematicUtils.specify_quartic_actions(w_T, w_J, v_0=40., v_T=0., acc_limits=None)
    assert np.sum(np.isfinite(T)) == 2

    # don't limit time & acceleration
    w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS.T
    ds, T = KinematicUtils.specify_quartic_actions(w_T, w_J, v_0=40., v_T=0., action_horizon_limit=np.inf, acc_limits=None)
    assert np.sum(np.isfinite(T)) == 3

    # weights are scalars
    w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]
    ds, T = KinematicUtils.specify_quartic_actions(w_T, w_J, v_0=np.array([20., 25.]), v_T=np.array([0., 0.]))
    assert np.sum(np.isfinite(T)) == 1

    # both weights and velocities are arrays
    w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]
    ds, T = KinematicUtils.specify_quartic_actions(np.array([w_T, w_T]), np.array([w_J*0.9, w_J]),
                                                   v_0=np.array([20., 25.]), v_T=np.array([0., 0.]))
    assert np.sum(np.isfinite(T)) == 1

    # don't limit the time horizon
    w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]
    ds, T = KinematicUtils.specify_quartic_actions(w_T, w_J, v_0=np.array([20., 25.]), v_T=np.array([0., 0.]),
                                                   action_horizon_limit=np.inf)
    assert np.sum(np.isfinite(T)) == 2



def test_getLateralAccelerationLimitByCurvature_oneDimensionalArray_returnsExpecetedValuesWell():
    curvatures = np.array([0.1, 0.05, 0.01, 0.005, 1./275, 0.001])

    LAT_ACC_LIMITS_BY_K = np.array([(0, 6, 3, 3),
                                    (6, 25, 3, 2.8),
                                    (25, 50, 2.8, 2.55),
                                    (50, 75, 2.55, 2.4),
                                    (75, 100, 2.4, 2.32),
                                    (100, 150, 2.32, 2.2),
                                    (150, 200, 2.2, 2.1),
                                    (200, 275, 2.1, 2),
                                    (275, np.inf, 2, 2)])

    acc_limits = KinematicUtils.get_lateral_acceleration_limit_by_curvature(curvatures, LAT_ACC_LIMITS_BY_K)

    expected = np.array([(10 - 6)/(25-6)*(-0.2)+3,
                         (20 - 6)/(25-6)*(-0.2)+3,
                         2.32,
                         2.1,
                         2,
                         2])
    np.testing.assert_array_equal(acc_limits, expected)


def test_getLateralAccelerationLimitByCurvature_multiDimensionalArray_returnsExpecetedValuesWell():
    curvatures = np.array([[[0.1, 0.05]], [[0.01, 0.005]], [[1./275, 0.001]]])

    LAT_ACC_LIMITS_BY_K = np.array([(0, 6, 3, 3),
                                    (6, 25, 3, 2.8),
                                    (25, 50, 2.8, 2.55),
                                    (50, 75, 2.55, 2.4),
                                    (75, 100, 2.4, 2.32),
                                    (100, 150, 2.32, 2.2),
                                    (150, 200, 2.2, 2.1),
                                    (200, 275, 2.1, 2),
                                    (275, np.inf, 2, 2)])

    acc_limits = KinematicUtils.get_lateral_acceleration_limit_by_curvature(curvatures, LAT_ACC_LIMITS_BY_K)

    expected = np.array([[[(10 - 6)/(25-6)*(-0.2)+3,
                         (20 - 6)/(25-6)*(-0.2)+3]],
                         [[2.32,
                         2.1]],
                         [[2,
                         2]]])
    np.testing.assert_array_equal(acc_limits, expected)
