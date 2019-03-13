from decision_making.src.planning.types import LIMIT_MAX
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, Poly1D, QuarticPoly1D
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.test.planning.utils.optimal_control.quartic_poly_formulas import QuarticMotionPredicatesCreator
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import QuinticMotionPredicatesCreator

import numpy as np

from decision_making.src.global_constants import EPS, SPECIFICATION_MARGIN_TIME_DELAY, BP_JERK_S_JERK_D_TIME_WEIGHTS, \
    BP_ACTION_T_LIMITS, TRAJECTORY_TIME_RESOLUTION, LON_ACC_LIMITS, VELOCITY_LIMITS, HOST_SAFETY_MARGIN_TIME_DELAY, \
    ACTOR_SAFETY_MARGIN_TIME_DELAY, BP_JERK_S_JERK_D_TIME_WEIGHTS_FOLLOW_LANE, TRAJECTORY_NUM_POINTS
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.utils.metric_logger import MetricLogger


# This file deals with finding "optimal" weights for time-jerk cost function for dynamic actions.
# The "optimality" consists of two components:
# 1. Coverage of feasible (unfiltered) semantic actions in different scenarios. The function
#    jerk_time_weights_optimization() deals with it.
# 2. Velocity/acceleration profile of the actions should be convenient or good feel for the passengers, particularly
#    to prevent too late braking. The function calc_braking_quality_and_print_graphs() draws these profiles
#    and calculates the convenience rate.


def jerk_time_weights_optimization():
    """
    Create 3D grid of configurations (states): initial velocity, end velocity, initial distance from object.
    Create 3D grid of Jerk-Time weights for 3 aggressiveness levels.
    For each state and for each triplet of weights, check validity of the state based on acceleration limits,
    velocity limits, action time limits and RSS safety.
    Output brief states coverage for all weights sets (number of invalid states for each weights set).
    Output detailed states coverage (3D grid of states) for the best weights set or compare two weights sets.
    """
    # states grid ranges
    V_STEP = 2   # velocity step in the states grid
    V_MAX = 30   # max velocity in the states grid
    S_MIN = 10   # min distance between two objects in the states grid
    S_MAX = 120  # max distance between two objects in the states grid

    # weights grid ranges
    W2_FROM = 0.01  # min of the range of w2 weight
    W2_TILL = 0.2  # max of the range of w2 weight
    W12_RATIO_FROM = 1.2  # min of the range of ratio w1/w2
    W12_RATIO_TILL = 32   # max of the range of ratio w1/w2
    W01_RATIO_FROM = 1.2  # min of the range of ratio w0/w1
    W01_RATIO_TILL = 32   # max of the range of ratio w0/w1
    GRID_RESOLUTION = 9   # the weights grid resolution

    # create ranges of the grid of states
    v0_range = np.arange(0, V_MAX + EPS, V_STEP)
    vT_range = np.arange(0, V_MAX + EPS, V_STEP)
    a0_range = np.array([0])
    s_range = np.arange(S_MIN, S_MAX + EPS, 10)

    # create the grid of states
    v0, vT, a0, s = np.meshgrid(v0_range, vT_range, a0_range, s_range)
    v0, vT, a0, s = np.ravel(v0), np.ravel(vT), np.ravel(a0), np.ravel(s)

    # create grid of weights
    test_full_range = True
    # s_weights is a matrix Wx3, where W is a set of jerk weights (time weight is constant) for 3 aggressiveness levels
    if test_full_range:
        # test a full range of weights (~8 minutes)
        s_weights = create_full_range_of_weights(W2_FROM, W2_TILL, W12_RATIO_FROM, W12_RATIO_TILL,
                                                 W01_RATIO_FROM, W01_RATIO_TILL, GRID_RESOLUTION)
    else:  # compare a pair of weights sets
        s_weights = np.array([[6., 1.6, 0.2]])

    # remove trivial states, for which T_s = 0
    non_trivial_states = np.where(~np.logical_and(np.isclose(v0, vT), np.isclose(vT * SPECIFICATION_MARGIN_TIME_DELAY, s)))
    v0, vT, a0, s = v0[non_trivial_states], vT[non_trivial_states], a0[non_trivial_states], s[non_trivial_states]
    not_too_far = np.where((v0 - vT) * BP_ACTION_T_LIMITS[1] > s)
    v0, vT, a0, s = v0[not_too_far], vT[not_too_far], a0[not_too_far], s[not_too_far]
    states_num = v0.shape[0]
    print('states num = %d' % (states_num))

    valid_states_mask = np.full((s_weights.shape[0], states_num), False)  # states that passed all limits & safety
    T_s = np.zeros((s_weights.shape[0], states_num, s_weights.shape[1]))
    profile_rates = np.zeros_like(s_weights)
    time_weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[:, 2]

    for wi, w in enumerate(s_weights):  # loop on weights' sets
        vel_acc_in_limits = np.zeros((states_num, s_weights.shape[1]))
        safe_actions = np.zeros_like(vel_acc_in_limits)
        for aggr in range(s_weights.shape[1]):  # loop on aggressiveness levels
            # calculate time horizon for all states
            T_s[wi, :, aggr] = T = QuinticMotionPredicatesCreator.calc_T_s(time_weights[aggr], w[aggr], v0, a0, vT, s)
            T[np.where(T == 0)] = 0.01  # prevent zero times
            # calculate states validity wrt velocity & acceleration limits
            vel_acc_in_limits[:, aggr], safe_actions[:, aggr], poly_coefs = check_action_limits_and_safety(T, v0, vT, s, a0)

            # Calculate average (on all valid actions) acceleration profile rate.
            # Late braking (less pleasant for passengers) gets higher rate than early braking.
            rate_sum = 0.
            rate_num = 0
            for i in range(states_num):
                if vel_acc_in_limits[i, aggr] and safe_actions[i, aggr] and T_s[wi, i, aggr] <= BP_ACTION_T_LIMITS[1]:
                    # calculate velocity & acceleration profile
                    time_samples = np.arange(0, T[i] + EPS, TRAJECTORY_TIME_RESOLUTION)
                    distance_profile = QuinticPoly1D.distance_from_target(a0[i], v0[i], vT[i], s[i], T[i], T_m=SPECIFICATION_MARGIN_TIME_DELAY)
                    acc_poly_coefs = Math.polyder2d(poly_coefs[i:i+1], m=2)
                    acc_samples = Math.polyval2d(acc_poly_coefs, time_samples)[0]
                    brake_rate, rate_weight = get_braking_quality(distance_profile(time_samples), acc_samples)  # calculate acceleration profile rate
                    if brake_rate is not None:  # else there was not braking
                        rate_sum += rate_weight * brake_rate
                        rate_num += rate_weight
            if rate_num > 0:
                profile_rates[wi, aggr] = rate_sum/rate_num

        # combine velocity & acceleration limits with time limits and safety, to obtain states validity
        time_in_limits = (T_s[wi, :, :] <= BP_ACTION_T_LIMITS[LIMIT_MAX])
        in_limits = np.logical_and(vel_acc_in_limits, np.logical_and(time_in_limits, safe_actions))
        valid_states_mask[wi] = in_limits.any(axis=-1)  # OR on aggressiveness levels

        print('weight: %7.3f %.3f %.3f: passed %.1f%%\t\tvel_acc %s   safety %s   time %s;\tprofile %s' %
              (w[0], w[1], w[2], np.sum(valid_states_mask[wi])*100./states_num,
               (np.sum(vel_acc_in_limits, axis=0)*100/states_num).astype(np.int),
               (np.sum(safe_actions, axis=0)*100/states_num).astype(np.int),
               (np.sum(time_in_limits, axis=0)*100/states_num).astype(np.int), profile_rates[wi]))

    if test_full_range:
        # Monitor a quality of the best set of weights (maximal roots).
        print_success_map_for_best_weights_set(v0_range, vT_range, s_range, v0, vT, s, valid_states_mask, s_weights)
    else:
        # Compare between two sets of weights (maximal roots).
        print_comparison_between_two_weights_sets(v0_range, vT_range, s_range, v0, vT, s, valid_states_mask)


def jerk_time_weights_optimization_for_braking():
    """
    Create 3D grid of configurations (states): initial velocity, end velocity, initial distance from object.
    Create 3D grid of Jerk-Time weights for 3 aggressiveness levels.
    For each state and for each triplet of weights, check validity of the state based on acceleration limits,
    velocity limits, action time limits and RSS safety.
    Output brief states coverage for all weights sets (number of invalid states for each weights set).
    Output detailed states coverage (3D grid of states) for the best weights set or compare two weights sets.
    """
    # states grid ranges
    V0 = 30.
    V_MIN = 25.
    V_MAX = 28.   # max velocity in the states grid
    V_STEP = 0.2   # velocity step in the states grid
    S_MAX = SPECIFICATION_MARGIN_TIME_DELAY * V0 - 0  # max distance between two objects in the states grid
    S_MIN = SPECIFICATION_MARGIN_TIME_DELAY * V0 - 5  # min distance between two objects in the states grid
    S_STEP = 0.5

    # weights grid ranges
    W2_FROM = 0.001  # min of the range of w2 weight
    W2_TILL = 0.02  # max of the range of w2 weight
    W12_RATIO_FROM = 3  # min of the range of ratio w1/w2
    W12_RATIO_TILL = 64   # max of the range of ratio w1/w2
    W01_RATIO_FROM = 1.2  # min of the range of ratio w0/w1
    W01_RATIO_TILL = 64   # max of the range of ratio w0/w1
    GRID_RESOLUTION = 10   # the weights grid resolution

    # create ranges of the grid of states
    v0_range = np.array([V0])  # np.arange(0, V_MAX + EPS, V_STEP)
    vT_range = np.arange(V_MIN, V_MAX + EPS, V_STEP)
    a0_range = np.array([0])
    s_range = np.arange(S_MIN, S_MAX + EPS, S_STEP)

    # create the grid of states
    v0, vT, a0, s = np.meshgrid(v0_range, vT_range, a0_range, s_range)
    v0, vT, a0, s = np.ravel(v0), np.ravel(vT), np.ravel(a0), np.ravel(s)

    # create grid of weights
    test_full_range = True
    # s_weights is a matrix Wx3, where W is a set of jerk weights (time weight is constant) for 3 aggressiveness levels
    if test_full_range:
        # test a full range of weights (~8 minutes)
        s_weights = create_full_range_of_weights(W2_FROM, W2_TILL, W12_RATIO_FROM, W12_RATIO_TILL,
                                                 W01_RATIO_FROM, W01_RATIO_TILL, GRID_RESOLUTION)
    else:  # compare a pair of weights sets
        s_weights = np.array([[6, 0.9, 0.16], [6, 0.1, 0.005]])

    # remove trivial states, for which T_s = 0
    non_trivial_states = np.where(~np.logical_and(np.isclose(v0, vT), np.isclose(vT * SPECIFICATION_MARGIN_TIME_DELAY, s)))
    v0, vT, a0, s = v0[non_trivial_states], vT[non_trivial_states], a0[non_trivial_states], s[non_trivial_states]
    states_num = v0.shape[0]
    print('states num = %d' % (states_num))

    valid_states_mask = np.full((s_weights.shape[0], states_num), False)  # states that passed all limits & safety
    T_s = np.zeros((s_weights.shape[0], states_num, s_weights.shape[1]))
    profile_rates = np.zeros_like(s_weights)
    time_weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[:, 2]

    for wi, w in enumerate(s_weights):  # loop on weights' sets
        vel_acc_in_limits = np.zeros((states_num, s_weights.shape[1]))
        safe_actions = np.zeros_like(vel_acc_in_limits)
        for aggr in range(s_weights.shape[1]):  # loop on aggressiveness levels
            # calculate time horizon for all states
            T_s[wi, :, aggr] = T = QuinticMotionPredicatesCreator.calc_T_s(time_weights[aggr], w[aggr], v0, a0, vT, s)
            T[np.where(T == 0)] = 0.01  # prevent zero times
            valid_T = np.logical_not(np.isnan(T))
            # calculate states validity wrt velocity & acceleration limits
            vel_acc_in_limits[valid_T, aggr], safe_actions[valid_T, aggr], poly_coefs = \
                check_action_limits_and_safety(T[valid_T], v0[valid_T], vT[valid_T], s[valid_T], a0[valid_T])

        # combine velocity & acceleration limits with time limits and safety, to obtain states validity
        time_in_limits = (T_s[wi, :, :] <= BP_ACTION_T_LIMITS[LIMIT_MAX])
        in_limits = np.logical_and(vel_acc_in_limits, np.logical_and(time_in_limits, safe_actions))
        valid_states_mask[wi] = in_limits.any(axis=-1)  # OR on aggressiveness levels

        print('weight: %7.3f %.3f %.3f: passed %.1f%%\t\tvel_acc %s   safety %s   time %s;\tprofile %s' %
              (w[0], w[1], w[2], np.sum(valid_states_mask[wi])*100./states_num,
               (np.sum(vel_acc_in_limits, axis=0)*100/states_num).astype(np.int),
               (np.sum(safe_actions, axis=0)*100/states_num).astype(np.int),
               (np.sum(time_in_limits, axis=0)*100/states_num).astype(np.int), profile_rates[wi]))

    if test_full_range:
        # Monitor a quality of the best set of weights (maximal roots).
        print_success_map_for_best_weights_set(v0_range, vT_range, s_range, v0, vT, s, valid_states_mask, s_weights)
    else:
        # Compare between two sets of weights (maximal roots).
        print_comparison_between_two_weights_sets(v0_range, vT_range, s_range, v0, vT, s, valid_states_mask)


def jerk_time_weights_optimization_quartic():
    """
    Create 3D grid of configurations (states): initial velocity, end velocity, initial distance from object.
    Create 3D grid of Jerk-Time weights for 3 aggressiveness levels.
    For each state and for each triplet of weights, check validity of the state based on acceleration limits,
    velocity limits, action time limits and RSS safety.
    Output brief states coverage for all weights sets (number of invalid states for each weights set).
    Output detailed states coverage (3D grid of states) for the best weights set or compare two weights sets.
    """
    # states grid ranges
    V_MAX = 30   # max velocity in the states grid
    V_STEP = 1   # velocity step in the states grid
    S_MIN = 10   # min distance between two objects in the states grid
    S_MAX = 120  # max distance between two objects in the states grid
    S_STEP = 5

    # weights grid ranges
    W2_FROM = 0.001  # min of the range of w2 weight
    W2_TILL = 0.01  # max of the range of w2 weight
    W12_RATIO_FROM = 1.2  # min of the range of ratio w1/w2
    W12_RATIO_TILL = 32   # max of the range of ratio w1/w2
    W01_RATIO_FROM = 1.2  # min of the range of ratio w0/w1
    W01_RATIO_TILL = 32   # max of the range of ratio w0/w1
    GRID_RESOLUTION = 9   # the weights grid resolution

    # create ranges of the grid of states
    v0_range = np.arange(0, V_MAX + EPS, V_STEP)
    vT_range = np.arange(0, V_MAX + EPS, V_STEP)
    a0_range = np.array([0])
    s_range = np.arange(S_MIN, S_MAX + EPS, S_STEP)

    # create the grid of states
    v0, vT, a0, s = np.meshgrid(v0_range, vT_range, a0_range, s_range)
    v0, vT, a0, s = np.ravel(v0), np.ravel(vT), np.ravel(a0), np.ravel(s)

    # create grid of weights
    test_full_range = False
    # s_weights is a matrix Wx3, where W is a set of jerk weights (time weight is constant) for 3 aggressiveness levels
    if test_full_range:
        # test a full range of weights (~8 minutes)
        s_weights = create_full_range_of_weights(W2_FROM, W2_TILL, W12_RATIO_FROM, W12_RATIO_TILL,
                                                 W01_RATIO_FROM, W01_RATIO_TILL, GRID_RESOLUTION)
    else:  # compare a pair of weights sets
        s_weights = np.array([[0.16, 0.015, 0.005], [0.16, 0.01, 0.004]])

    # remove trivial states, for which T_s = 0
    non_trivial_states = np.where(~np.logical_and(np.isclose(v0, vT), np.isclose(vT * SPECIFICATION_MARGIN_TIME_DELAY, s)))
    v0, vT, a0, s = v0[non_trivial_states], vT[non_trivial_states], a0[non_trivial_states], s[non_trivial_states]
    not_too_far = np.where(np.logical_or((v0 - vT) * BP_ACTION_T_LIMITS[1] > s, np.logical_and(vT > v0, s > vT)))
    v0, vT, a0, s = v0[not_too_far], vT[not_too_far], a0[not_too_far], s[not_too_far]
    states_num = v0.shape[0]
    print('states num = %d' % (states_num))

    valid_states_mask = np.full((s_weights.shape[0], states_num), False)  # states that passed all limits & safety
    T_s = np.zeros((s_weights.shape[0], states_num, s_weights.shape[1]))
    profile_rates = np.zeros_like(s_weights)
    time_weights = BP_JERK_S_JERK_D_TIME_WEIGHTS_FOLLOW_LANE[:, 2]

    for wi, w in enumerate(s_weights):  # loop on weights' sets
        vel_acc_in_limits = np.zeros((states_num, s_weights.shape[1]))
        safe_actions = np.zeros_like(vel_acc_in_limits)
        for aggr in range(s_weights.shape[1]):  # loop on aggressiveness levels
            # calculate time horizon for all states
            T_s[wi, :, aggr] = T = QuarticMotionPredicatesCreator.calc_T_s(time_weights[aggr], w[aggr], v0, a0, vT)
            T[np.where(T == 0)] = 0.01  # prevent zero times
            # calculate states validity wrt velocity & acceleration limits
            vel_acc_in_limits[:, aggr], safe_actions[:, aggr], poly_coefs = \
                check_action_limits_and_safety_quartic(T, v0, vT, s, a0)

        # combine velocity & acceleration limits with time limits and safety, to obtain states validity
        time_in_limits = (T_s[wi, :, :] <= BP_ACTION_T_LIMITS[LIMIT_MAX])
        in_limits = np.logical_and(vel_acc_in_limits, np.logical_and(time_in_limits, safe_actions))
        valid_states_mask[wi] = in_limits.any(axis=-1)  # OR on aggressiveness levels

        print('weight: %7.3f %.3f %.3f: passed %.1f%%\t\tvel_acc %s   safety %s   time %s;\tprofile %s' %
              (w[0], w[1], w[2], np.sum(valid_states_mask[wi])*100./states_num,
               (np.sum(vel_acc_in_limits, axis=0)*100/states_num).astype(np.int),
               (np.sum(safe_actions, axis=0)*100/states_num).astype(np.int),
               (np.sum(time_in_limits, axis=0)*100/states_num).astype(np.int), profile_rates[wi]))

    best_wi = np.argmax(np.sum(valid_states_mask, axis=-1))
    passed = np.sum(valid_states_mask[best_wi])
    print('best weights for max: %s; passed %d (%.2f%%)' % (s_weights[best_wi], passed, 100.*passed/valid_states_mask.shape[-1]))


def create_full_range_of_weights(w2_from: float, w2_till: float, w12_ratio_from: float, w12_ratio_till: float,
                                 w01_ratio_from: float, w01_ratio_till: float, resolution: int) -> np.array:
    """
    Create grid of full range of weights (the full optimization runs ~8 minutes)
    :param w2_from: min of the range of w2 weight
    :param w2_till: max of the range of w2 weight
    :param w12_ratio_from: min of the range of ratio w1/w2
    :param w12_ratio_till: max of the range of ratio w1/w2
    :param w01_ratio_from: min of the range of ratio w0/w1
    :param w01_ratio_till: max of the range of ratio w0/w1
    :param resolution: the weights grid resolution
    :return: grid of weights: 3D matrix of shape resolution x resolution x resolution
    """
    # geomspace creates geometrical progression
    w2_range = np.geomspace(w2_from, w2_till, num=resolution)
    w12_range = np.geomspace(w12_ratio_from, w12_ratio_till, num=resolution)
    w01_range = np.geomspace(w01_ratio_from, w01_ratio_till, num=resolution)
    w01, w12, w2 = np.meshgrid(w01_range, w12_range, w2_range)
    w01, w12, w2 = np.ravel(w01), np.ravel(w12), np.ravel(w2)
    weights = np.c_[w01 * w12 * w2, w12 * w2, w2]
    return weights


def check_action_limits_and_safety(T: np.array, v_0: np.array, v_T: np.array, s_T: np.array, a_0: np.array) -> \
        [np.array, np.array, np.array]:
    """
    Given longitudinal action dynamics, calculate validity wrt velocity & acceleration limits, and safety of each action.
    :param T: array of action times
    :param v_0: array of initial velocities
    :param v_T: array of final velocities
    :param a_0: array of initial acceleration
    :param s_T: array of initial distances from target
    :return: (1) boolean array: are velocity & acceleration in limits,
             (2) boolean array: is the baseline trajectory safe
             (3) coefficients of s polynomial
    """
    # check limits
    is_in_limits, poly_coefs = QuinticMotionPredicatesCreator.check_action_limits(T, v_0, v_T, s_T, a_0)
    # check safety
    is_safe = get_lon_safety_for_action_specs(poly_coefs, T, v_T, s_T)
    return is_in_limits, is_safe, poly_coefs


def check_action_limits_and_safety_quartic(T: np.array, v_0: np.array, v_T: np.array, s_T: np.array, a_0: np.array) -> \
        [np.array, np.array, np.array]:
    """
    Given longitudinal action dynamics, calculate validity wrt velocity & acceleration limits, and safety of each action.
    :param T: array of action times
    :param v_0: array of initial velocities
    :param v_T: array of final velocities
    :param a_0: array of initial acceleration
    :param s_T: array of initial distances from target
    :return: (1) boolean array: are velocity & acceleration in limits,
             (2) boolean array: is the baseline trajectory safe
             (3) coefficients of s polynomial
    """
    # check limits
    is_in_limits, poly_coefs = QuarticMotionPredicatesCreator.check_action_limits(T, v_0, v_T, a_0)
    # check safety
    is_safe = get_lon_safety_for_action_specs(poly_coefs, T, v_T, s_T)
    return is_in_limits, is_safe, poly_coefs


def get_lon_safety_for_action_specs(poly_coefs: np.array, T: np.array, v_T: np.array, s_T: np.array) -> np.array:
    """
    Given polynomial coefficients and action specs for each polynomial, calculate longitudinal safety
    w.r.t. the front cars with constant velocities, described by the specs.
    :param poly_coefs: 2D matrix Nx6: N quintic polynomials of ego
    :param T: array of action times
    :param v_T: array of final velocities
    :param s_T: array of initial distances from target
    :return: boolean array of size N: longitudinal safety for each spec
    """
    time_samples = np.arange(0, np.max(T) + EPS, TRAJECTORY_TIME_RESOLUTION)

    # sample polynomials and create ftrajectories_s
    trajectories_s = Poly1D.polyval_with_derivatives(poly_coefs, time_samples)
    trajectories = np.concatenate((trajectories_s, np.zeros_like(trajectories_s)), axis=-1)

    ego_trajectories = [trajectory[:int(T[i] / TRAJECTORY_TIME_RESOLUTION) + 1]
                        for i, trajectory in enumerate(trajectories)]

    obj_trajectories = [np.c_[(s_T[i] + np.linspace(0, v_T[i] * (len(ego_trajectory)-1) * TRAJECTORY_TIME_RESOLUTION,
                                                   len(ego_trajectory)),
                              np.full(len(ego_trajectory), v_T[i]),
                              np.zeros(len(ego_trajectory)),
                              np.zeros(len(ego_trajectory)),
                              np.zeros(len(ego_trajectory)),
                              np.zeros(len(ego_trajectory)))]
                        for i, ego_trajectory in enumerate(ego_trajectories)]

    # concatenate all trajectories to a single long trajectory
    concat_ego_trajectory = np.concatenate(ego_trajectories)
    concat_obj_trajectory = np.concatenate(obj_trajectories)

    # calc longitudinal RSS for the long trajectory
    concat_safe_times = SafetyUtils._get_lon_safe_dist(
        concat_ego_trajectory, HOST_SAFETY_MARGIN_TIME_DELAY, concat_obj_trajectory, ACTOR_SAFETY_MARGIN_TIME_DELAY,
        margin=0) > 0

    # split the safety results according to the original trajectories
    trajectories_lengths = [len(trajectory) for trajectory in ego_trajectories]
    safe_times_matrix = np.split(concat_safe_times, np.cumsum(trajectories_lengths[:-1]))
    # AND on all time samples for each action
    safe_specs = [action_safe_times[:TRAJECTORY_NUM_POINTS].all() for action_safe_times in safe_times_matrix]
    return np.array(safe_specs)


def get_braking_quality(distances: np.array, acc_samples: np.array) -> [float, float]:
    """
    calculate normalized center of mass of braking: braking close to target (less pleasant) gets higher rate
    :param distances: array of distances from the followed object
    :param acc_samples: array of acceleration samples
    :return: 1. center of mass of negative acc_samples as function of distance from the object;
                the lower rate the better
             2. the rate weight: the higher accelerations the higher weight
    """
    final_dist = distances[-1]
    max_brake = max(0., -min(acc_samples))
    if max_brake == 0:
        return 0, 0
    brake_idxs = np.where(acc_samples < 0)[0]
    cum_brake = -np.sum(acc_samples[brake_idxs])
    cum_brake_dist = -np.sum(acc_samples[brake_idxs] / np.maximum(1, distances[brake_idxs])) * final_dist
    return (cum_brake_dist / cum_brake) ** 2, max_brake


def print_success_map_for_const_vT(v0_range: np.array, s_range: np.array, v0_grid: np.array, s_grid: np.array,
                                   is_valid_state: np.array, weights: np.array):
    """
    Use for monitoring limits for all sets of weights. Print tables (v0 x s) per weights set
    :param v0_range: v0 values in the 3D grid
    :param s_range:  s values in the 3D grid
    :param v0_grid: v0 of meshgrid of v0_range, vT_range, s_range
    :param s_grid: s of meshgrid of v0_range, vT_range, s_range
    :param is_valid_state: boolean array: True if the state is valid (complies all thresholds & safety)
    :param weights: array of all sets of weights (Nx3)
    """
    np.set_printoptions(suppress=True)
    success_map = np.ones((weights.shape[0], v0_range.shape[0], s_range.shape[0]))
    for wi, w in enumerate(weights):
        for i, b in enumerate(is_valid_state[wi]):
            if not b:
                success_map[wi, np.argwhere(v0_range == v0_grid[i]), np.argwhere(s_range == s_grid[i])] = 0
        print('%3d: weights = %s' % (wi, w))
        print(success_map[wi].astype(int))
    print('max over all weights:')
    print(np.max(success_map, axis=0).astype(int))


def print_success_map_for_best_weights_set(v0_range: np.array, vT_range: np.array, s_range: np.array,
                                           v0_grid: np.array, vT_grid: np.array, s_grid: np.array,
                                           is_valid_state: np.array, weights: np.array):
    """
    Use for monitoring a quality of the best set of weights (maximal roots).
    Print tables (vT x s) for the best weights per v0
    :param v0_range: v0 values in the 3D grid
    :param vT_range: vT values in the 3D grid
    :param s_range:  s values in the 3D grid
    :param v0_grid: v0 of meshgrid of v0_range, vT_range, s_range
    :param vT_grid: vT of meshgrid of v0_range, vT_range, s_range
    :param s_grid: s of meshgrid of v0_range, vT_range, s_range
    :param is_valid_state: boolean array: True if the state is valid (complies all thresholds & safety)
    :param weights: array of all sets of weights (Nx3)
    """
    best_wi = np.argmax(np.sum(is_valid_state, axis=-1))
    passed = np.sum(is_valid_state[best_wi])
    print('best weights for max: %s; passed %d (%.2f%%)' % (weights[best_wi], passed, 100.*passed/is_valid_state.shape[-1]))

    for v0_fixed in v0_range:
        success_map = np.ones((vT_range.shape[0], s_range.shape[0]))
        for i, b in enumerate(is_valid_state[best_wi]):
            if v0_grid[i] == v0_fixed and not b:
                success_map[np.argwhere(vT_range == vT_grid[i]), np.argwhere(s_range == s_grid[i])] = 0
        print('v0_fixed = %d' % v0_fixed)
        print(success_map.astype(int))


def print_comparison_between_two_weights_sets(v0_range: np.array, vT_range: np.array, s_range: np.array,
                                              v0_grid: np.array, vT_grid: np.array, s_grid: np.array,
                                              is_valid_state: np.array):
    """
    Use for comparison between two sets of weights (maximal roots).
    Print tables (vT x s) for weights1 relatively to weights0 per v0.
    Each value in the table means:
       1: weights1 is better than weights0 for this state
      -1: weights1 is worse than weights0 for this state
       0: weights1 is like weights0 for this state
    :param v0_range: v0 values in the 3D grid
    :param vT_range: vT values in the 3D grid
    :param s_range:  s values in the 3D grid
    :param v0_grid: v0 of meshgrid of v0_range, vT_range, s_range
    :param vT_grid: vT of meshgrid of v0_range, vT_range, s_range
    :param s_grid: s of meshgrid of v0_range, vT_range, s_range
    :param is_valid_state: boolean array: True if the state is valid for weights1 (complies all thresholds & safety)
    """
    print('Comparison:')
    for v0_fixed in v0_range:
        success_map = np.zeros((vT_range.shape[0], s_range.shape[0]))
        for i in range(v0_grid.shape[0]):  # loop over states' successes for weights[0]
            if v0_grid[i] == v0_fixed:
                success_map[np.argwhere(vT_range == vT_grid[i]), np.argwhere(s_range == s_grid[i])] = \
                    (is_valid_state[1, i].astype(int) - is_valid_state[0, i].astype(int))  # diff between the two weights qualities
        print('v0_fixed = %d' % v0_fixed)
        print(success_map.astype(int))


if __name__ == '__main__':
    jerk_time_weights_optimization_for_braking()
    #jerk_time_weights_optimization_quartic()
    #calc_braking_quality_and_print_graphs()
