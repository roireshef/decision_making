from decision_making.src.planning.types import LIMIT_MAX
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import QuinticMotionPredicatesCreator

import numpy as np

from decision_making.src.global_constants import EPS, SPECIFICATION_MARGIN_TIME_DELAY, BP_JERK_S_JERK_D_TIME_WEIGHTS, \
    BP_ACTION_T_LIMITS, TRAJECTORY_TIME_RESOLUTION
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
    V_MAX = 18   # max velocity in the states grid
    S_MIN = 10   # min distance between two objects in the states grid
    S_MAX = 120  # max distance between two objects in the states grid

    # weights grid ranges
    W2_FROM = 0.04  # min of the range of w2 weight
    W2_TILL = 0.2  # max of the range of w2 weight
    W12_RATIO_FROM = 3  # min of the range of ratio w1/w2
    W12_RATIO_TILL = 32   # max of the range of ratio w1/w2
    W01_RATIO_FROM = 3  # min of the range of ratio w0/w1
    W01_RATIO_TILL = 32   # max of the range of ratio w0/w1
    GRID_RESOLUTION = 8   # the weights grid resolution

    # create ranges of the grid of states
    v0_range = np.arange(0, V_MAX + EPS, V_STEP)
    vT_range = np.arange(0, V_MAX + EPS, V_STEP)
    a0_range = np.array([0])
    s_range = np.arange(S_MIN, S_MAX + EPS, 10)

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
        s_weights = np.array([[6., 1.6, 0.2], [3.6, 1.2, 0.2]])

    # remove trivial states, for which T_s = 0
    non_trivial_states = np.where(~np.logical_and(np.isclose(v0, vT), np.isclose(vT * SPECIFICATION_MARGIN_TIME_DELAY, s)))
    v0, vT, a0, s = v0[non_trivial_states], vT[non_trivial_states], a0[non_trivial_states], s[non_trivial_states]
    not_too_high_accel = np.where(vT - v0 < 8)
    v0, vT, a0, s = v0[not_too_high_accel], vT[not_too_high_accel], a0[not_too_high_accel], s[not_too_high_accel]
    not_too_far_target = np.where(s <= np.maximum(v0, 2) * BP_ACTION_T_LIMITS[LIMIT_MAX])
    v0, vT, a0, s = v0[not_too_far_target], vT[not_too_far_target], a0[not_too_far_target], s[not_too_far_target]
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
            vel_acc_in_limits[:, aggr], safe_actions[:, aggr], poly_coefs = \
                QuinticMotionPredicatesCreator.check_action_validity(T, v0, vT, s, a0)

            # Calculate average (on all valid actions) acceleration profile rate.
            # Late braking (less pleasant for passengers) gets higher rate than early braking.
            if not test_full_range:
                rate_sum = 0.
                rate_num = 0
                for i in range(states_num):
                    if vel_acc_in_limits[i, aggr] and safe_actions[i, aggr] and T_s[wi, i, aggr] <= BP_ACTION_T_LIMITS[1]:
                        # calculate velocity & acceleration profile
                        time_samples = np.arange(0, T[i] + EPS, TRAJECTORY_TIME_RESOLUTION)
                        acc_poly_coefs = Math.polyder2d(poly_coefs[i:i+1], m=2)
                        acc_samples = Math.polyval2d(acc_poly_coefs, time_samples)[0]
                        brake_rate = get_braking_quality(acc_samples)  # calculate acceleration profile rate
                        if brake_rate is not None:  # else there was not braking
                            rate_sum += brake_rate
                            rate_num += 1
                if rate_num > 0:
                    profile_rates[wi, aggr] = rate_sum/rate_num

        # combine velocity & acceleration limits with time limits and safety, to obtain states validity
        time_in_limits = (T_s[wi, :, :] <= BP_ACTION_T_LIMITS[LIMIT_MAX])
        in_limits = np.logical_and(vel_acc_in_limits, np.logical_and(time_in_limits, safe_actions))
        valid_states_mask[wi] = in_limits.any(axis=-1)  # OR on aggressiveness levels

        print('weight: %7.3f %.3f %.3f: passed %d%%\t\tvel_acc %s   safety %s   time %s;\tprofile %s' %
              (w[0], w[1], w[2], np.sum(valid_states_mask[wi])*100/states_num,
               (np.sum(vel_acc_in_limits, axis=0)*100/states_num).astype(np.int),
               (np.sum(safe_actions, axis=0)*100/states_num).astype(np.int),
               (np.sum(time_in_limits, axis=0)*100/states_num).astype(np.int), profile_rates[wi]))

    if test_full_range:
        # Monitor a quality of the best set of weights (maximal roots).
        print_success_map_for_weights_set(v0_range, vT_range, s_range, v0, vT, s, valid_states_mask, s_weights)
    else:
        # Compare between two sets of weights (maximal roots).
        print_comparison_between_two_weights_sets(v0_range, vT_range, s_range, v0, vT, s, valid_states_mask)


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


def get_braking_quality(acc_samples: np.array) -> float:
    """
    calculate normalized center of mass of braking: late braking (less pleasant) gets higher rate than early braking
    :param acc_samples: array of acceleration samples
    :return: center of mass of negative acc_samples in the interval [0..1]
    """
    cum_acceleration_time = cum_acceleration = 0
    for i, acc in enumerate(acc_samples):
        if acc < 0:
            cum_acceleration_time += i * acc
            cum_acceleration += acc
    if cum_acceleration * len(acc_samples) == 0:
        return None
    return cum_acceleration_time / (cum_acceleration * len(acc_samples))


def print_success_map_for_weights_set(v0_range: np.array, vT_range: np.array, s_range: np.array,
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
    """
    best_wi = np.argmax(np.sum(is_valid_state, axis=-1))
    passed = np.sum(is_valid_state[best_wi])
    print('best weights for max: %s; passed %d (%.2f)' % (weights[best_wi], passed, float(passed)/is_valid_state.shape[-1]))

    for v0_fixed in v0_range:
        success_map = np.ones((vT_range.shape[0], s_range.shape[0]))
        for i, b in enumerate(is_valid_state[best_wi]):
            if v0_grid[i] == v0_fixed and not b:
                success_map[np.argwhere(vT_range == vT_grid[i]), np.argwhere(s_range == s_grid[i])] = b
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


def calc_braking_quality_and_print_graphs():
    """
    Loop on list of Jerk-Time weights sets and on initial states (v0, vT, s), calculate trajectory quality,
    according to the relative center of mass of negative accelerations (too late braking is bad).
    Draw graphs of velocity & acceleration profiles for all above settings.
    """
    MetricLogger.init('VelProfiles')
    ml = MetricLogger.get_logger()

    V0_FROM = 4
    V0_TILL = 18
    VT_FROM = 0
    VT_TILL_RELATIVE = -2  # terminal velocity - initial velocity
    V_STEP = 4
    S_FROM = 40  # 40
    S_TILL = 100  # 100

    a0 = 0.

    # jerk weights list
    jerk_weights_list = [np.array([16, 1.6, 0.08]), np.array([8.5, 0.34, 0.08])]
    time_weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[:, 2]


    all_acc_samples = {}
    acc_rate = {}
    all_T_s = {}

    # loop on all jerk weights, initial and terminal velocities (v0, vT) and distances between two objects (s);
    # calculate velocity / acceleration profile for each state in the grid
    for wi, w_J in enumerate(jerk_weights_list):
        for v0 in np.arange(V0_FROM, V0_TILL+EPS, V_STEP):
            for vT in np.arange(VT_FROM, v0 + VT_TILL_RELATIVE + EPS, V_STEP):
                for s in np.arange(S_FROM, S_TILL+EPS, 10):
                    for aggr in range(3):  # loop on aggressiveness levels
                        # check if the state is valid
                        T = QuinticMotionPredicatesCreator.calc_T_s(time_weights[aggr], w_J[aggr], v0, a0, vT, s)
                        vel_acc_in_limits, is_safe, poly_coefs = QuinticMotionPredicatesCreator.check_action_validity(
                            T, np.array([v0]), np.array([vT]), np.array([s]), np.array([a0]))
                        time_in_limits = NumpyUtils.is_in_limits(T, BP_ACTION_T_LIMITS)
                        is_valid_state = vel_acc_in_limits[0] and time_in_limits[0] and is_safe[0]

                        if is_valid_state:
                            # calculate velocity & acceleration profile
                            times = np.arange(0, T[0] + EPS, 0.1)
                            vel_poly = Math.polyder2d(poly_coefs, m=1)
                            acc_poly = Math.polyder2d(poly_coefs, m=2)
                            vel_samples = Math.polyval2d(vel_poly, times)[0]
                            acc_samples = Math.polyval2d(acc_poly, times)[0]

                            # calculate acceleration profile rate (late braking gets higher rate than early braking)
                            all_T_s[(wi, v0, vT, s, aggr)] = T[0]
                            all_acc_samples[(wi, v0, vT, s, aggr)] = acc_samples
                            rate = get_braking_quality(acc_samples)
                            acc_rate[(wi, v0, vT, s, aggr)] = rate

                            # add velocity & acceleration profile to the metric logger
                            print('wi=%d v0=%.2f vT=%.2f s=%.2f aggr=%d' % (wi, v0, vT, s, aggr))
                            key_str = str(wi) + '_' + str(v0) + '_' + str(vT) + '_' + str(s) + '_' + str(aggr)
                            ml.bind(key_str=key_str)
                            ml.bind(w=wi, v0=v0, vT=vT, s=s, ag=aggr)
                            for i, acc in enumerate(acc_samples):
                                ml.bind(time=times[i], vel_sample=vel_samples[i], acc_sample=acc)
                                ml.report()


if __name__ == '__main__':
    jerk_time_weights_optimization()
    #calc_braking_quality_and_print_graphs()
