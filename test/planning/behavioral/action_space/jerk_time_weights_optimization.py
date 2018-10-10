from typing import List

import numpy as np
import copy

from decision_making.src.global_constants import EPS, SPECIFICATION_MARGIN_TIME_DELAY, BP_JERK_S_JERK_D_TIME_WEIGHTS, \
    BP_ACTION_T_LIMITS, VELOCITY_LIMITS, LON_ACC_LIMITS, TRAJECTORY_TIME_RESOLUTION, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_MARGIN_TIME_DELAY
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.types import FrenetTrajectories2D, LIMIT_MIN, FS_SX, FS_SV
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, Poly1D
from decision_making.src.utils.metric_logger import MetricLogger
from decision_making.src.utils.safety_utils import SafetyUtils


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
    S_MAX = 160  # max distance between two objects in the states grid

    # weights grid ranges
    W2_FROM = 0.01  # min of the range of w2 weight
    W2_TILL = 0.16  # max of the range of w2 weight
    W12_RATIO_FROM = 1.2  # min of the range of ratio w1/w2
    W12_RATIO_TILL = 32   # max of the range of ratio w1/w2
    W01_RATIO_FROM = 1.2  # min of the range of ratio w0/w1
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
    if test_full_range:
        # test a full range of weights (~8 minutes)
        weights = create_full_range_of_weights(W2_FROM, W2_TILL, W12_RATIO_FROM, W12_RATIO_TILL,
                                               W01_RATIO_FROM, W01_RATIO_TILL, GRID_RESOLUTION)
    else:  # compare a pair of weights sets
        weights = np.array([[8.5, 0.34, 0.08], [16, 1.6, 0.08]])

    # remove trivial states, for which T_s = 0
    non_trivial_states = np.where(~np.logical_and(np.isclose(v0, vT), np.isclose(vT * SPECIFICATION_MARGIN_TIME_DELAY, s)))
    v0, vT, a0, s = v0[non_trivial_states], vT[non_trivial_states], a0[non_trivial_states], s[non_trivial_states]
    states_num = v0.shape[0]
    print('states num = %d' % (states_num))

    # the second dimension (of size 2) is for switching between min/max roots
    is_good_state = np.full((weights.shape[0], states_num), False)  # states that passed all limits & safety
    T_s = np.zeros((weights.shape[0], states_num, weights.shape[1]))

    for wi, w in enumerate(weights):  # loop on weights' sets
        vel_acc_in_limits = np.zeros((states_num, weights.shape[1]))
        safe_actions = copy.deepcopy(vel_acc_in_limits)
        for aggr in range(3):  # loop on aggressiveness levels
            # calculate time horizon for all states
            T_s[wi, :, aggr] = T = calculate_T_s(v0, vT, s, a0, BP_JERK_S_JERK_D_TIME_WEIGHTS[aggr, 2], w[aggr])
            # calculate states validity wrt velocity & acceleration limits
            vel_acc_in_limits[:, aggr], safe_actions[:, aggr], _ = check_action_validity(T, v0, vT, s, a0)

        # combine velocity & acceleration limits with time limits and safety, to obtain states validity
        time_in_limits = NumpyUtils.is_in_limits(T_s[wi], BP_ACTION_T_LIMITS)
        in_limits = np.logical_and(vel_acc_in_limits, np.logical_and(time_in_limits, safe_actions))
        is_good_state[wi] = in_limits.any(axis=-1)  # OR on aggressiveness levels
        print('weight: %7.3f %.3f %.3f: failed %d' % (w[0], w[1], w[2], np.sum(~is_good_state[wi])))

    if test_full_range:
        # Monitor a quality of the best set of weights (maximal roots).
        print_success_map_for_weights_set(v0_range, vT_range, s_range, v0, vT, s, is_good_state, weights)
    else:
        # Compare between two sets of weights (maximal roots).
        print_comparison_between_two_weights_sets(v0_range, vT_range, s_range, v0, vT, s, is_good_state)


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
    w2_range = np.geomspace(w2_from, w2_till, num=resolution)
    w12_range = np.geomspace(w12_ratio_from, w12_ratio_till, num=resolution)
    w01_range = np.geomspace(w01_ratio_from, w01_ratio_till, num=resolution)
    w01, w12, w2 = np.meshgrid(w01_range, w12_range, w2_range)
    w01, w12, w2 = np.ravel(w01), np.ravel(w12), np.ravel(w2)
    weights = np.c_[w01 * w12 * w2, w12 * w2, w2]
    return weights


def calculate_T_s(v0_grid: np.array, vT_grid: np.array, s_grid: np.array, a0_grid: np.array,
                  time_weights: np.array, jerk_weights: np.array) -> np.array:
    """
    Given time-jerk weights and v0, vT, s grids, calculate T_s minimizing the cost function.
    :param v0_grid: v0 of meshgrid of v0_range, vT_range, s_range
    :param vT_grid: vT of meshgrid of v0_range, vT_range, s_range
    :param a0_grid: initial acceleration of meshgrid of v0_range, vT_range, s_range
    :param s_grid: s of meshgrid of v0_range, vT_range, s_range
    :param jerk_weights:
    :return: array of optimal T_s (time horizon) for every state from the grid (v0, vT, a0, s)
    """
    cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
        w_T=time_weights, w_J=jerk_weights, dx=s_grid,
        a_0=a0_grid, v_0=v0_grid, v_T=vT_grid, T_m=SPECIFICATION_MARGIN_TIME_DELAY)
    real_roots = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, np.inf]))
    T_s = np.fmax.reduce(real_roots, axis=-1)
    T_s[np.where(T_s == 0)] = 0.01  # prevent zero times
    return T_s


def check_action_validity(T: np.array, v0_grid: np.array, vT_grid: np.array, s_grid: np.array, a0_grid: np.array) -> \
        [np.array, np.array]:
    """
    Given time horizon T (T_s) and grid of states, calculate validity wrt velocity & acceleration limits,
    and safety of each state.
    :param T: optimal T_s for every grid state
    :param v0_grid: v0 of meshgrid of v0_range, vT_range, s_range
    :param vT_grid: vT of meshgrid of v0_range, vT_range, s_range
    :param a0_grid: initial acceleration of meshgrid of v0_range, vT_range, s_range
    :param s_grid: s of meshgrid of v0_range, vT_range, s_range
    :return: two boolean arrays: (1) is in limits of velocity & acceleration, (2) is the baseline trajectory safe
    """
    zeros = np.zeros(v0_grid.shape[0])
    A = QuinticPoly1D.time_constraints_tensor(T)
    A_inv = np.linalg.inv(A)
    constraints = np.c_[zeros, v0_grid, a0_grid, s_grid + vT_grid * (T - SPECIFICATION_MARGIN_TIME_DELAY), vT_grid, zeros]
    poly_coefs = QuinticPoly1D.zip_solve(A_inv, constraints)
    # check acc & vel limits
    poly_coefs[np.where(poly_coefs[:, 0] == 0), 0] = EPS
    acc_in_limits = QuinticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)
    vel_in_limits = QuinticPoly1D.are_velocities_in_limits(poly_coefs, T, VELOCITY_LIMITS)
    is_in_limits = np.logical_and(acc_in_limits, vel_in_limits)
    # check safety
    action_specs = [ActionSpec(t=T[i], v=vT_grid[i], s=s_grid[i], d=0) for i in range(v0_grid.shape[0])]
    is_safe = get_lon_safety_for_action_specs(poly_coefs, action_specs, 0)
    return is_in_limits, is_safe, poly_coefs


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
    failed_num = np.sum(~is_valid_state[best_wi])
    print('best weights for max: %s; failed %d (%.2f)' % (weights[best_wi], failed_num, float(failed_num)/is_valid_state.shape[-1]))

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
    :param is_valid_state_w1: boolean array: True if the state is valid for weights1 (complies all thresholds & safety)
    :param is_valid_state_w0: boolean array: True if the state is valid for weights0 (complies all thresholds & safety)
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
                        T = calculate_T_s(v0, vT, s, a0, BP_JERK_S_JERK_D_TIME_WEIGHTS[aggr, 2], w_J[aggr])
                        vel_acc_in_limits, is_safe, poly_coefs = \
                            check_action_validity(T, np.array([v0]), np.array([vT]), np.array([s]), np.array([a0]))
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
                            rate = rate_acceleration_quality(acc_samples)
                            acc_rate[(wi, v0, vT, s, aggr)] = rate

                            # add velocity & acceleration profile to the metric logger
                            print('wi=%d v0=%.2f vT=%.2f s=%.2f aggr=%d' % (wi, v0, vT, s, aggr))
                            key_str = str(wi) + '_' + str(v0) + '_' + str(vT) + '_' + str(s) + '_' + str(aggr)
                            ml.bind(key_str=key_str)
                            ml.bind(w=wi, v0=v0, vT=vT, s=s, ag=aggr)
                            for i, acc in enumerate(acc_samples):
                                ml.bind(time=times[i], vel_sample=vel_samples[i], acc_sample=acc)
                                ml.report()


def rate_acceleration_quality(acc_samples: np.array) -> float:
    """
    calculate normalized center of mass of braking: late braking gets higher rate than early braking
    :param acc_samples:
    :return: center of mass of acc_samples from the interval [0..1]
    """
    cum_acceleration_time = cum_acceleration = 0
    for i, acc in enumerate(acc_samples):
        if acc < 0:
            cum_acceleration_time += i * acc
            cum_acceleration += acc
    return cum_acceleration_time / (cum_acceleration * len(acc_samples))


def get_lon_safety_for_action_specs(poly_coefs: np.array, action_specs: List[ActionSpec],
                                    cars_size_margin: float) -> np.array:
    """
    Given polynomial coefficients and action specs for each polynomial, calculate longitudinal safety
    w.r.t. the front cars with constant velocities, described by the specs.
    :param poly_coefs: 2D matrix Nx6: N quintic polynomials of ego
    :param action_specs: list of N action specs
    :param cars_size_margin: sum of half sizes of the cars
    :return: boolean array of size N: longitudinal safety for each spec
    """
    assert poly_coefs.shape[0] == len(action_specs)
    specs_arr = np.array([[spec.t, spec.s, spec.v] for spec in action_specs])
    t_arr, s_arr, v_arr = np.split(specs_arr, 3, axis=1)
    time_samples = np.arange(0, np.max(t_arr) + EPS, TRAJECTORY_TIME_RESOLUTION)

    # sample polynomials and create ftrajectories_s
    trajectories_s = Poly1D.polyval_with_derivatives(poly_coefs, time_samples)
    ego_trajectories = [trajectory[0:int(t_arr[i] / TRAJECTORY_TIME_RESOLUTION) + 1]
                        for i, trajectory in enumerate(trajectories_s)]

    obj_trajectories = [np.c_[v_arr[i] * SPECIFICATION_MARGIN_TIME_DELAY + cars_size_margin +
                              np.linspace(s_arr[i], s_arr[i] + v_arr[i] * t_arr[i] + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT,
                                          len(ego_trajectory)),
               np.full(len(ego_trajectory), v_arr[i]), np.zeros(len(ego_trajectory))]
         for i, ego_trajectory in enumerate(ego_trajectories)]

    # concatenate all trajectories to a single long trajectory
    ego_trajectory = np.concatenate(ego_trajectories)
    obj_trajectory = np.concatenate(obj_trajectories)

    # calc longitudinal RSS for the long trajectory
    safe_times = SafetyUtils._get_lon_safety(ego_trajectory, SAFETY_MARGIN_TIME_DELAY,
                                             obj_trajectory, SPECIFICATION_MARGIN_TIME_DELAY, cars_size_margin)

    # split the safety results according to the original trajectories
    trajectories_lengths = [len(trajectory) for trajectory in ego_trajectories]
    safe_times = np.split(safe_times, np.cumsum(trajectories_lengths[:-1]))
    # AND on all time samples
    safe_specs = [spec_safety.all() for spec_safety in safe_times]
    return np.array(safe_specs)


if __name__ == '__main__':
    jerk_time_weights_optimization()
    #calc_braking_quality_and_print_graphs()
