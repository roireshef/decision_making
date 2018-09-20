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


def jerk_time_weights_optimization():
    """
    Create 3D grid of configurations (states): initial velocity, end velocity, initial distance from object.
    Create 3D grid of Jerk-Time weights for 3 aggressiveness levels.
    For each state and for each triplet of weights, check validity of the state based on acceleration limits,
    velocity limits, action time limits and RSS safety.
    Output brief states coverage for all weights sets (number of invalid states for each weights set).
    Output detailed states coverage (3D grid of states) for the best weights set or compare two weights sets.
    """
    # states ranges
    v_step = 2
    v0_range = np.arange(0, 18 + EPS, v_step)
    vT_range = np.arange(0, 18 + EPS, v_step)
    # a0_range = np.arange(-3, 2 + EPS, 1)
    a0_range = np.array([0])
    s_range = np.arange(10, 160 + EPS, 10)

    #
    # test a full range of weights (~8 minutes)
    #
    # w2_range = np.geomspace(0.01, 0.16, num=8)
    # w12_range = np.geomspace(1.2, 32, num=8)
    # w01_range = np.geomspace(1.2, 32, num=8)
    # w01, w12, w2 = np.meshgrid(w01_range, w12_range, w2_range)
    # w01, w12, w2 = np.ravel(w01), np.ravel(w12), np.ravel(w2)
    # weights = np.c_[w01 * w12 * w2, w12 * w2, w2]

    # w2_range = np.geomspace(0.072, 0.08, num=3)
    # w12_range = np.geomspace(20, 30, num=6)
    # w01_range = np.geomspace(5, 10, num=4)
    # w01, w12, w2 = np.meshgrid(w01_range, w12_range, w2_range)
    # w01, w12, w2 = np.ravel(w01), np.ravel(w12), np.ravel(w2)
    # weights = np.c_[w01 * w12 * w2, w12 * w2, w2]

    #
    # compare a pair of weights sets (or test a single set)
    #
    # weights = np.array([[0.426, 0.139, 0.072], [8.5, 0.34, 0.08]])  # global maximum over sets vs. local max with high weights
    # weights = np.array([BP_JERK_S_JERK_D_TIME_WEIGHTS[:, 0], [8.5, 0.34, 0.08]])  # original set vs. local max with high weights
    # weights = np.array([[8.5, 0.34, 0.08], [16, 1.6, 0.08]])  # local max with high weights
    weights = np.array([[12, 2, 0.01], [16, 1.6, 0.08]])

    # create all states
    v0, vT, a0, s = np.meshgrid(v0_range, vT_range, a0_range, s_range)
    v0, vT, a0, s = np.ravel(v0), np.ravel(vT), np.ravel(a0), np.ravel(s)

    # remove trivial states, for which T_s = 0
    non_trivial_states = np.where(~np.logical_and(np.isclose(v0, vT), np.isclose(vT * SPECIFICATION_MARGIN_TIME_DELAY, s)))
    v0, vT, a0, s = v0[non_trivial_states], vT[non_trivial_states], a0[non_trivial_states], s[non_trivial_states]
    states_num = v0.shape[0]
    zeros = np.zeros(states_num)
    print('states num = %d' % (states_num))

    # the second dimension (of size 2) is for switching between min/max roots
    is_good_state = np.full((weights.shape[0], 2, states_num), False)  # states that passed all limits & safety
    T_s = np.zeros((weights.shape[0], 2, states_num, weights.shape[1]))

    for wi, w in enumerate(weights):  # loop on weights' sets
        vel_acc_in_limits = np.zeros((2, states_num, weights.shape[1]))
        safe_actions = copy.deepcopy(vel_acc_in_limits)
        for aggr in range(3):  # loop on aggressiveness levels
            cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
                w_T=BP_JERK_S_JERK_D_TIME_WEIGHTS[aggr, 2], w_J=w[aggr], dx=s,
                a_0=a0, v_0=v0, v_T=vT, T_m=SPECIFICATION_MARGIN_TIME_DELAY)
            real_roots = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, np.inf]))
            T_min = np.fmin.reduce(real_roots, axis=-1)
            T_max = np.fmax.reduce(real_roots, axis=-1)
            T_min[np.where(T_min == 0)] = 0.01  # prevent zero times
            T_max[np.where(T_max == 0)] = 0.01  # prevent zero times
            T_s[wi, 0, :, aggr] = T_min
            T_s[wi, 1, :, aggr] = T_max

            for minmax, T in enumerate([T_min, T_max]):  # switch between min/max roots
                if minmax == 0:
                    continue
                A = QuinticPoly1D.time_constraints_tensor(T)
                A_inv = np.linalg.inv(A)
                constraints = np.c_[zeros, v0, a0, s + vT * (T - SPECIFICATION_MARGIN_TIME_DELAY), vT, zeros]
                poly_coefs = QuinticPoly1D.zip_solve(A_inv, constraints)
                # check acc & vel limits
                poly_coefs[np.where(poly_coefs[:, 0] == 0), 0] = EPS
                acc_in_limits = QuinticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)
                vel_in_limits = QuinticPoly1D.are_velocities_in_limits(poly_coefs, T, VELOCITY_LIMITS)
                vel_acc_in_limits[minmax, :, aggr] = np.logical_and(acc_in_limits, vel_in_limits)
                # check safety
                action_specs = [ActionSpec(t=T[i], v=vT[i], s=s[i], d=0) for i in range(states_num)]
                safe_actions[minmax, :, aggr] = SafetyUtils.get_lon_safety_for_action_specs(poly_coefs, action_specs, 0)

        time_in_limits = NumpyUtils.is_in_limits(T_s[wi], BP_ACTION_T_LIMITS)
        in_limits = np.logical_and(vel_acc_in_limits, np.logical_and(time_in_limits, safe_actions))
        is_good_state[wi] = in_limits.any(axis=-1)  # OR on aggressiveness levels

        print('weight: %7.3f %.3f %.3f: failed %d' % (w[0], w[1], w[2], np.sum(~is_good_state[wi, minmax])))
        # for minmax, T in enumerate(T_s[wi]):
        #   print('%s: total = %d; failed = %d' % ('min' if minmax == 0 else 'max', is_good_state.shape[-1], np.sum(~is_good_state[wi, minmax])))

    good_min = is_good_state[:, 0]  # successes of states for min roots
    good_max = is_good_state[:, 1]  # successes of states for max roots
    best_min_wi = np.argmax(np.sum(good_min, axis=-1))
    best_max_wi = np.argmax(np.sum(good_max, axis=-1))
    best_min = good_min[best_min_wi]
    best_max = good_max[best_max_wi]
    failed_min = np.sum(~best_min)
    failed_max = np.sum(~best_max)
    print('best weights for min: %s; failed %d (%.2f)' % (weights[best_min_wi], failed_min, float(failed_min)/is_good_state.shape[-1]))
    print('best weights for max: %s; failed %d (%.2f)' % (weights[best_max_wi], failed_max, float(failed_max)/is_good_state.shape[-1]))
    # how many states work for min_roots and don't work for max_roots
    print('min worked, max failed: %d' % np.sum(best_min & ~best_max))

    # print('\nList of states that failed for the best set of weights (max roots):')
    # for i, b in enumerate(good_max[best_max]):
    #     if not b:
    #         print([v0[i], vT[i], a0[i], s[i], T_s[best_max, 1, i, 0], T_s[best_max, 1, i, 1], T_s[best_max, 1, i, 2]])

    # list of states that worked for min_roots and didn't work for max_roots
    print('\nmin worked, max failed')
    for i in np.where(~best_max & best_min)[0]:
        print([v0[i], vT[i], a0[i], s[i]])

    #
    # Use for monitoring a quality of the best set of weights (maximal roots).
    # Print tables (vT x s) for the best weights per v0
    #
    for v0_fixed in v0_range:
        success_map = np.ones((vT_range.shape[0], s_range.shape[0]))
        for i, b in enumerate(good_max[best_max_wi]):
            if v0[i] == v0_fixed and not b:
                success_map[np.argwhere(vT_range == vT[i]), np.argwhere(s_range == s[i])] = b
        print('v0_fixed = %d' % v0_fixed)
        print(success_map.astype(int))

    #
    # Use for comparison between two sets of weights (maximal roots).
    # Print tables (vT x s) for weights[1] relatively to weights[0] per v0.
    # Each value in the table means:
    #    1: weights[1] is better than weights[0] for this state
    #   -1: weights[1] is worse than weights[0] for this state
    #    0: weights[1] is like weights[0] for this state
    #
    # for v0_fixed in v0_range:
    #     success_map = np.zeros((vT_range.shape[0], s_range.shape[0]))
    #     for i in range(states_num):  # loop over states' successes for weights[0]
    #         if v0[i] == v0_fixed:
    #             success_map[np.argwhere(vT_range == vT[i]), np.argwhere(s_range == s[i])] = \
    #                 (good_max[1, i].astype(int) - good_max[0, i].astype(int))  # diff between the two weights qualities
    #     print('v0_fixed = %d' % v0_fixed)
    #     print(success_map.astype(int))


def calc_braking_quality_and_print_graphs():
    """
    Loop on list of Jerk-Time weights sets and on initial states (v0, vT, s), calculate trajectory quality,
    according to the relative center of mass of negative accelerations (too late braking is bad).
    Draw graphs of velocity & acceleration profiles for all above settings.
    """
    MetricLogger.init('VelProfiles')
    ml = MetricLogger.get_logger()

    v0_from = 4
    v0_till = 18
    vT_from = 0
    vT_till_rel = -2
    v_step = 4
    s_from = 40  # 40
    s_till = 100  # 100
    a0 = 0.
    w_J_list = [np.array([16, 1.6, 0.08]), np.array([8.5, 0.34, 0.08])]
    # w_J_list = [np.array([12, 2, 0.01]), np.array([16.9, 3.446, 0.108])]
    # w_J_list = [np.array([12, 2, 0.01]), np.array([10.565, 2.155, 0.108])]
    # w_J_list = [np.array([12, 2, 0.01]), np.array([11.364, 1.451, 0.072])]
    # w_J_list = [np.array([12, 2, 0.01]), np.array([12.7, 1.6, 0.08])]
    # w_J_list = [np.array([8.5, 0.34, 0.08]), np.array([11.364, 1.451, 0.072])]
    # w_J_list = [np.array([12, 2, 0.01]), np.array([16, 1.6, 0.08])]

    zeros = np.zeros(3)

    all_acc_samples = {}
    acc_rate = {}
    all_T_s = {}
    failure_reason = {}

    new_states = []
    old_states = []
    old_rates_sum = new_rates_sum = 0

    for wi, w_J in enumerate(w_J_list):
        for v0 in np.arange(v0_from, v0_till+EPS, v_step):
            for vT in np.arange(vT_from, v0 + vT_till_rel + EPS, v_step):
                for s in np.arange(s_from, s_till+EPS, 10):

                    cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
                        w_T=BP_JERK_S_JERK_D_TIME_WEIGHTS[:, 2], w_J=w_J, dx=s, a_0=a0, v_0=v0, v_T=vT,
                        T_m=SPECIFICATION_MARGIN_TIME_DELAY)
                    real_roots = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, np.inf]))
                    T = np.fmax.reduce(real_roots, axis=-1)

                    A = QuinticPoly1D.time_constraints_tensor(T)
                    A_inv = np.linalg.inv(A)
                    constraints = np.c_[zeros, v0 + zeros, a0 + zeros, s + vT * (T - SPECIFICATION_MARGIN_TIME_DELAY),
                                        vT + zeros, zeros]
                    poly_coefs = QuinticPoly1D.zip_solve(A_inv, constraints)
                    # check acc & vel limits
                    acc_in_limits = QuinticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)
                    vel_in_limits = QuinticPoly1D.are_velocities_in_limits(poly_coefs, T, VELOCITY_LIMITS)
                    vel_acc_in_limits = np.logical_and(acc_in_limits, vel_in_limits)
                    # check safety
                    action_specs = [ActionSpec(t=T[aggr], v=vT, s=s, d=0) for aggr in range(3)]
                    safe_actions = SafetyUtils.get_lon_safety_for_action_specs(poly_coefs, action_specs, 0)

                    time_in_limits = NumpyUtils.is_in_limits(T, BP_ACTION_T_LIMITS)
                    valid_state = np.logical_and(vel_acc_in_limits, np.logical_and(time_in_limits, safe_actions))

                    for ag in range(3):
                        if valid_state[ag]:  # OR on aggressiveness levels
                            T_s = T[ag]
                            times = np.arange(0, T_s + EPS, 0.1)
                            vel_poly = Math.polyder2d(poly_coefs[ag][np.newaxis], m=1)[0]
                            acc_poly = Math.polyder2d(poly_coefs[ag][np.newaxis], m=2)[0]
                            vel_samples = Math.polyval2d(vel_poly[np.newaxis], times)[0]
                            acc_samples = Math.polyval2d(acc_poly[np.newaxis], times)[0]

                            all_T_s[(wi, v0, vT, s, ag)] = T_s
                            all_acc_samples[(wi, v0, vT, s, ag)] = acc_samples
                            rate = rate_acceleration_quality(acc_samples)
                            acc_rate[(wi, v0, vT, s, ag)] = rate
                            if wi == 1:
                                new_states.append((v0, vT, s, ag))
                                new_rates_sum += rate
                            else:
                                old_states.append((v0, vT, s, ag))
                                old_rates_sum += rate

                            key_str = str(wi) + '_' + str(v0) + '_' + str(vT) + '_' + str(s) + '_' + str(ag)
                            ml.bind(key_str=key_str)
                            ml.bind(w=wi, v0=v0, vT=vT, s=s, ag=ag)
                            for i, acc in enumerate(acc_samples):
                                print('wi=%d; time=%.2f' % (wi, times[i]))
                                ml.bind(time=times[i], vel_sample=vel_samples[i], acc_sample=acc)
                                ml.report()

                            # if wi == 1 and (0, v0, vT, s, ag) in acc_rate and rate > 0.5 and acc_rate[(0, v0, vT, s, ag)] < rate:
                            #     worsen_rates.append([acc_rate[(0, v0, vT, s, ag)], acc_rate[(1, v0, vT, s, ag)]])
                            #     worsen_states.append((v0, vT, s, ag))

                        elif wi == 1:
                            if not acc_in_limits[ag]:
                                failure_reason[(v0, vT, s, ag)] = "acc"
                            elif not vel_in_limits[ag]:
                                failure_reason[(v0, vT, s, ag)] = "vel"
                            elif T[ag] < 2:
                                failure_reason[(v0, vT, s, ag)] = "short"
                            elif T[ag] > 20:
                                failure_reason[(v0, vT, s, ag)] = "long"
                            else:
                                failure_reason[(v0, vT, s, ag)] = "unsafe"


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


LON_SAFETY_ACCEL_DURING_DELAY = 3

class SafetyUtils:

    @staticmethod
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

    @staticmethod
    def _get_lon_safety(ego_trajectories: FrenetTrajectories2D, ego_response_time: float,
                        obj_trajectories: FrenetTrajectories2D, obj_response_time: float,
                        margin: float, max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        Longitudinal safety between two objects considers only their longitudinal data: longitude and longitudinal velocity.
        Longitudinal RSS formula considers distance reduction during the reaction time and difference between
        objects' braking distances.
        An object is defined safe if it's safe either longitudinally OR laterally.
        :param ego_trajectories: ego frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x 6
        :param ego_response_time: [sec] ego response time
        :param obj_trajectories: object's frenet trajectories: 2D matrix: timestamps_num x 6
        :param obj_response_time: [sec] object's response time
        :param margin: [m] cars' lengths half sum
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] longitudinal safety per timestamp. 2D matrix shape: traj_num x timestamps_num
        """
        # extract the relevant longitudinal data from the trajectories
        ego_lon, ego_vel = ego_trajectories[..., FS_SX], ego_trajectories[..., FS_SV]
        obj_lon, obj_vel = obj_trajectories[..., FS_SX], obj_trajectories[..., FS_SV]

        # determine which object is in front (per trajectory and timestamp)
        lon_relative_to_obj = ego_lon - obj_lon
        sign_of_lon_relative_to_obj = np.sign(lon_relative_to_obj)
        ego_ahead = (sign_of_lon_relative_to_obj > 0).astype(int)

        # The worst-case velocity of the rear object (either ego or another object) may increase during its reaction
        # time, since it may accelerate before it starts to brake.
        ego_vel_after_reaction_time = ego_vel + (1-ego_ahead) * ego_response_time * LON_SAFETY_ACCEL_DURING_DELAY
        obj_vel_after_reaction_time = obj_vel + ego_ahead * obj_response_time * LON_SAFETY_ACCEL_DURING_DELAY

        # longitudinal RSS formula considers distance reduction during the reaction time and difference between
        # objects' braking distances
        ego_acceleration_dist = 0.5 * LON_SAFETY_ACCEL_DURING_DELAY * SAFETY_MARGIN_TIME_DELAY ** 2
        obj_acceleration_dist = 0.5 * LON_SAFETY_ACCEL_DURING_DELAY * SPECIFICATION_MARGIN_TIME_DELAY ** 2
        safe_dist = np.maximum(np.divide(sign_of_lon_relative_to_obj * (obj_vel_after_reaction_time ** 2 -
                                                                        ego_vel_after_reaction_time ** 2),
                                         2 * max_brake), 0) + \
                    (1 - ego_ahead) * (ego_vel * ego_response_time + ego_acceleration_dist) + \
                    ego_ahead * (obj_vel * obj_response_time + obj_acceleration_dist) + margin

        return sign_of_lon_relative_to_obj * lon_relative_to_obj > safe_dist


if __name__ == '__main__':
    jerk_time_weights_optimization()
