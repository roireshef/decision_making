from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.types import LIMIT_MAX
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D

import numpy as np
import re

from decision_making.src.global_constants import EPS, BP_JERK_S_JERK_D_TIME_WEIGHTS, BP_ACTION_T_LIMITS, \
    TRAJECTORY_TIME_RESOLUTION, SPECIFICATION_HEADWAY, SAFETY_HEADWAY, LON_ACC_LIMITS, VELOCITY_LIMITS, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT
from decision_making.src.planning.utils.math_utils import Math


# This file deals with finding "optimal" weights for time-jerk cost function for dynamic actions.
# The "optimality" consists of two components:
# 1. Coverage of feasible (unfiltered) semantic actions in different scenarios. The function
#    jerk_time_weights_optimization() deals with it.
# 2. Velocity/acceleration profile of the actions should be convenient or good feel for the passengers, particularly
#    to prevent too late braking. The function calc_braking_quality_and_print_graphs() draws these profiles
#    and calculates the convenience rate.


class TimeJerkWeightsOptimization:

    @staticmethod
    def jerk_time_weights_optimization_for_slower_front_car():
        """
        Create 3D grid of configurations (states): initial velocity, end velocity, initial distance from object.
        Create 3D grid of Jerk-Time weights for 3 aggressiveness levels.
        For each state and for each triplet of weights, check validity of the state based on acceleration limits,
        velocity limits, action time limits and RSS safety.
        Output brief states coverage for all weights sets (number of invalid states for each weights set).
        Output detailed states coverage (3D grid of states) for the best weights set or compare two weights sets.
        """
        # states grid ranges
        V_MIN = 0
        V_MAX = 27   # max velocity in the states grid
        V_STEP = 1  # velocity step in the states grid
        S_MIN = 10   # min distance between two objects in the states grid
        S_MAX = 100  # max distance between two objects in the states grid
        S_STEP = 2.

        # weights grid ranges
        W2_FROM = 0.001  # min of the range of w2 weight
        W2_TILL = 0.02  # max of the range of w2 weight
        W12_RATIO_FROM = 3  # min of the range of ratio w1/w2
        W12_RATIO_TILL = 64   # max of the range of ratio w1/w2
        W01_RATIO_FROM = 1.2  # min of the range of ratio w0/w1
        W01_RATIO_TILL = 64   # max of the range of ratio w0/w1
        GRID_RESOLUTION = 10   # the weights grid resolution

        TYPICAL_CAR_LENGTH = 5
        safety_margin = LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + TYPICAL_CAR_LENGTH
        specify_margin = LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT + TYPICAL_CAR_LENGTH

        # create ranges of the grid of states
        v0_range = np.arange(V_MIN, V_MAX + EPS, 2)
        vT_range = np.arange(V_MIN, V_MAX + EPS, V_STEP)
        a0_range = np.array([0])
        s_range = np.arange(S_MIN, S_MAX + EPS, S_STEP)

        # create the grid of states
        v0, vT, a0, s = np.meshgrid(v0_range, vT_range, a0_range, s_range)
        braking = np.where(np.logical_and(0 < v0 - vT, v0 - vT <= 8))
        v0, vT, a0, s = v0[braking], vT[braking], a0[braking], s[braking]
        limited_headway = np.where((s >= v0 * SAFETY_HEADWAY) & (s < v0 * 3))
        v0, vT, a0, s = v0[limited_headway], vT[limited_headway], a0[limited_headway], s[limited_headway]
        v0, vT, a0, s = np.ravel(v0), np.ravel(vT), np.ravel(a0), np.ravel(s)

        # Decide whether to create the full range of weights or just compare two sets of weights.
        test_full_range = True
        # s_weights is a matrix Wx3, where W is a set of jerk weights (time weight is constant) for 3 aggressiveness levels
        if test_full_range:
            # test a full range of weights (~8 minutes)
            s_weights = TimeJerkWeightsOptimization.create_full_range_of_weights(
                W2_FROM, W2_TILL, W12_RATIO_FROM, W12_RATIO_TILL, W01_RATIO_FROM, W01_RATIO_TILL, GRID_RESOLUTION)
        else:  # compare a pair of weights sets
            # s_weights = np.array([[0.7, 0.015, 0.005], [0.7, 0.015, 0.015]])
            s_weights = np.array([[12, 2, 0.01], [6, 0.2, 0.004]])

        # remove trivial states, for which T_s = 0
        non_trivial_states = np.where(~np.logical_and(np.isclose(v0, vT), np.isclose(vT * SPECIFICATION_HEADWAY, s)))
        v0, vT, a0, s = v0[non_trivial_states], vT[non_trivial_states], a0[non_trivial_states], s[non_trivial_states]
        states_num = v0.shape[0]
        print('states num = %d' % (states_num))

        valid_states_mask = np.full((s_weights.shape[0], states_num), False)  # states that passed all limits & safety
        profile_rates = np.zeros_like(s_weights)

        for wi, w in enumerate(s_weights):  # loop on weights' sets
            in_limits = np.full((s_weights.shape[1], states_num), False)
            tot_kinematics_in_limits = np.zeros(s_weights.shape[1])
            tot_safe = np.zeros(s_weights.shape[1])
            tot_time_in_limits = np.zeros(s_weights.shape[1])

            for aggr in range(s_weights.shape[1]):  # loop on aggressiveness levels
                # calculate time horizon for all states
                # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
                time_weights = np.full(vT.shape[0], BP_JERK_S_JERK_D_TIME_WEIGHTS[aggr, 2])
                jerk_weights = np.full(vT.shape[0], w[aggr])

                # calculate planning time like in DynamicActionSpace.specify_goal
                T = DynamicActionSpace.calc_T_s(time_weights, jerk_weights, s - specify_margin, a0[0], v0, vT)

                # extract valid actions
                valid_idxs = np.where(np.logical_and(~np.isnan(T), T > 0))[0]
                tot_time_in_limits[aggr] = valid_idxs.shape[0]

                in_limits[aggr, valid_idxs], profile_rates[wi, aggr], tot_kinematics_in_limits[aggr], tot_safe[aggr] = \
                    TimeJerkWeightsOptimization.check_actions_in_limits(T[valid_idxs], a0[valid_idxs], v0[valid_idxs],
                                                                        vT[valid_idxs], s[valid_idxs],
                                                                        specify_margin, safety_margin)

            # combine velocity & acceleration limits with time limits and safety, to obtain states validity
            valid_states_mask[wi] = in_limits.any(axis=0)  # OR on aggressiveness levels

            print('weight: %7.3f %.3f %.3f: passed %.1f%%\t\tvel_acc %s   safety %s   time %s;\tprofile %s' %
                  (w[0], w[1], w[2], np.sum(valid_states_mask[wi])*100./states_num,
                   (tot_kinematics_in_limits*100/states_num).astype(np.int), (tot_safe*100/states_num).astype(np.int),
                   (tot_time_in_limits*100/states_num).astype(np.int), profile_rates[wi]))

        if test_full_range:  # Monitor a quality of the best set of weights (maximal roots).
            TimeJerkWeightsOptimization.print_success_map_for_best_weights_set(v0_range, vT_range, s_range, v0, vT, s,
                                                                               valid_states_mask, s_weights)
        else:  # Compare between two sets of weights (maximal roots).
            TimeJerkWeightsOptimization.print_comparison_between_two_weights_sets(v0_range, vT_range, s_range, v0, vT, s,
                                                                                  valid_states_mask)

    @staticmethod
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

    @staticmethod
    def check_actions_in_limits(T: np.array, a0: np.array, v0: np.array, vT: np.array, s: np.array,
                                specify_margin: float, safety_margin: float):
        """
        given a set of actions, calculate which of them are in limits (kinematics, safety, time limits)
        and return braking quality and limits statistics
        :param T: time horizons array
        :param a0: initial accelerations array
        :param v0: initial velocities array
        :param vT: target velocities array
        :param s: array of initial distances from the target
        :param specify_margin: target distance between cars' centers in addition to the headway
        :param safety_margin: minimal safe distance between cars' centers in addition to the headway
        :return: boolean array of actions in limits, braking quality of actions in limit, statistics data
        """
        # calculate s profile of host & target
        zeros = np.zeros_like(T)
        poly_host = QuinticPoly1D.s_profile_coefficients(a0, v0, vT, s - specify_margin, T, SPECIFICATION_HEADWAY)
        poly_target = np.c_[zeros, zeros, zeros, zeros, vT, s]

        # calculate states validity wrt velocity, acceleration and time limits
        vel_acc_in_limits = KinematicUtils.filter_by_longitudinal_frenet_limits(
            poly_host, T, LON_ACC_LIMITS, VELOCITY_LIMITS, np.array([-np.inf, np.inf]))
        safe_actions = KinematicUtils.are_maintaining_distance(poly_host, poly_target, safety_margin, SAFETY_HEADWAY, np.c_[zeros, T])
        in_limits = np.logical_and(vel_acc_in_limits, safe_actions)

        # Calculate average (on all valid actions) braking profile quality.
        # Late braking (less pleasant for passengers) gets lower rate than early braking.
        braking_rates = TimeJerkWeightsOptimization.calculate_braking_profile_qualities(poly_host, T, a0, v0, vT, s, in_limits)

        return in_limits, braking_rates, np.sum(vel_acc_in_limits), np.sum(safe_actions)

    @staticmethod
    def calculate_braking_profile_qualities(poly_host: np.array, T: np.array, a0: np.array, v0: np.array, vT: np.array,
                                            s: np.array, in_limits: np.array) -> float:
        """
        Given a set of actions, calculate average braking rate (only on valid actions).
        Late braking (less pleasant for passengers) gets higher rate than early braking.
        :param poly_host: 2D array Nx6 of polynomial coefficients of host s profile
        :param T: time horizons array
        :param a0: initial accelerations array
        :param v0: initial velocities array
        :param vT: target velocities array
        :param s: array of initial distances from the target
        :param in_limits: boolean array: which actions are in vel, acc, time limits
        :return: average braking quality for the given actions. Range: [0, 1]; 0 the worst, 1 the best.
        """
        # Calculate average (on all valid actions) braking profile quality.
        # Late braking (less pleasant for passengers) gets lower rate than early braking.
        weights_sum = weighted_rates_sum = 0
        for i in range(T.shape[0]):
            if not in_limits[i]:
                continue
            # calculate velocity & acceleration profile
            time_samples = np.arange(0, T[i] + EPS, TRAJECTORY_TIME_RESOLUTION)
            distance_profile = QuinticPoly1D.distance_from_target(a0[i], v0[i], vT[i], s[i], T[i], SPECIFICATION_HEADWAY)
            acc_poly_coefs = Math.polyder2d(poly_host[i:i + 1], m=2)
            acc_samples = Math.polyval2d(acc_poly_coefs, time_samples)[0]
            # calculate acceleration profile rate
            brake_rate, rate_weight = TimeJerkWeightsOptimization.get_braking_quality(distance_profile(time_samples),
                                                                                      acc_samples)
            if brake_rate is not None:  # else there was not braking
                weighted_rates_sum += rate_weight * brake_rate
                weights_sum += rate_weight

        return weighted_rates_sum / weights_sum if weights_sum > 0 else 0

    @staticmethod
    def get_braking_quality(distances: np.array, acc_samples: np.array) -> [float, float]:
        """
        calculate normalized center of mass of braking: braking close to target (less pleasant) gets lower rate
        :param distances: array of distances from the followed object
        :param acc_samples: array of acceleration samples
        :return: 1. center of mass of negative acc_samples as function of distance from the object;
                    the higher rate the better
                 2. the rate weight: the higher accelerations the higher weight
        """
        max_brake = max(0., -min(acc_samples))
        if max_brake == 0:
            return 0, 0
        brake_idxs = np.where(acc_samples < 0)[0]
        brake_samples = -acc_samples[brake_idxs]
        cum_brake = np.sum(brake_samples)
        return np.sum(brake_samples * distances[brake_idxs]) / (cum_brake * distances[0]), max_brake

    @staticmethod
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
            success_map = 2*np.ones((vT_range.shape[0], s_range.shape[0]))
            for i, b in enumerate(is_valid_state[best_wi]):
                if v0_grid[i] == v0_fixed:
                    success_map[np.argwhere(vT_range == vT_grid[i]), np.argwhere(s_range == s_grid[i])] = b
            print('\nv0_fixed = %d' % v0_fixed)
            s_str = np.array_repr(s_range/v0_fixed).replace('\n', '')[7:-2]
            print('headway: %s' % s_str)
            for vti in range(len(vT_range)):
                arr = success_map[vti].astype(int)
                st = np.array_repr(arr).replace('\n', '')[7:-2]
                st = re.sub(' +', ' ', st)
                # st = re.sub(',', '', st)
                print('vT %.1f: %s' % (vT_range[vti], st))

    @staticmethod
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
            print('\nv0_fixed = %d' % v0_fixed)
            # print(success_map.astype(int))
            for vti in range(len(vT_range)):
                arr = success_map[vti].astype(int)
                st = np.array_repr(arr).replace('\n', '')[7:-2]
                st = re.sub(' +', ' ', st)
                # st = re.sub(',', '', st)
                print('vT %.1f: %s' % (vT_range[vti], st))


if __name__ == '__main__':
    TimeJerkWeightsOptimization.jerk_time_weights_optimization_for_slower_front_car()
