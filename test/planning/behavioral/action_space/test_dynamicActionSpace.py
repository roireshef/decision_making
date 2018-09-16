from logging import Logger
from typing import List

import copy
import numpy as np

from decision_making.src.global_constants import SPECIFICATION_MARGIN_TIME_DELAY, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, LON_ACC_LIMITS, BP_ACTION_T_LIMITS, BP_JERK_S_JERK_D_TIME_WEIGHTS, EPS, \
    VELOCITY_LIMITS, TRAJECTORY_TIME_RESOLUTION, SAFETY_MARGIN_TIME_DELAY
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, ActionSpec
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.types import FS_SX, FS_SV, FS_SA, FrenetTrajectories2D, LIMIT_MIN, FrenetTrajectory2D
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, Poly1D
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import ObjectSize, EgoState, DynamicObject, State
from decision_making.src.utils.metric_logger import MetricLogger
from rte.python.logger.AV_logger import AV_Logger

from decision_making.test.planning.behavioral.behavioral_state_fixtures import behavioral_grid_state, \
    follow_vehicle_recipes_towards_front_cells, state_with_sorrounding_objects, pg_map_api, \
    follow_vehicle_recipes_towards_front_same_lane


# specifies follow actions for front vehicles in 3 lanes. longitudinal and latitudinal coordinates
# of terminal states in action specification should be as expected
def test_specifyGoals_stateWithSorroundingObjects_specifiesFollowTowardsFrontCellsWell(
        behavioral_grid_state: BehavioralGridState,
        follow_vehicle_recipes_towards_front_cells: List[DynamicActionRecipe]):
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)
    actions = dynamic_action_space.specify_goals(follow_vehicle_recipes_towards_front_cells, behavioral_grid_state)

    targets = [behavioral_grid_state.road_occupancy_grid[(recipe.relative_lane, recipe.relative_lon)][0]
               for recipe in follow_vehicle_recipes_towards_front_cells]

    # terminal action-spec latitude equals the current latitude of target vehicle
    expected_latitudes = [1.8, 1.8, 1.8, 5.4, 5.4, 5.4, 9, 9, 9]
    latitudes = [action.d for action in actions]
    np.testing.assert_array_almost_equal(latitudes, expected_latitudes)

    # terminal action-spec longitude equals the terminal longitude of target vehicle
    # (according to prediction at the terminal time)
    expected_longitudes = [target.dynamic_object.map_state.road_fstate[FS_SX] +
                           target.dynamic_object.map_state.road_fstate[FS_SV] * actions[i].t -
                           actions[i].v * SPECIFICATION_MARGIN_TIME_DELAY -
                           LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT -
                           behavioral_grid_state.ego_state.size.length / 2 - targets[i].dynamic_object.size.length / 2
                           for i, target in enumerate(targets)]
    longitudes = [action.s for action in actions]
    np.testing.assert_array_almost_equal(longitudes, expected_longitudes)


def test_specifyGoals_followSlowFrontObject_specsComplyAccAndTimeLimits(
        follow_vehicle_recipes_towards_front_same_lane: List[DynamicActionRecipe]):
    """
    Test specify_goals for dynamic actions towards much slower front object.
    Verify that choosing of maximal roots of 'time_cost_function_derivative_coefs' polynomial create specs
    that comply with acceleration and time limits, while choosing minimal roots does not comply the limits.
    :param follow_vehicle_recipes_towards_front_same_lane:
    """
    logger = AV_Logger.get_logger()
    predictor = RoadFollowingPredictor(logger)
    dynamic_action_space = DynamicActionSpace(logger, predictor, filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)
    road_id = 20

    ego_init_fstate = np.array([13.3209042, 8.3501435, 0.08652155, 1.8, 0, 0])
    obj_map_state = MapState(np.array([29.7766345, 2.77391842, 0, 1.8, 0, 0]), road_id)
    # ego_init_fstate = np.array([16.0896022, 11.1188409, 0.0865254213, 1.8, 0, 0])
    # obj_map_state = MapState(np.array([27.7781913, 2.77510263, 0, 1.8, 0, 0]), road_id)

    ego_map_state = MapState(ego_init_fstate, road_id)
    ego = EgoState.create_from_map_state(0, 0, ego_map_state, ObjectSize(5, 2, 0), 0)
    obj = DynamicObject.create_from_map_state(1, 0, obj_map_state, ObjectSize(2.66, 1.33, 0), 0)
    state = State(None, [obj], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    actions = dynamic_action_space.specify_goals(follow_vehicle_recipes_towards_front_same_lane, behavioral_state)
    specs = [spec for spec in actions if spec is not None]

    acc_in_limits = np.zeros(len(specs))
    time_in_limits = np.zeros(len(specs))
    for spec_idx, spec in enumerate(specs):
        print('spec = %s' % spec)
        time_samples = np.arange(0, spec.t, 0.1)
        constraints_s = np.array([[ego_init_fstate[FS_SX], ego_init_fstate[FS_SV], ego_init_fstate[FS_SA], spec.s, spec.v, 0]])
        poly_s = WerlingPlanner._solve_1d_poly(constraints_s, spec.t, QuinticPoly1D)
        ftraj = QuinticPoly1D.polyval_with_derivatives(np.array(poly_s), time_samples)[0]
        acc_in_limits[spec_idx] = np.logical_and(LON_ACC_LIMITS[0] < ftraj[:, FS_SA], ftraj[:, FS_SA] < LON_ACC_LIMITS[1]).all()
        time_in_limits[spec_idx] = spec.t < BP_ACTION_T_LIMITS[1]

    assert np.any(np.logical_and(acc_in_limits, time_in_limits))


LON_SAFETY_ACCEL_DURING_DELAY = 3

def test_stress():

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
    # w12_range = np.geomspace(3, 6, num=8)
    # w01_range = np.geomspace(20, 50, num=8)
    # w01, w12, w2 = np.meshgrid(w01_range, w12_range, w2_range)
    # w01, w12, w2 = np.ravel(w01), np.ravel(w12), np.ravel(w2)
    # weights = np.c_[w01 * w12 * w2, w12 * w2, w2]

    #
    # compare a pair of weights sets (or test a single set)
    #
    # weights = np.array([[0.426, 0.139, 0.072], [8.5, 0.34, 0.08]])  # global maximum over sets vs. local max with high weights
    # weights = np.array([BP_JERK_S_JERK_D_TIME_WEIGHTS[:, 0], [8.5, 0.34, 0.08]])  # original set vs. local max with high weights
    weights = np.array([[8.5, 0.34, 0.08]])  # local max with high weights
    # weights = np.array([BP_JERK_S_JERK_D_TIME_WEIGHTS[:, 0]])

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


def test_braking_print_graphs():

    MetricLogger.init('VelProfiles')
    ml = MetricLogger.get_logger()

    v0_from = 4
    v0_till = 24
    vT_from = 0
    vT_till_rel = -2
    v_step = 4
    s_from = 40
    s_till = 100
    a0 = 0.
    w_J = np.array([8.5, 0.34, 0.08])

    aggressiveness = []
    states = []
    T_arr = []
    vel_poly_coefs = []
    extrema = []

    zeros = np.zeros(3)

    for v0 in np.arange(v0_from, v0_till+EPS, v_step):
        for vT in np.arange(vT_from, v0 + vT_till_rel + EPS, v_step):
            for s in np.arange(s_from, s_till+EPS, 10):

                cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
                    w_T=BP_JERK_S_JERK_D_TIME_WEIGHTS[:, 2], w_J=w_J, dx=s, a_0=a0, v_0=v0, v_T=vT, T_m=SPECIFICATION_MARGIN_TIME_DELAY)
                real_roots = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, np.inf]))
                T = np.fmax.reduce(real_roots, axis=-1)

                A = QuinticPoly1D.time_constraints_tensor(T)
                A_inv = np.linalg.inv(A)
                constraints = np.c_[zeros, v0 + zeros, a0 + zeros, s + vT * (T - SPECIFICATION_MARGIN_TIME_DELAY), vT + zeros, zeros]
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

                lowest_aggr = -1
                T_s = -1
                vel_poly = np.zeros(5)
                extremum = np.zeros(9)
                if valid_state.any():  # OR on aggressiveness levels
                    lowest_aggr = np.argmax(valid_state)
                    T_s = T[lowest_aggr]
                    times = np.arange(0, T_s + EPS, 0.1)
                    vel_poly = Math.polyder2d(poly_coefs[lowest_aggr][np.newaxis], m=1)[0]
                    vel_samples = Math.polyval2d(vel_poly[np.newaxis], times)[0]
                    mini = np.argmin(vel_samples)
                    maxi = np.argmax(vel_samples)
                    minv = vel_samples[mini]
                    maxv = vel_samples[maxi]
                    if 0 < mini < len(times)-1 and 0 < maxi < len(times)-1:
                        extremum = np.array([[0, times[mini], times[maxi], T_s, s], [v0, minv, maxv, vT, s]])
                    elif 0 < mini < len(times)-1:
                        extremum = np.array([[0, times[mini], 0, T_s, s], [v0, minv, 0, vT, s]])
                    elif 0 < maxi < len(times)-1:
                        extremum = np.array([[0, 0, times[maxi], T_s, s], [v0, 0, maxv, vT, s]])

                states.append(np.array([v0, vT, s]))
                T_arr.append(T_s)
                aggressiveness.append(lowest_aggr)
                vel_poly_coefs.append(vel_poly)
                extrema.append(extremum)

                if valid_state.any():  # OR on aggressiveness levels
                    ml.bind(v0=v0, vT=vT, s=s)
                    for i in range(len(vel_samples)):
                        ml.bind(time=times[i], sample=vel_samples[i])
                        ml.report()

    a=0


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
                                  np.linspace(s_arr[i], s_arr[i] + v_arr[i] * t_arr[i] + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, len(ego_trajectory)),
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
