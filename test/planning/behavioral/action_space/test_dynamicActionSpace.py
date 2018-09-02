from logging import Logger
from typing import List

import copy
import numpy as np

from decision_making.src.global_constants import SPECIFICATION_MARGIN_TIME_DELAY, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, LON_ACC_LIMITS, BP_ACTION_T_LIMITS, BP_JERK_S_JERK_D_TIME_WEIGHTS, EPS
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.types import FS_SX, FS_SV, FS_SA
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, Poly1D
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import ObjectSize, EgoState, DynamicObject, State
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


def test_stress():

    v_step = 2
    s_step = 10
    v0_range = np.arange(0, 20 + EPS, v_step)
    vT_range = np.arange(0, 20 + EPS, v_step)
    a0_range = np.array([0])  # np.arange(-3, 3 + EPS, 0.5)
    s_range = np.arange(10, 120 + EPS, s_step)

    # v0_range = np.arange(10, 11 + EPS, 0.1)
    # vT_range = np.arange(10, 11 + EPS, 0.1)
    # a0_range = np.arange(-3, 3.1, 0.1)
    # s_range = np.arange(20, 22 + EPS, 0.1)

    v0, vT, a0, s = np.meshgrid(v0_range, vT_range, a0_range, s_range)
    v0, vT, a0, s = np.ravel(v0), np.ravel(vT), np.ravel(a0), np.ravel(s)

    # remove trivial states (T=0)
    good_indices = np.where(~np.logical_and(np.isclose(v0, vT), np.isclose(vT * 2, s)))
    v0, vT, a0, s = v0[good_indices], vT[good_indices], a0[good_indices], s[good_indices]
    zeros = np.zeros(v0.shape[0])
    print('states num = %d' % (v0.shape[0]))

    weights = BP_JERK_S_JERK_D_TIME_WEIGHTS
    T_s_min = np.zeros((v0.shape[0], weights.shape[0]))
    T_s_max = copy.deepcopy(T_s_min)
    acc_in_limits = np.zeros((2, v0.shape[0], weights.shape[0]))
    for aggr in range(3):
        cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[aggr, 2], w_J=weights[aggr, 0], dx=s,
            a_0=a0, v_0=v0, v_T=vT, T_m=SPECIFICATION_MARGIN_TIME_DELAY)
        real_roots = Math.find_real_roots_in_limits(cost_coeffs_s, np.array([0, np.inf]))
        T_s_min[:, aggr] = np.fmin.reduce(real_roots, axis=-1)
        T_s_max[:, aggr] = np.fmax.reduce(real_roots, axis=-1)

        T_s_min[np.where(T_s_min[:, aggr] == 0), aggr] += EPS  # prevent zero times

        for minmax, T_s in enumerate([T_s_min, T_s_max]):
            A = QuinticPoly1D.time_constraints_tensor(T_s[:, aggr])
            A_inv = np.linalg.inv(A)
            constraints = np.c_[zeros, v0, a0, s + vT * (T_s[:, aggr] - SPECIFICATION_MARGIN_TIME_DELAY), vT, zeros]
            poly_coefs = QuinticPoly1D.zip_solve(A_inv, constraints)
            acc_in_limits[minmax, :, aggr] = Poly1D.are_derivatives_in_limits(2, poly_coefs, T_s[:, aggr], LON_ACC_LIMITS)

    print('')
    legal_T_s = np.full((2, T_s_min.shape[0]), False)
    for minmax, T_s in enumerate([T_s_min, T_s_max]):
        # check time limits and acc limits
        legal_T_s[minmax] = np.logical_and(np.logical_and(T_s >= BP_ACTION_T_LIMITS[0], T_s <= BP_ACTION_T_LIMITS[1]),
                                           acc_in_limits[minmax]).any(axis=-1)
        print('%s: total = %d; failed = %d' % ('min' if minmax == 0 else 'max', legal_T_s.shape[1], np.sum(~legal_T_s[minmax])))
    print('min worked, max failed: %d' % np.sum((np.logical_and(legal_T_s[0], ~legal_T_s[1]))))

    # print('\nParams that failed')
    # for i, b in enumerate(legal_T_s[1]):
    #     if not b:
    #         print([v0[i], vT[i], a0[i], s[i], T_s_max[i, 0], T_s_max[i, 1], T_s_max[i, 2]])

    print('\nmin worked, max failed')
    for i, b in enumerate(legal_T_s[1]):
        if not b and legal_T_s[0, i]:
            print([v0[i], vT[i], a0[i], s[i], [T_s_min[i, 0], T_s_min[i, 1], T_s_min[i, 2]],
                                              [T_s_max[i, 0], T_s_max[i, 1], T_s_max[i, 2]]])

    success_map = np.ones((vT_range.shape[0], s_range.shape[0]))
    for i, b in enumerate(legal_T_s[1]):
        if v0[i] == 10 and not b:
            success_map[np.argwhere(vT_range == vT[i]), np.argwhere(s_range == s[i])] = b
    print('')
    print(success_map.astype(int))
