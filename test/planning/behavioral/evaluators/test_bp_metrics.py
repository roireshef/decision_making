import time
from logging import Logger
from unittest.mock import patch

import numpy as np
import pytest
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.data_objects import ActionType, AggressivenessLevel, \
    ActionRecipe, RelativeLane, DynamicActionRecipe, RelativeLongitudinalPosition, StaticActionRecipe
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, MINIMAL_STATIC_ACTION_TIME, AV_TIME_DELAY
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING, \
    DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.evaluators.heuristic_action_spec_evaluator import \
    HeuristicActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.naive_value_approximator import NaiveValueApproximator
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile
from decision_making.src.planning.behavioral.filtering import action_spec_filter_bank
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering, \
    ActionSpecFilter
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.types import C_X, C_Y, C_YAW, C_V, C_A, C_K
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import DynamicObject, ObjectSize, EgoState, State
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger


# def calc_collision_time(v_init: float, v_max: float, acc: float, v_tar: float, dist: float) -> float:
#
#     v_init_rel = v_init - v_tar
#     v_max_rel = v_max - v_tar
#     if v_max_rel <= 0 and v_init_rel <= 0:
#         return np.inf
#     if v_init_rel < v_max_rel:
#         acceleration_dist = (v_max_rel**2 - v_init_rel**2) / (2*acc)
#         if acceleration_dist < dist:
#             acceleration_time = (v_max_rel - v_init_rel) / acc
#             const_vel_time = (dist - acceleration_dist) / v_max_rel
#             return acceleration_time + const_vel_time
#         else:  # acceleration_dist >= dist; solve for t: v*t + at^2/2 = dist
#             acceleration_time = (np.sqrt(v_init_rel**2 + 2*acc*dist) - v_init_rel) / acc
#             return acceleration_time
#     else:  # v_init_rel > v_max_rel
#         return dist / v_init_rel


def test_evaluate_ranges_F_LF():
    logger = Logger("test_BP_costs")
    road_id = 20
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    #action_space = DynamicActionSpace(logger, predictor)
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    value_approximator = NaiveValueApproximator(logger)
    road_frenet = MapUtils.get_road_rhs_frenet_by_road_id(road_id)

    ego_vel = des_vel - 4
    sec_to_F = 3

    LF_vel_range = np.arange(ego_vel-4, des_vel+4.001, 2)
    sec_to_LF_range = np.arange(2, 8.001, 2)
    F_vel_range = np.arange(ego_vel-4, des_vel+4.001, 2)

    for LF_vel in LF_vel_range:
        for sec_to_LF in sec_to_LF_range:
            for F_vel in F_vel_range:

                ego = MapUtils.create_canonic_ego(0, ego_lon, lane_width / 2, ego_vel, size, road_frenet)
                F_lon = ego_lon + sec_to_F * (ego_vel + des_vel) / 2
                F = MapUtils.create_canonic_object(1, 0, F_lon, lane_width / 2, F_vel, size, road_frenet)

                t1 = abs(des_vel - ego_vel) / 1.
                t2 = max(0., 100 - t1)
                vel_profile = VelocityProfile(v_init=ego_vel, t1=t1, v_mid=des_vel, t2=t2, t3=0, v_tar=des_vel)
                safe_to_F = vel_profile.calc_last_safe_time(ego_lon, size.length / 2, F, np.inf, 2, 2)

                LF_lon = ego_lon + sec_to_LF * (ego_vel + des_vel) / 2
                LF = MapUtils.create_canonic_object(3, 0, LF_lon, 3 * lane_width / 2, LF_vel, size, road_frenet)
                safe_to_LF = vel_profile.calc_last_safe_time(ego_lon, size.length / 2, LF, np.inf, 2, 2)

                objects = [F, LF]
                state = State(None, objects, ego)
                behavioral_state = BehavioralGridState.create_from_state(state, logger)
                recipes = action_space.recipes
                recipes_mask = action_space.filter_recipes(recipes, behavioral_state)

                # Action specification
                specs = np.full(recipes.__len__(), None)
                valid_action_recipes = [recipe for i, recipe in enumerate(recipes) if recipes_mask[i]]
                specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

                specs_mask = action_spec_validator.filter_action_specs(specs, behavioral_state)

                # State-Action Evaluation
                costs = spec_evaluator.evaluate(behavioral_state, recipes, list(specs), specs_mask)
                # approximate cost-to-go per terminal state
                terminal_behavioral_states = SingleStepBehavioralPlanner.generate_terminal_states(
                    state, list(specs), recipes, specs_mask, logger)

                # generate goals for all terminal_behavioral_states
                navigation_goals = SingleStepBehavioralPlanner.generate_goals(
                    ego.road_localization.road_id, ego.road_localization.road_lon, terminal_behavioral_states)

                terminal_states_values = np.array([value_approximator.approximate(state, navigation_goals[i])
                                                   if specs_mask[i] else np.nan
                                                   for i, state in enumerate(terminal_behavioral_states)])

                # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
                action_q_cost = costs + terminal_states_values

                valid_idx = np.where(specs_mask)[0]
                best_idx = valid_idx[action_q_cost[valid_idx].argmin()]
                selected_lane = recipes[best_idx].relative_lane.value

                print('ego_vel=%.2f: i=%d lane=%d\n'
                      'F_vel =%.2f safe_to_F =%.2f dist=%.2f\n'
                      'LF_vel=%.2f safe_to_LF=%.2f dist=%.2f\n' %
                      (ego_vel, best_idx, selected_lane,
                       F_vel, safe_to_F, F_lon-ego_lon,
                       LF_vel, safe_to_LF, LF_lon-ego_lon))

                if (safe_to_F+8) / (safe_to_LF+8) > 1.4 * ((LF_vel+2)/(F_vel+2))**2 or F_vel >= des_vel-1:
                    assert selected_lane == 0
                if (safe_to_F+8) / (safe_to_LF+8) < 0.7 * ((LF_vel+2)/(F_vel+2))**2 and F_vel <= des_vel-3:
                    assert selected_lane == 1


def test_evaluate_rangesForLB():
    logger = Logger("test_BP_costs")
    road_id = 20
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    size = ObjectSize(4, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    #action_space = DynamicActionSpace(logger, predictor)
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    value_approximator = NaiveValueApproximator(logger)
    road_frenet = MapUtils.get_road_rhs_frenet_by_road_id(road_id)

    ego_vel = des_vel - 4
    F_vel = des_vel - 4
    sec_to_F = 3

    LB_vel_range = np.arange(ego_vel-4, des_vel+4.001, 2)
    sec_to_LB_range = np.arange(2, 8.001, 2)

    for LB_vel in LB_vel_range:
        for sec_to_LB in sec_to_LB_range:

            ego = MapUtils.create_canonic_ego(0, ego_lon, lane_width / 2, ego_vel, size, road_frenet)
            F_lon = ego_lon + sec_to_F * (ego_vel + des_vel) / 2
            F = MapUtils.create_canonic_object(1, 0, F_lon, lane_width / 2, F_vel, size, road_frenet)

            t1 = abs(des_vel - ego_vel) / 1.
            t2 = max(0., 100 - t1)
            vel_profile = VelocityProfile(v_init=ego_vel, t1=t1, v_mid=des_vel, t2=t2, t3=0, v_tar=des_vel)
            safe_to_F = vel_profile.calc_last_safe_time(ego_lon, size.length / 2, F, 5, 2, 2)

            LB_lon = ego_lon - LB_vel * sec_to_LB - 6 * (LB_vel - ego_vel)
            LB = MapUtils.create_canonic_object(2, 0, LB_lon, 3 * lane_width / 2, LB_vel, size, road_frenet)
            safe_to_LB = vel_profile.calc_last_safe_time(ego_lon, size.length / 2, LB, np.inf, 2.0, 2.0)

            objects = [F, LB]
            state = State(None, objects, ego)
            behavioral_state = BehavioralGridState.create_from_state(state, logger)
            recipes = action_space.recipes
            recipes_mask = action_space.filter_recipes(recipes, behavioral_state)

            # Action specification
            specs = np.full(recipes.__len__(), None)
            valid_action_recipes = [recipe for i, recipe in enumerate(recipes) if recipes_mask[i]]
            specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

            specs_mask = action_spec_validator.filter_action_specs(specs, behavioral_state)

            # State-Action Evaluation
            costs = spec_evaluator.evaluate(behavioral_state, recipes, list(specs), specs_mask)
            # approximate cost-to-go per terminal state
            terminal_behavioral_states = SingleStepBehavioralPlanner.generate_terminal_states(
                state, list(specs), recipes, specs_mask, logger)

            # generate goals for all terminal_behavioral_states
            navigation_goals = SingleStepBehavioralPlanner.generate_goals(
                ego.road_localization.road_id, ego.road_localization.road_lon, terminal_behavioral_states)

            terminal_states_values = np.array([value_approximator.approximate(state, navigation_goals[i])
                                               if specs_mask[i] else np.nan
                                               for i, state in enumerate(terminal_behavioral_states)])

            # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
            action_q_cost = costs + terminal_states_values

            valid_idx = np.where(specs_mask)[0]
            best_idx = valid_idx[action_q_cost[valid_idx].argmin()]
            selected_lane = recipes[best_idx].relative_lane.value

            print('ego_vel=%.2f: i=%d lane=%d\n'
                  'LB_vel=%.2f safe_to_LB=%.2f dist=%.2f\n' %
                  (ego_vel, best_idx, selected_lane,
                   LB_vel, safe_to_LB, ego_lon-LB_lon))

            if safe_to_LB < 6:
                assert selected_lane == 0
            if safe_to_LB > 8:
                assert selected_lane == 1


def test_evaluate_rangesForF():
    logger = Logger("test_BP_costs")
    road_id = 20
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    size = ObjectSize(4, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    # action_space = DynamicActionSpace(logger, predictor)
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    value_approximator = NaiveValueApproximator(logger)
    road_frenet = MapUtils.get_road_rhs_frenet_by_road_id(road_id)

    ego_vel_range = np.arange(des_vel-9, des_vel+9.1, 3)
    F_vel_range = np.arange(des_vel-9, des_vel+9.1, 3)
    sec_to_F_range = np.array([2, 2.5, 3, 4, 6, 8])

    for ego_vel in ego_vel_range:
        for F_vel in F_vel_range:
            for sec_to_F in sec_to_F_range:
                ego = MapUtils.create_canonic_ego(0, ego_lon, lane_width / 2, ego_vel, size, road_frenet)
                F_lon = ego_lon + sec_to_F * (ego_vel + des_vel) / 2
                F = MapUtils.create_canonic_object(1, 0, F_lon, lane_width / 2, F_vel, size, road_frenet)

                t1 = abs(des_vel - ego_vel) / 1.
                t2 = max(0., 100 - t1)
                vel_profile = VelocityProfile(v_init=ego_vel, t1=t1, v_mid=des_vel, t2=t2, t3=0, v_tar=des_vel)
                safe_to_F = vel_profile.calc_last_safe_time(ego_lon, size.length/2, F, np.inf, 1.6, 1.6)
                if safe_to_F < 0:  # unsafe state
                    continue

                state = State(None, [F], ego)
                behavioral_state = BehavioralGridState.create_from_state(state, logger)
                recipes = action_space.recipes
                recipes_mask = action_space.filter_recipes(recipes, behavioral_state)

                # Action specification
                specs = np.full(recipes.__len__(), None)
                valid_action_recipes = [recipe for i, recipe in enumerate(recipes) if recipes_mask[i]]
                specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

                specs_mask = action_spec_validator.filter_action_specs(specs, behavioral_state)

                # State-Action Evaluation
                costs = spec_evaluator.evaluate(behavioral_state, recipes, list(specs), specs_mask)

                # approximate cost-to-go per terminal state
                terminal_behavioral_states = SingleStepBehavioralPlanner.generate_terminal_states(
                    state, list(specs), recipes, specs_mask, logger)

                # generate goals for all terminal_behavioral_states
                navigation_goals = SingleStepBehavioralPlanner.generate_goals(
                    ego.road_localization.road_id, ego.road_localization.road_lon, terminal_behavioral_states)

                terminal_states_values = np.array([value_approximator.approximate(state, navigation_goals[i])
                                                   if specs_mask[i] else np.nan
                                                   for i, state in enumerate(terminal_behavioral_states)])

                # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
                action_q_cost = costs + terminal_states_values

                valid_idx = np.where(specs_mask)[0]
                best_idx = valid_idx[action_q_cost[valid_idx].argmin()]
                selected_lane = recipes[best_idx].relative_lane.value

                print('ego_vel=%.2f F_vel=%.2f sec=%.2f dist=%.2f t_c=%.2f: cost=%.2f val=%.2f; i=%d lane=%d\n' %
                      (ego_vel, F_vel, sec_to_F, F_lon-ego_lon, safe_to_F, costs[best_idx], terminal_states_values[best_idx],
                       best_idx, selected_lane))

                if ego_vel>=20 and F_vel>=8 and sec_to_F>=2.5:
                    continue

                if safe_to_F > 9 or F_vel >= des_vel-1:
                    assert selected_lane == 0
                if 3. < safe_to_F < 6 and F_vel <= des_vel-3:
                    assert selected_lane == 1


def test_evaluate_ranges_FtoRight():
    logger = Logger("test_BP_costs")
    road_id = 20
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    size = ObjectSize(4, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    #action_space = DynamicActionSpace(logger, predictor)
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    value_approximator = NaiveValueApproximator(logger)
    road_frenet = MapUtils.get_road_rhs_frenet_by_road_id(road_id)

    ego_vel = des_vel - 4

    sec_to_RF_range = np.arange(2, 8.001, 2)
    RF_vel_range = np.arange(ego_vel-4, des_vel+4.001, 2)

    for sec_to_RF in sec_to_RF_range:
        for RF_vel in RF_vel_range:

            ego = MapUtils.create_canonic_ego(0, ego_lon, 3 * lane_width / 2, ego_vel, size, road_frenet)
            RF_lon = ego_lon + sec_to_RF * ego_vel
            RF = MapUtils.create_canonic_object(1, 0, RF_lon, lane_width / 2, RF_vel, size, road_frenet)

            objects = [RF]
            state = State(None, objects, ego)
            behavioral_state = BehavioralGridState.create_from_state(state, logger)
            recipes = action_space.recipes
            recipes_mask = action_space.filter_recipes(recipes, behavioral_state)

            # Action specification
            specs = np.full(recipes.__len__(), None)
            valid_action_recipes = [recipe for i, recipe in enumerate(recipes) if recipes_mask[i]]
            specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

            specs_mask = action_spec_validator.filter_action_specs(specs, behavioral_state)

            # State-Action Evaluation
            costs = spec_evaluator.evaluate(behavioral_state, recipes, list(specs), specs_mask)
            # approximate cost-to-go per terminal state
            terminal_behavioral_states = SingleStepBehavioralPlanner.generate_terminal_states(
                state, list(specs), recipes, specs_mask, logger)

            # generate goals for all terminal_behavioral_states
            navigation_goals = SingleStepBehavioralPlanner.generate_goals(
                ego.road_localization.road_id, ego.road_localization.road_lon, terminal_behavioral_states)

            terminal_states_values = np.array([value_approximator.approximate(state, navigation_goals[i])
                                               if specs_mask[i] else np.nan
                                               for i, state in enumerate(terminal_behavioral_states)])

            # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
            action_q_cost = costs + terminal_states_values

            valid_idx = np.where(specs_mask)[0]
            best_idx = valid_idx[action_q_cost[valid_idx].argmin()]
            rel_lane = recipes[best_idx].relative_lane.value

            print('ego_vel=%.2f: i=%d rel_lane=%d\n'
                  'RF_vel =%.2f sec_to_RF =%.2f dist=%.2f\n' %
                  (ego_vel, best_idx, rel_lane,
                   RF_vel, sec_to_RF, RF_lon-ego_lon))

            if sec_to_RF > 9 or (RF_vel >= des_vel and sec_to_RF > 2.5):
                assert rel_lane == -1
            if (sec_to_RF < 6 and RF_vel < des_vel - 2) or sec_to_RF <= AV_TIME_DELAY:
                assert rel_lane == 0


def test_speedProfiling():
    logger = Logger("test_behavioralScenarios")
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 250
    road_id = 20
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    lat0 = lane_width / 2
    lat1 = lat0 + lane_width
    lat2 = lat0 + 2*lane_width
    road_frenet = MapUtils.get_road_rhs_frenet_by_road_id(road_id)
    size = ObjectSize(4, 2, 1)
    ego_vel = 12
    F_vel = ego_vel
    LB_vel = des_vel

    evaluator = HeuristicActionSpecEvaluator(logger)

    ego = MapUtils.create_canonic_ego(0, ego_lon, lat1, ego_vel, size, road_frenet)

    dist_from_F_in_sec = 4.0
    dist_from_LB_in_sec  = 4.0

    LB_lon = ego_lon - des_vel * dist_from_LB_in_sec
    F_lon = ego_lon + dist_from_F_in_sec * ego_vel

    F = MapUtils.create_canonic_object(1, 0, F_lon, lat0, F_vel, size, road_frenet)
    LF = MapUtils.create_canonic_object(2, 0, F_lon, lat2, des_vel, size, road_frenet)
    LB = MapUtils.create_canonic_object(4, 0, LB_lon, lat2, LB_vel, size, road_frenet)
    RF = MapUtils.create_canonic_object(5, 0, F_lon, lat0, des_vel, size, road_frenet)
    RB = MapUtils.create_canonic_object(7, 0, LB_lon, lat0, LB_vel, size, road_frenet)
    B = MapUtils.create_canonic_object(8, 0, LB_lon, lat1, F_vel, size, road_frenet)

    objects = [F, LF, LB, RF, RB, B]

    state = State(None, objects, ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    action_recipes = action_space.recipes

    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)

    action_specs = [action_space.specify_goal(recipe, behavioral_state) if recipes_mask[i] else None
                    for i, recipe in enumerate(action_recipes)]

    # ActionSpec filtering
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    action_specs_mask = action_spec_validator.filter_action_specs(action_specs, behavioral_state)

    start = time.time()

    costs = evaluator.evaluate(behavioral_state, action_recipes, action_specs, action_specs_mask)

    end = time.time()

    print('action num=%d; filtered actions=%d; time = %f' % (len(action_recipes), np.count_nonzero(recipes_mask), end - start))


import matplotlib.pyplot as plt
@pytest.mark.skip(reason="not test, just building a table of parameteres")
def test_behavioralScenarios_moveToLeft_findThresholds():
    """
    Test lane change to the left if the velocity of the car F is des_vel-2.
    Test different distances from F, different initial velocities of ego, empty/occupied left lane, longitude of LB.
    Desired result: for all initial velocities, don't overtake when dist=4 sec and overtake when dist=3 sec.
    :return:
    """
    logger = Logger("test_behavioralScenarios")
    road_id = 20
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 200.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    evaluator = HeuristicActionRecipeEvaluator(logger)

    dv = 1.
    ego_vel_range = np.arange(des_vel - des_vel/2., des_vel + des_vel/2. + 0.1, dv)
    F_vel_range = np.arange(des_vel - des_vel/2., des_vel + des_vel/2. + 0.1, dv)
    dist_range = np.arange(7.0, 0.9, -0.2)

    F_lon_tab = np.zeros((len(ego_vel_range), len(F_vel_range)))

    for ego_vel in ego_vel_range:
        for F_vel in F_vel_range:
            for dist_from_F_in_sec in dist_range:

                ego_cpoint, ego_yaw = MapService.get_instance().convert_road_rhs_to_global_coordinates(
                    road_id, ego_lon, road_mid_lat - lane_width)
                ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

                F_lon = ego_lon + dist_from_F_in_sec * (ego_vel + des_vel)/2 + (des_vel**2 - F_vel**2)/16  # 3*(des_vel - F_vel)**2
                F_cpoint, F_yaw = MapService.get_instance().convert_road_rhs_to_global_coordinates(
                    road_id, F_lon, road_mid_lat - lane_width)
                F = DynamicObject(1, 0, F_cpoint[0], F_cpoint[1], F_cpoint[2], F_yaw, size, 0, F_vel, 0, 0, 0)

                objects = [F]
                state = State(None, objects, ego)
                behavioral_state = BehavioralGridState.create_from_state(state, logger)

                actions = []
                # create actions based on the occupancy grid
                if (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
                    right_action = DynamicActionRecipe(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT,
                                                       ActionType.FOLLOW_VEHICLE, AggressivenessLevel.CALM)
                    actions.append(right_action)
                right_action_stat = StaticActionRecipe(RelativeLane.SAME_LANE, des_vel, AggressivenessLevel.CALM)
                actions.append(right_action_stat)
                left_action = StaticActionRecipe(RelativeLane.LEFT_LANE, des_vel, AggressivenessLevel.CALM)
                actions.append(left_action)

                costs = evaluator.evaluate(behavioral_state, actions, [True] * len(actions))

                best_idx = np.argmin(costs)

                if actions[best_idx].relative_lane == RelativeLane.LEFT_LANE:
                    ego_i = int((ego_vel - ego_vel_range[0]) / dv + 0.5)
                    F_i = int((F_vel - F_vel_range[0]) / dv + 0.5)
                    F_lon_tab[ego_i, F_i] = dist_from_F_in_sec
                    break

    np.savetxt('test.txt', F_lon_tab, fmt='%1.1f ')

    #plt.imshow(F_lon_tab)
    #plt.show()
    #time.sleep(10)
