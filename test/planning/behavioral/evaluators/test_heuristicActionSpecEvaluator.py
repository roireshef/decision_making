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

from decision_making.src.global_constants import BP_DEFAULT_DESIRED_SPEED, \
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, MINIMAL_STATIC_ACTION_TIME, SAFETY_MARGIN_TIME_DELAY, LON_ACC_LIMITS, \
    VELOCITY_LIMITS, SPECIFICATION_MARGIN_TIME_DELAY
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
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import CostBasedBehavioralPlanner
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.types import C_X, C_Y, C_YAW, C_V, C_A, C_K
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import DynamicObject, ObjectSize, EgoState, State
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger

"""
Definitions for the tests:
    TC(F) is "time_to_collision", i.e. the latest time for which ego is safe w.r.t. F.
    V(F) is the constant velocity of F.
"""

def test_evaluate_differentDistancesAndVeloctiesOfF_laneChangeAccordingToTheLogic():
    """
    Test the criterion for lane change for different velocities of ego and of F and distances from F:

    If one of the following conditions holds:
        TC(F) > 9                         F is far
        V_DESIRED – V(F) < 1              F is fast
    then stay on the same lane

    If all following conditions hold:
       3 < TC(F) < 6                     F is not far and not too close
       V_DESIRED – V(F) > 3              F is slow
       V(ego) >= V(F)                    ego is faster than F
    then overtake F to the left
    """
    logger = Logger("test_BP_costs")
    road_id = 20
    v_des = BP_DEFAULT_DESIRED_SPEED
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    length = 4
    size = ObjectSize(length, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    value_approximator = NaiveValueApproximator(logger)
    planner = SingleStepBehavioralPlanner(action_space, None, spec_evaluator,
                                          action_spec_validator, value_approximator, predictor, logger)

    road_frenet = get_road_rhs_frenet_by_road_id(road_id)

    ego_vel_range = np.arange(v_des-9, min(VELOCITY_LIMITS[1], v_des+9.1), 3)
    F_vel_range = np.arange(v_des-9, v_des+9.1, 3)
    TC_F_range = np.arange(2, 8.001)

    for ego_vel in ego_vel_range:
        for F_vel in F_vel_range:
            for TC_F in TC_F_range:
                ego = create_canonic_ego(0, ego_lon, lane_width / 2, ego_vel, size, road_frenet)
                F_lon = ego_lon + calc_init_dist_by_safe_time(True, ego_vel, F_vel, TC_F, SAFETY_MARGIN_TIME_DELAY, length)
                if np.isinf(F_lon):
                    continue
                F = create_canonic_object(1, 0, F_lon, lane_width / 2, F_vel, size, road_frenet)

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
                terminal_behavioral_states = planner._generate_terminal_states(state, list(specs), specs_mask)

                # generate goals for all terminal_behavioral_states
                navigation_goals = CostBasedBehavioralPlanner._generate_goals(
                    ego.road_localization.road_id, ego.road_localization.road_lon, terminal_behavioral_states)

                terminal_states_values = np.array([value_approximator.approximate(state, navigation_goals[i])
                                                   if specs_mask[i] else np.nan
                                                   for i, state in enumerate(terminal_behavioral_states)])

                # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
                action_q_cost = costs + terminal_states_values

                valid_idx = np.where(specs_mask)[0]
                best_idx = valid_idx[action_q_cost[valid_idx].argmin()]
                selected_lane = recipes[best_idx].relative_lane.value

                print('ego_vel=%.2f F_vel=%.2f TC_F=%.2f dist=%.2f cost=%.2f val=%.2f; i=%d lane=%d\n' %
                      (ego_vel, F_vel, TC_F, F_lon-ego_lon, costs[best_idx], terminal_states_values[best_idx],
                       best_idx, selected_lane))

                if TC_F > 9 or F_vel >= v_des-1:
                    assert selected_lane == 0
                if 3 < TC_F < 6 and F_vel <= v_des-3 and ego_vel >= F_vel:
                    assert selected_lane == 1


def test_evaluate_differentDistancesAndVeloctiesOfFandLF_laneChangeAccordingToTheLogic():
    """
    Test criterion for the lane change for different velocities and distances from F and LF.

    If one of the following conditions holds:

        TC(LF) < 0                              LF is not safe
        TC(LF) – TC(F) < 0                      LF is not further
        TC(F) > 9                               F is far
        V_DESIRED - V(F) < 1                    F is not slow
    then stay on the same lane

    If all following conditions hold:
        TC(LF) >= 0                             LF is safe
        TC(LF) – TC(F) > 2                      LF is further than F
        3 < TC(F) < 6                           F is not far and not too close
        V_DESIRED - V(F) >= 3                   F is slow
    then overtake F to the left
    """
    logger = Logger("test_BP_costs")
    road_id = 20
    v_des = BP_DEFAULT_DESIRED_SPEED
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    length = 4
    size = ObjectSize(length, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    value_approximator = NaiveValueApproximator(logger)
    road_frenet = get_road_rhs_frenet_by_road_id(road_id)
    planner = SingleStepBehavioralPlanner(action_space, None, spec_evaluator,
                                          action_spec_validator, value_approximator, predictor, logger)

    ego_vel = v_des - 4
    TC_F = 5

    LF_vel_range = np.arange(ego_vel-4, v_des+4.001, 2)
    TC_LF_range = np.arange(-2, 8.001, 2)
    F_vel_range = np.arange(ego_vel-4, v_des+4.001, 2)

    for LF_vel in LF_vel_range:
        for TC_LF in TC_LF_range:
            for F_vel in F_vel_range:

                ego = create_canonic_ego(0, ego_lon, lane_width / 2, ego_vel, size, road_frenet)
                F_lon = ego_lon + calc_init_dist_by_safe_time(True, ego_vel, F_vel, TC_F, SAFETY_MARGIN_TIME_DELAY, length)
                if np.isinf(F_lon):
                    continue
                F = create_canonic_object(1, 0, F_lon, lane_width / 2, F_vel, size, road_frenet)

                LF_lon = ego_lon + calc_init_dist_by_safe_time(True, ego_vel, F_vel, TC_LF, SAFETY_MARGIN_TIME_DELAY, length)
                if np.isinf(LF_lon):
                    continue
                LF = create_canonic_object(3, 0, LF_lon, 3 * lane_width / 2, LF_vel, size, road_frenet)

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
                terminal_behavioral_states = planner._generate_terminal_states(state, list(specs), specs_mask)

                # generate goals for all terminal_behavioral_states
                navigation_goals = SingleStepBehavioralPlanner._generate_goals(
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
                      'F_vel =%.2f TC_F =%.2f dist=%.2f\n'
                      'LF_vel=%.2f TC_LF=%.2f dist=%.2f\n' %
                      (ego_vel, best_idx, selected_lane,
                       F_vel, TC_F, F_lon-ego_lon,
                       LF_vel, TC_LF, LF_lon-ego_lon))

                if TC_LF < 0 or (TC_LF - TC_F) + (LF_vel - F_vel) < -3 or TC_F > 9 or F_vel >= v_des-1:
                    assert selected_lane == 0
                if TC_LF >= 0 and (TC_LF - TC_F) + (LF_vel - F_vel) > 1 and TC_F < 6 and F_vel <= min(v_des, LF_vel) - 3:
                    assert selected_lane == 1


def test_evaluate_differentStatesLBandF_laneChangeAccordingToTheLogic():
    """
    Test the criterion for lane change for different distances from LB and different velocities of F.

    If one of the following conditions holds:
        TC(LB) < 3                      LB is close
        F_vel >= v_des-1                F is not slow
    then stay on the same lane

    If all following conditions hold:
        TC(LB) > 5                      LB is far
        F_vel <= v_des-3                F is slow
    then overtake F to the left
    """
    logger = Logger("test_BP_costs")
    road_id = 20
    v_des = BP_DEFAULT_DESIRED_SPEED
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    length = 4
    size = ObjectSize(length, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    value_approximator = NaiveValueApproximator(logger)
    road_frenet = get_road_rhs_frenet_by_road_id(road_id)
    planner = SingleStepBehavioralPlanner(action_space, None, spec_evaluator,
                                          action_spec_validator, value_approximator, predictor, logger)

    ego_vel = v_des - 4
    TC_F = 6
    LB_vel = 20

    F_vel_range = np.arange(v_des-8, v_des+0.001, 2)
    TC_LB_range = np.arange(-2, 8.001, 2)

    for F_vel in F_vel_range:
        for TC_LB in TC_LB_range:

            ego = create_canonic_ego(0, ego_lon, lane_width / 2, ego_vel, size, road_frenet)
            F_lon = ego_lon + calc_init_dist_by_safe_time(True, ego_vel, F_vel, TC_F, SAFETY_MARGIN_TIME_DELAY, length)
            if np.isinf(F_lon):
                continue
            F = create_canonic_object(1, 0, F_lon, lane_width / 2, F_vel, size, road_frenet)

            LB_lon = ego_lon - calc_init_dist_by_safe_time(False, ego_vel, LB_vel, TC_LB, SPECIFICATION_MARGIN_TIME_DELAY, length)
            if np.isinf(LB_lon):
                continue
            LB = create_canonic_object(2, 0, LB_lon, 3 * lane_width / 2, LB_vel, size, road_frenet)

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
            terminal_behavioral_states = planner._generate_terminal_states(state, list(specs), specs_mask)

            # generate goals for all terminal_behavioral_states
            navigation_goals = CostBasedBehavioralPlanner._generate_goals(
                ego.road_localization.road_id, ego.road_localization.road_lon, terminal_behavioral_states)

            terminal_states_values = np.array([value_approximator.approximate(state, navigation_goals[i])
                                               if specs_mask[i] else np.nan
                                               for i, state in enumerate(terminal_behavioral_states)])

            # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
            action_q_cost = costs + terminal_states_values

            valid_idx = np.where(specs_mask)[0]
            best_idx = valid_idx[action_q_cost[valid_idx].argmin()]
            selected_lane = recipes[best_idx].relative_lane.value

            print('ego_vel=%.2f, F_vel=%.2f; i=%d lane=%d\n'
                  'TC_LB=%.2f dist=%.2f\n' %
                  (ego_vel, F_vel, best_idx, selected_lane,
                   TC_LB, ego_lon-LB_lon))

            if F_vel >= v_des-1 or TC_LB < 3:
                assert selected_lane == 0
            if F_vel <= v_des-3 and ego_vel >= F_vel and TC_LB > 5:
                assert selected_lane == 1


def test_evaluate_differentVeloctiesF_tradeOffBetweenLatJerkAndEfficiency():
    """
    Test the trade off between lateral jerk and efficiency for different velocities of F.
    The lateral jerk is caused both by F and by LB.
    When F is too slow (6 m/s), the agent overtakes F even on expense of a tangible lateral jerk.
    When F is not so slow (9 m/s), the agent prefers to follow F.
    """
    logger = Logger("test_BP_costs")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    length = 4
    size = ObjectSize(length, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    value_approximator = NaiveValueApproximator(logger)
    road_frenet = get_road_rhs_frenet_by_road_id(road_id)
    planner = SingleStepBehavioralPlanner(action_space, None, spec_evaluator,
                                          action_spec_validator, value_approximator, predictor, logger)

    ego_vel = 10
    F_vel_list = [6, 9]
    F_dist_list = [45, 33]
    LB_vel = 25
    LB_dist = 160

    for i, F_vel in enumerate(F_vel_list):
        F_dist = F_dist_list[i]

        ego = create_canonic_ego(0, ego_lon, lane_width / 2, ego_vel, size, road_frenet)
        F_lon = ego_lon + F_dist
        F = create_canonic_object(1, 0, F_lon, lane_width / 2, F_vel, size, road_frenet)

        LB_lon = ego_lon - LB_dist
        LB = create_canonic_object(2, 0, LB_lon, 3 * lane_width / 2, LB_vel, size, road_frenet)

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
        terminal_behavioral_states = planner._generate_terminal_states(state, list(specs), specs_mask)

        # generate goals for all terminal_behavioral_states
        navigation_goals = CostBasedBehavioralPlanner._generate_goals(
            ego.road_localization.road_id, ego.road_localization.road_lon, terminal_behavioral_states)

        terminal_states_values = np.array([value_approximator.approximate(state, navigation_goals[i])
                                           if specs_mask[i] else np.nan
                                           for i, state in enumerate(terminal_behavioral_states)])

        # compute "approximated Q-value" (action cost +  cost-to-go) for all actions
        action_q_cost = costs + terminal_states_values

        valid_idx = np.where(specs_mask)[0]
        best_idx = valid_idx[action_q_cost[valid_idx].argmin()]
        selected_lane = recipes[best_idx].relative_lane.value

        print('ego_vel=%.2f, F_vel=%.2f; i=%d lane=%d\n'
              'LB_vel=%.2f LB_dist=%.2f dist=%.2f\n' %
              (ego_vel, F_vel, best_idx, selected_lane,
               LB_vel, LB_dist, ego_lon-LB_lon))

        if F_vel > 7:
            assert selected_lane == 0
        else:
            assert selected_lane == 1


def test_evaluate_differentDistancesAndVeloctiesOfFandRF_laneChangeAccordingToTheLogic():
    """
    Test the criterion for lane change to the right for different velocities and distances from RF.

    If one of the following conditions holds:
        V_DESIRED - V(RF) > 2 and TC(RF) < 6         RF is slow and not far
        TC(RF) < 2                                   RF is not safe longitudinally
    then stay on the same lane

    If all following conditions hold:
        TC(RF) > 9                                   RF is far
        V_DESIRED < V(RF) and TC(RF) > 2.5           RF is not slow and safe longitudinally
    then overtake F to the left
    """
    logger = Logger("test_BP_costs")
    road_id = 20
    v_des = BP_DEFAULT_DESIRED_SPEED
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    size = ObjectSize(4, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    value_approximator = NaiveValueApproximator(logger)
    road_frenet = get_road_rhs_frenet_by_road_id(road_id)
    planner = SingleStepBehavioralPlanner(action_space, None, spec_evaluator,
                                          action_spec_validator, value_approximator, predictor, logger)

    ego_vel = v_des - 4

    sec_to_RF_range = np.arange(2, 8.001, 2)
    RF_vel_range = np.arange(ego_vel-4, v_des+4.001, 2)

    for sec_to_RF in sec_to_RF_range:
        for RF_vel in RF_vel_range:

            ego = create_canonic_ego(0, ego_lon, 3 * lane_width / 2, ego_vel, size, road_frenet)
            RF_lon = ego_lon + sec_to_RF * ego_vel
            RF = create_canonic_object(1, 0, RF_lon, lane_width / 2, RF_vel, size, road_frenet)

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
            terminal_behavioral_states = planner._generate_terminal_states(state, list(specs), specs_mask)

            # generate goals for all terminal_behavioral_states
            navigation_goals = SingleStepBehavioralPlanner._generate_goals(
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

            if sec_to_RF > 9 or (RF_vel >= v_des and sec_to_RF > 2.5):
                assert rel_lane == -1
            if (sec_to_RF < 6 and RF_vel < v_des - 2) or sec_to_RF <= SAFETY_MARGIN_TIME_DELAY:
                assert rel_lane == 0


def test_speedProfiling():
    logger = Logger("test_behavioralScenarios")
    v_des = BP_DEFAULT_DESIRED_SPEED
    ego_lon = 250
    road_id = 20
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    lat0 = lane_width / 2
    lat1 = lat0 + lane_width
    lat2 = lat0 + 2*lane_width
    road_frenet = get_road_rhs_frenet_by_road_id(road_id)
    size = ObjectSize(4, 2, 1)
    ego_vel = 12
    F_vel = ego_vel
    LB_vel = v_des

    evaluator = HeuristicActionSpecEvaluator(logger)

    ego = create_canonic_ego(0, ego_lon, lat1, ego_vel, size, road_frenet)

    dist_from_F_in_sec = 4.0
    dist_from_LB_in_sec  = 4.0

    LB_lon = ego_lon - v_des * dist_from_LB_in_sec
    F_lon = ego_lon + dist_from_F_in_sec * ego_vel

    F = create_canonic_object(1, 0, F_lon, lat0, F_vel, size, road_frenet)
    LF = create_canonic_object(2, 0, F_lon, lat2, v_des, size, road_frenet)
    LB = create_canonic_object(4, 0, LB_lon, lat2, LB_vel, size, road_frenet)
    RF = create_canonic_object(5, 0, F_lon, lat0, v_des, size, road_frenet)
    RB = create_canonic_object(7, 0, LB_lon, lat0, LB_vel, size, road_frenet)
    B = create_canonic_object(8, 0, LB_lon, lat1, F_vel, size, road_frenet)

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


def calc_init_dist_by_safe_time(obj_ahead: bool, ego_v: float, obj_v: float, TC: float, time_delay: float,
                                margin: float, max_brake: float=-LON_ACC_LIMITS[0]):
    """
    The approximated inverse function of VelocityProfile.calc_last_safe_time(), is used by the above BP cost tests.
    Given "time to collision" from an object, calculate the initial distance from the object.
    :param obj_ahead: [bool] True if the object is ahead of ego, False if the object is behind ego
    :param ego_v: [m/s] ego initial velocity
    :param obj_v: [m/s] object's constant velocity
    :param TC: [sec] "time to collision" (the result of VelProfile.calc_last_safe_time)
    :param time_delay: [sec] reaction delay of the back object (ego or obj)
    :param margin: [m] cars' lengths margin (half sum of the two cars lengths)
    :param max_brake: [m/s^2] maximal deceleration of the objects
    :return: initial distance from the object to obtain the required "time to collision"
    """
    des_v = BP_DEFAULT_DESIRED_SPEED
    t1 = abs(des_v - ego_v) / 0.5
    t2 = max(0., 100 - t1)
    vel_profile = VelocityProfile(v_init=ego_v, t_first=t1, v_mid=des_v, t_flat=t2, t_last=0, v_tar=des_v)

    if obj_ahead:
        if TC < 0:
            return 0
        if TC >= 0 and obj_v >= des_v:
            return np.inf
        ego_s_at_td, ego_v_at_td = vel_profile.sample_at(TC + time_delay)
        dist = max(0., ego_v_at_td**2 - obj_v**2) / (2*max_brake) + ego_s_at_td - obj_v * TC + margin
    else:
        if TC < 0:
            return 0
        if TC >= 0 and obj_v <= min(ego_v, des_v):
            return np.inf
        ego_s_at_t, ego_v_at_t = vel_profile.sample_at(TC)
        dist = max(0., obj_v ** 2 - ego_v_at_t ** 2) / (2 * max_brake) + obj_v * (TC + time_delay) - ego_s_at_t + margin

    return dist


def create_canonic_ego(timestamp: int, lon: float, lat: float, vel: float, size: ObjectSize,
                       road_frenet: FrenetSerret2DFrame) -> EgoState:
    """
    Create ego with zero lateral velocity and zero accelerations
    """
    fstate = np.array([lon, vel, 0, lat, 0, 0])
    cstate = road_frenet.fstate_to_cstate(fstate)
    return EgoState(0, timestamp, cstate[C_X], cstate[C_Y], 0, cstate[C_YAW], size, 0, cstate[C_V], 0,
                    cstate[C_A], cstate[C_K])


def create_canonic_object(obj_id: int, timestamp: int, lon: float, lat: float, vel: float, size: ObjectSize,
                          road_frenet: FrenetSerret2DFrame) -> DynamicObject:
    """
    Create object with zero lateral velocity and zero accelerations
    """
    fstate = np.array([lon, vel, 0, lat, 0, 0])
    cstate = road_frenet.fstate_to_cstate(fstate)
    return DynamicObject(obj_id, timestamp, cstate[C_X], cstate[C_Y], 0, cstate[C_YAW], size, 0, cstate[C_V], 0,
                         cstate[C_A], cstate[C_K])


def get_road_rhs_frenet_by_road_id(road_id: int):
    return MapService.get_instance()._rhs_roads_frenet[road_id]
