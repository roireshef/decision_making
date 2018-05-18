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
    BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, BP_MAX_VELOCITY_TOLERANCE
from decision_making.src.planning.behavioral.evaluators.heuristic_action_spec_evaluator import \
    HeuristicActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.naive_value_approximator import NaiveValueApproximator
from decision_making.src.planning.behavioral.filtering import action_spec_filter_bank
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering, \
    ActionSpecFilter
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import approximate_value_function
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import DynamicObject, ObjectSize, EgoState, State
from mapping.src.service.map_service import MapService
from rte.python.logger.AV_logger import AV_Logger


def calc_collision_time(v_init: float, v_max: float, acc: float, v_tar: float, dist: float) -> float:

    v_init_rel = v_init - v_tar
    v_max_rel = v_max - v_tar
    if v_max_rel <= 0 and v_init_rel <= 0:
        return np.inf
    if v_init_rel < v_max_rel:
        acceleration_dist = (v_max_rel**2 - v_init_rel**2) / (2*acc)
        if acceleration_dist < dist:
            acceleration_time = (v_max_rel - v_init_rel) / acc
            const_vel_time = (dist - acceleration_dist) / v_max_rel
            return acceleration_time + const_vel_time
        else:  # acceleration_dist >= dist; solve for t: v*t + at^2/2 = dist
            acceleration_time = (np.sqrt(v_init_rel**2 + 2*acc*dist) - v_init_rel) / acc
            return acceleration_time
    else:  # v_init_rel > v_max_rel
        return dist / v_init_rel


def test_evaluate():
    logger = Logger("test_BP_costs")
    road_id = 20
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger), DynamicActionSpace(logger, predictor)])
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering(action_spec_filter_bank.action_spec_filters)
    value_approximator = NaiveValueApproximator(logger)

    ego_vel = 10
    dist_from_F_in_sec = 3
    F_vel = 10
    dist_from_LB_in_sec = 3
    LB_vel = 10

    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(
        road_id, ego_lon, road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    F_lon = ego_lon + dist_from_F_in_sec * (ego_vel + des_vel) / 2
    F_cpoint, F_yaw = MapService.get_instance().convert_road_to_global_coordinates(
        road_id, F_lon, road_mid_lat - lane_width)
    F = DynamicObject(1, 0, F_cpoint[0], F_cpoint[1], F_cpoint[2], F_yaw, size, 0, F_vel, 0, 0, 0)

    LB_lon = ego_lon - (LB_vel - ego_vel) * 4 - LB_vel * dist_from_LB_in_sec
    LB_cpoint, LB_yaw = MapService.get_instance().convert_road_to_global_coordinates(
        road_id, LB_lon, road_mid_lat)
    LB = DynamicObject(2, 0, LB_cpoint[0], LB_cpoint[1], LB_cpoint[2], LB_yaw, size, 0, LB_vel, 0, 0, 0)

    LF_cpoint, LF_yaw = MapService.get_instance().convert_road_to_global_coordinates(
        road_id, F_lon, road_mid_lat)
    LF = DynamicObject(2, 0, LF_cpoint[0], LF_cpoint[1], LF_cpoint[2], LF_yaw, size, 0, des_vel, 0, 0, 0)
    objects = [F, LF, LB]
    state = State(None, objects, ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)

    # Action specification
    action_specs = np.full(action_recipes.__len__(), None)
    valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_recipes) if recipes_mask[i]]
    action_specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

    action_specs_mask = action_spec_validator.filter_action_specs(action_specs, behavioral_state)

    # State-Action Evaluation
    action_costs = spec_evaluator.evaluate(behavioral_state, action_recipes, list(action_specs), action_specs_mask)

    next_state_values = np.array([approximate_value_function(state, action_recipes[i], spec, value_approximator)
                                  if action_specs_mask[i] else np.inf for i, spec in enumerate(action_specs)])

    # Q-values evaluation (action_cost + value_function(next_state))
    Q_values = np.array([action_costs[i] + next_state_values[i] for i, spec in enumerate(action_specs)])

    valid_idx = np.where(action_specs_mask)[0]
    selected_action_index = valid_idx[Q_values[valid_idx].argmin()]
    logger.debug('Selected recipe: ', action_recipes[selected_action_index].__dict__)
    selected_action_spec = action_specs[selected_action_index]


def test_behavioralScenarios_moveToLeft_differentDistFrom_F_and_initVel():
    """
    Test lane change to the left if the velocity of the car F is des_vel-2.
    Test different distances from F, different initial velocities of ego, empty/occupied left lane, longitude of LB.
    Desired result: for all initial velocities, don't overtake when dist=4 sec and overtake when dist=3 sec.
    :return:
    """
    logger = Logger("test_behavioralScenarios")
    road_id = 20
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    evaluator = HeuristicActionSpecEvaluator(logger)
    predictor = RoadFollowingPredictor(logger)  # TODO: adapt to new changes

    static_action_space = StaticActionSpace(logger)
    dynamic_action_space = DynamicActionSpace(logger, predictor)

    for ego_vel in np.arange(des_vel - des_vel/4., des_vel + des_vel/4.+0.1, des_vel/7.):
        #for F_vel in [des_vel - des_vel/5., des_vel - des_vel/7., des_vel - des_vel/14., des_vel, des_vel + des_vel/7.]:
        for F_vel in [des_vel - des_vel/7., des_vel, des_vel + des_vel/3.]:
            for dist_from_F_in_sec in [8., 6., 4., 2.5]:
                for left_action_type in [ActionType.FOLLOW_LANE, ActionType.FOLLOW_VEHICLE]:
                    for LB_vel in np.arange(des_vel - 2, des_vel + 6.1, 2):
                    #for LB_vel in [des_vel]:
                        for dist_from_LB_in_sec in [8., 6., 4.5, 3.5, 1.5, 0.5]:
                        #for dist_from_LB_in_sec in [8.]:

                            ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(
                                road_id, ego_lon, road_mid_lat - lane_width)
                            ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

                            F_lon = ego_lon + dist_from_F_in_sec * (ego_vel + des_vel)/2
                            F_cpoint, F_yaw = MapService.get_instance().convert_road_to_global_coordinates(
                                road_id, F_lon, road_mid_lat - lane_width)
                            F = DynamicObject(1, 0, F_cpoint[0], F_cpoint[1], F_cpoint[2], F_yaw, size, 0, F_vel, 0, 0, 0)

                            LB_lon = ego_lon - (LB_vel - ego_vel) * 4 - LB_vel * dist_from_LB_in_sec
                            LB_cpoint, LB_yaw = MapService.get_instance().convert_road_to_global_coordinates(
                                road_id, LB_lon, road_mid_lat)
                            LB = DynamicObject(2, 0, LB_cpoint[0], LB_cpoint[1], LB_cpoint[2], LB_yaw, size, 0, LB_vel, 0, 0, 0)
                            objects = [F, LB]

                            if left_action_type == ActionType.FOLLOW_VEHICLE:
                                LF_cpoint, LF_yaw = MapService.get_instance().convert_road_to_global_coordinates(
                                    road_id, F_lon, road_mid_lat)
                                LF = DynamicObject(2, 0, LF_cpoint[0], LF_cpoint[1], LF_cpoint[2], LF_yaw, size, 0, des_vel, 0, 0, 0)
                                objects.append(LF)
                            print('F_vel=%.2f dist_from_F=%.2f  LB_vel=%.2f dist_from_LB=%.2fs %.2fm' %
                                  (F_vel, dist_from_F_in_sec, LB_vel, dist_from_LB_in_sec, ego_lon-LB_lon))

                            state = State(None, [], ego)
                            behavioral_state = BehavioralGridState.create_from_state(state, logger)

                            action_specs = [action_space.specify_goal(recipe, behavioral_state) for i, recipe in
                                            enumerate(action_space.recipes)]

                            specs = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                                     if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.aggressiveness == AggressivenessLevel.CALM
                                     and recipe.velocity == target_vel]

                            state = State(None, objects, ego)
                            behavioral_state = BehavioralGridState.create_from_state(state, logger)

                            actions = []
                            # create actions based on the occupancy grid
                            # add dynamic and static actions going to the same lane
                            F_exists = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid
                            if F_exists:
                                right_action = DynamicActionRecipe(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT,
                                                                   ActionType.FOLLOW_VEHICLE, AggressivenessLevel.CALM)
                                actions.append(right_action)
                            # Skip static action if its velocity > F_vel, since it's cost always will be better than
                            # goto_left_lane, regardless F_vel.
                            # This condition will be removed, when a real value function will be used.
                            if not F_exists or des_vel <= F_vel + BP_MAX_VELOCITY_TOLERANCE:
                                actions.append(StaticActionRecipe(RelativeLane.SAME_LANE, des_vel, AggressivenessLevel.CALM))

                            # add dynamic and static actions to the left lane
                            if (RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
                                left_action = DynamicActionRecipe(RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.FRONT,
                                                                  ActionType.FOLLOW_VEHICLE, AggressivenessLevel.CALM)
                                actions.append(left_action)
                            actions.append(StaticActionRecipe(RelativeLane.LEFT_LANE, des_vel, AggressivenessLevel.CALM))

                            costs = evaluator.evaluate(behavioral_state, actions, [True] * len(actions))

                            best_idx = np.argmin(costs)
                            # if F is far or too close
                            if dist_from_F_in_sec > 3.5 or F_vel >= des_vel or \
                                    dist_from_LB_in_sec < 2.5:  # or (dist_from_LB_in_sec < 3.5 and LB_vel > ego_vel):
                                assert actions[best_idx].relative_lane == RelativeLane.SAME_LANE  # don't change lane
                            else:  # F is close
                                assert actions[best_idx].relative_lane == RelativeLane.LEFT_LANE  # change lane


def test_behavioralScenarios_moveToLeft_aggressivenessBySafety():
    """
    Test lane change to the left if the velocity of the car F is des_vel-2.
    Test different distances from F, different initial velocities of ego, empty/occupied left lane, longitude of LB.
    Desired result: for all initial velocities, don't overtake when dist=4 sec and overtake when dist=3 sec.
    :return:
    """
    logger = Logger("test_behavioralScenarios")
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 150
    road_id = 20
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)
    init_vel = 10
    F_vel = init_vel
    LB_vel = des_vel + 2

    evaluator = HeuristicActionRecipeEvaluator(logger)

    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat-lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, init_vel, 0, 0, 0, 0)

    for dist_from_F_in_sec in [4, 2.5]:
        for dist_from_LB_in_sec in [4.0, 3.5]:

            LB_lon = ego_lon - des_vel * dist_from_LB_in_sec
            F_lon = ego_lon + dist_from_F_in_sec * init_vel

            F_cpoint, F_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, F_lon, road_mid_lat-lane_width)
            F = DynamicObject(1, 0, F_cpoint[0], F_cpoint[1], F_cpoint[2], F_yaw, size, 0, F_vel, 0, 0, 0)

            LB_cpoint, LB_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, LB_lon, road_mid_lat)
            LB = DynamicObject(2, 0, LB_cpoint[0], LB_cpoint[1], LB_cpoint[2], LB_yaw, size, 0, LB_vel, 0, 0, 0)
            objects = [F, LB]

            right_action = DynamicActionRecipe(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT,
                                               ActionType.FOLLOW_VEHICLE, AggressivenessLevel.CALM)
            left_action1 = StaticActionRecipe(RelativeLane.LEFT_LANE, des_vel, AggressivenessLevel.CALM)
            left_action2 = StaticActionRecipe(RelativeLane.LEFT_LANE, des_vel, AggressivenessLevel.STANDARD)
            left_action3 = StaticActionRecipe(RelativeLane.LEFT_LANE, des_vel, AggressivenessLevel.AGGRESSIVE)
            actions = [right_action, left_action1, left_action2, left_action3]

            state = State(None, objects, ego)
            behavioral_state = BehavioralGridState.create_from_state(state, logger)
            costs = evaluator.evaluate(behavioral_state, actions, [True]*len(actions))
            best_idx = np.argmin(costs)

            if dist_from_F_in_sec > 3:  # F is far
                if dist_from_LB_in_sec >= 4:  # LB is far enough for STANDARD overtake, but too close for CALM
                    assert actions[best_idx].relative_lane == RelativeLane.LEFT_LANE and \
                           actions[best_idx].aggressiveness == AggressivenessLevel.STANDARD
                else:  # LB is close
                    assert actions[best_idx].relative_lane == RelativeLane.SAME_LANE  # don't overtake aggressively
            else:  # F is close
                if dist_from_LB_in_sec >= 4:  # LB is far; can't overtake by STANDARD because of F
                    assert actions[best_idx].relative_lane == RelativeLane.LEFT_LANE and \
                           actions[best_idx].aggressiveness == AggressivenessLevel.CALM
                else:  # LB is close
                    assert actions[best_idx].relative_lane == RelativeLane.SAME_LANE  # don't change lane


def test_behavioralScenarios_moveToToRight_differentDistFromRF():
    """
    test return to the right lane
    """
    logger = Logger("test_behavioralScenarios")
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 150
    road_id = 20
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    RF_vel = des_vel-2
    logger = Logger("test_BP_metrics")
    size = ObjectSize(4, 2, 1)

    left_action = StaticActionRecipe(RelativeLane.SAME_LANE, des_vel, AggressivenessLevel.CALM)
    right_action = StaticActionRecipe(RelativeLane.RIGHT_LANE, des_vel, AggressivenessLevel.CALM)
    actions = [right_action, left_action]
    evaluator = HeuristicActionRecipeEvaluator(logger)

    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, des_vel, 0, 0, 0, 0)

    lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width
    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)
    costs = evaluator.evaluate(behavioral_state, actions, [True]*len(actions))
    assert costs[0] < costs[1]  # right is better

    for dist_from_RF_in_sec in [7, 4]:
        RF_lon = ego_lon + dist_from_RF_in_sec * des_vel
        right_action = DynamicActionRecipe(RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.FRONT,
                                           ActionType.FOLLOW_VEHICLE, AggressivenessLevel.CALM)
        actions = [right_action, left_action]

        RF_cpoint, RF_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, RF_lon, road_mid_lat-lane_width)
        RF = DynamicObject(1, 0, RF_cpoint[0], RF_cpoint[1], RF_cpoint[2], RF_yaw, size, 0, RF_vel, 0, 0, 0)

        state = State(None, [RF], ego)
        behavioral_state = BehavioralGridState.create_from_state(state, logger)

        costs = evaluator.evaluate(behavioral_state, actions, [True]*len(actions))

        best_idx = np.argmin(costs)
        if dist_from_RF_in_sec > 6:
            assert actions[best_idx].relative_lane == RelativeLane.RIGHT_LANE  # change lane to right
        else:
            assert actions[best_idx].relative_lane == RelativeLane.SAME_LANE  # stay in the left lane


def test_speedProfiling():
    logger = Logger("test_behavioralScenarios")
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 250
    road_id = 20
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)
    init_vel = 10
    F_vel = init_vel
    LB_vel = des_vel + 2

    evaluator = HeuristicActionSpecEvaluator(logger)

    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, init_vel, 0, 0, 0, 0)

    dist_from_F_in_sec = 3.0
    dist_from_LB_in_sec  = 4.0

    LB_lon = ego_lon - des_vel * dist_from_LB_in_sec
    F_lon = ego_lon + dist_from_F_in_sec * init_vel

    F_cpoint, F_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, F_lon, road_mid_lat)
    F = DynamicObject(1, 0, F_cpoint[0], F_cpoint[1], F_cpoint[2], F_yaw, size, 0, F_vel, 0, 0, 0)

    LF_cpoint, LF_yaw = MapService.get_instance().convert_road_to_global_coordinates(
        road_id, F_lon, road_mid_lat+lane_width)
    LF = DynamicObject(2, 0, LF_cpoint[0], LF_cpoint[1], LF_cpoint[2], LF_yaw, size, 0, des_vel, 0, 0, 0)

    L_cpoint, L_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat+lane_width)
    L = DynamicObject(3, 0, L_cpoint[0], L_cpoint[1], L_cpoint[2], L_yaw, size, 0, des_vel, 0, 0, 0)

    LB_cpoint, LB_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, LB_lon, road_mid_lat+lane_width)
    LB = DynamicObject(4, 0, LB_cpoint[0], LB_cpoint[1], LB_cpoint[2], LB_yaw, size, 0, LB_vel, 0, 0, 0)

    RF_cpoint, RF_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, F_lon, road_mid_lat-lane_width)
    RF = DynamicObject(5, 0, RF_cpoint[0], RF_cpoint[1], RF_cpoint[2], RF_yaw, size, 0, des_vel, 0, 0, 0)

    R_cpoint, R_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat-lane_width)
    R = DynamicObject(6, 0, R_cpoint[0], R_cpoint[1], R_cpoint[2], R_yaw, size, 0, des_vel, 0, 0, 0)

    RB_cpoint, RB_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, LB_lon, road_mid_lat-lane_width)
    RB = DynamicObject(7, 0, RB_cpoint[0], RB_cpoint[1], RB_cpoint[2], RB_yaw, size, 0, LB_vel, 0, 0, 0)

    B_cpoint, B_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, LB_lon, road_mid_lat)
    B = DynamicObject(8, 0, B_cpoint[0], B_cpoint[1], B_cpoint[2], B_yaw, size, 0, F_vel, 0, 0, 0)

    objects = [F, LF, L, LB, RF, R, RB, B]

    state = State(None, objects, ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger), DynamicActionSpace(logger, predictor)])
    action_recipes = action_space.recipes

    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)

    action_specs = [action_space.specify_goal(recipe, behavioral_state) if recipes_mask[i] else None
                    for i, recipe in enumerate(action_recipes)]

    # ActionSpec filtering
    action_spec_validator = ActionSpecFiltering(action_spec_filter_bank.action_spec_filters)
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

                ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(
                    road_id, ego_lon, road_mid_lat - lane_width)
                ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

                F_lon = ego_lon + dist_from_F_in_sec * (ego_vel + des_vel)/2 + (des_vel**2 - F_vel**2)/16  # 3*(des_vel - F_vel)**2
                F_cpoint, F_yaw = MapService.get_instance().convert_road_to_global_coordinates(
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
