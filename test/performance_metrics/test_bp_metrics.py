from unittest.mock import patch
import pytest
from logging import Logger

import numpy as np

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.planning.behavioral.architecture.components.evaluators.heuristic_state_action_recipe_evaluator import \
    HeuristicStateActionRecipeEvaluator
from decision_making.src.planning.behavioral.architecture.data_objects import ActionType, AggressivenessLevel, \
    ActionRecipe, RelativeLane, DynamicActionRecipe, RelativeLongitudinalPosition, StaticActionRecipe
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState
from decision_making.src.state.state import DynamicObject, ObjectSize, EgoState, State
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.src.service.map_service import MapService
from mapping.test.model.testable_map_fixtures import map_api_mock


#@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
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
    ego_lon = 150.
    F_vel = des_vel - 2.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    right_action = DynamicActionRecipe(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT,
                                       ActionType.FOLLOW_VEHICLE, AggressivenessLevel.CALM)
    evaluator = HeuristicStateActionRecipeEvaluator(logger)

    for dist_from_F_in_sec in [6., 4., 2.5]:
        for init_vel in np.arange(des_vel-6, des_vel+2.1, 2):
            for left_action_type in [ActionType.FOLLOW_LANE, ActionType.FOLLOW_VEHICLE]:
                for LB_vel in np.arange(des_vel - 2, des_vel + 2.1, 2):
                    for dist_from_LB_in_sec in [3., 2.]:

                        ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(
                            road_id, ego_lon, road_mid_lat - lane_width)
                        ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, init_vel, 0, 0, 0, 0)

                        F_lon = ego_lon + dist_from_F_in_sec * (init_vel + des_vel)/2
                        F_cpoint, F_yaw = MapService.get_instance().convert_road_to_global_coordinates(
                            road_id, F_lon, road_mid_lat - lane_width)
                        F = DynamicObject(1, 0, F_cpoint[0], F_cpoint[1], F_cpoint[2], F_yaw, size, 0, F_vel, 0, 0, 0)

                        LB_lon = ego_lon - (LB_vel - init_vel) * 3 - LB_vel * dist_from_LB_in_sec
                        LB_cpoint, LB_yaw = MapService.get_instance().convert_road_to_global_coordinates(
                            road_id, LB_lon, road_mid_lat)
                        LB = DynamicObject(2, 0, LB_cpoint[0], LB_cpoint[1], LB_cpoint[2], LB_yaw, size, 0, LB_vel, 0, 0, 0)
                        objects = [F, LB]

                        if left_action_type == ActionType.FOLLOW_VEHICLE:

                            LF_cpoint, LF_yaw = MapService.get_instance().convert_road_to_global_coordinates(
                                road_id, F_lon, road_mid_lat)
                            LF = DynamicObject(2, 0, LF_cpoint[0], LF_cpoint[1], LF_cpoint[2], LF_yaw, size, 0, des_vel, 0, 0, 0)
                            objects.append(LF)
                            left_action = DynamicActionRecipe(RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.FRONT,
                                                              ActionType.FOLLOW_VEHICLE, AggressivenessLevel.CALM)
                        else:
                            left_action = StaticActionRecipe(RelativeLane.LEFT_LANE, des_vel, AggressivenessLevel.CALM)
                        actions = [right_action, left_action]
                        print('dist_from_F=%.2f  LB_vel=%.2f  dist_from_LB=%.2fs %.2fm' % (dist_from_F_in_sec, LB_vel, dist_from_LB_in_sec, ego_lon-LB_lon))

                        state = State(None, objects, ego)
                        behavioral_state = SemanticBehavioralGridState.create_from_state(state, logger)
                        costs = evaluator.evaluate_recipes(behavioral_state, actions, [True] * len(actions), None)

                        best_idx = np.argmin(costs)
                        if dist_from_F_in_sec > 3.5 or dist_from_LB_in_sec < 2.5:  # if F is far or too close
                            assert actions[best_idx].relative_lane == RelativeLane.SAME_LANE  # don't change lane
                        else:  # F is close
                            assert actions[best_idx].relative_lane == RelativeLane.LEFT_LANE  # change lane


#@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
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

    evaluator = HeuristicStateActionRecipeEvaluator(logger)

    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat-lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, init_vel, 0, 0, 0, 0)

    for dist_from_F_in_sec in [4, 3]:
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
            behavioral_state = SemanticBehavioralGridState.create_from_state(state, logger)
            costs = evaluator.evaluate_recipes(behavioral_state, actions, [True]*len(actions), None)
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


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_behavioralScenarios_moveToToRight_differentDistFromRF():
    """
    test return to the right lane
    """
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    ego_lon = 50
    RF_vel = des_vel-2
    logger = Logger("test_BP_metrics")
    size = ObjectSize(4, 2, 1)

    left_action = StaticActionRecipe(RelativeLane.SAME_LANE, des_vel, AggressivenessLevel.CALM)
    right_action = StaticActionRecipe(RelativeLane.RIGHT_LANE, des_vel, AggressivenessLevel.CALM)
    actions = [right_action, left_action]
    evaluator = HeuristicStateActionRecipeEvaluator(logger)
    ego = EgoState(0, 0, ego_lon, 0, 0, 0, size, 0, des_vel, 0, 0, 0, 0)
    lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width
    state = State(None, [], ego)
    behavioral_state = SemanticBehavioralGridState.create_from_state(state, logger)
    costs = evaluator.evaluate_recipes(behavioral_state, actions, [True]*len(actions), None)
    assert costs[0] < costs[1]  # right is better

    for dist_from_F_in_sec in [7, 4]:
        RF_lon = ego_lon + dist_from_F_in_sec * des_vel
        right_action = DynamicActionRecipe(RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.FRONT,
                                           ActionType.FOLLOW_VEHICLE, AggressivenessLevel.CALM)
        actions = [right_action, left_action]
        RF = DynamicObject(1, 0, RF_lon, -lane_width, 0, 0, size, 0, RF_vel, 0, 0, 0)
        state = State(None, [RF], ego)
        behavioral_state = SemanticBehavioralGridState.create_from_state(state, logger)
        costs = evaluator.evaluate_recipes(behavioral_state, actions, [True]*len(actions), None)
        best_idx = np.argmin(costs)
        if dist_from_F_in_sec > 6:
            assert actions[best_idx].relative_lane == RelativeLane.RIGHT_LANE  # change lane to right
        else:
            assert actions[best_idx].relative_lane == RelativeLane.SAME_LANE  # stay in the left lane
