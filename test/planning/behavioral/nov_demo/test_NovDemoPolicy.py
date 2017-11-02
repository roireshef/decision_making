import numpy as np
from decision_making.src.planning.behavioral.policies.november_demo_semantic_policy import NovDemoPolicy, \
    NovDemoBehavioralState
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import OccupancyState, ObjectSize, DynamicObject, EgoState, State
from mapping.src.model.map_api import MapAPI
from mapping.test.model.testable_map_fixtures import testable_map_api
from decision_making.test.planning.behavioral.nov_demo.test_NovDemoBehavioralState import state_with_sorrounding_objects
from rte.python.logger.AV_logger import AV_Logger
import pytest


@pytest.fixture(scope='function')
def state_with_ego_on_right_lane(testable_map_api: MapAPI):
    map_api = testable_map_api

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_road_id = 1
    ego_road_lon = 15.0
    ego_road_lat = 1.5

    ego_pos, ego_yaw = map_api._convert_road_to_global_coordinates(road_id=ego_road_id, lon=ego_road_lon,
                                                                   lat=ego_road_lat)

    road_localization = DynamicObject.compute_road_localization(global_pos=ego_pos, global_yaw=ego_yaw,
                                                                map_api=map_api)
    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_pos[0], y=ego_pos[1], z=ego_pos[2], yaw=ego_yaw,
                         size=car_size, confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=road_localization)

    dynamic_objects = []

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_ego_on_left_lane(testable_map_api: MapAPI):
    map_api = testable_map_api

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_road_id = 1
    ego_road_lon = 15.0
    ego_road_lat = 7.5

    ego_pos, ego_yaw = map_api._convert_road_to_global_coordinates(road_id=ego_road_id, lon=ego_road_lon,
                                                                   lat=ego_road_lat)

    road_localization = DynamicObject.compute_road_localization(global_pos=ego_pos, global_yaw=ego_yaw,
                                                                map_api=map_api)
    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_pos[0], y=ego_pos[1], z=ego_pos[2], yaw=ego_yaw,
                         size=car_size, confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=road_localization)

    dynamic_objects = []

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


def test_enumerate_actions_gridFull_allActionsEnumerated(state_with_sorrounding_objects, testable_map_api):
    logger = AV_Logger.get_logger('Nov demo - semantic occupancy grid')
    map_api = testable_map_api
    state = state_with_sorrounding_objects
    predictor = RoadFollowingPredictor(map_api=map_api)

    policy = NovDemoPolicy(logger=logger, policy_config=None, predictor=predictor, map_api=map_api)

    behavioral_state = NovDemoBehavioralState.create_from_state(state=state, map_api=map_api, logger=logger)
    actions = policy._enumerate_actions(behavioral_state=behavioral_state)

    action_index = 0
    lane = -1
    lon = 1
    obj_id = 9

    cell = (lane, lon)
    assert actions[action_index].cell == cell and actions[action_index].target_obj is not None \
           and actions[action_index].target_obj.obj_id == obj_id

    action_index = 1
    lane = 0
    lon = 1
    obj_id = 10

    cell = (lane, lon)
    assert actions[action_index].cell == cell and actions[action_index].target_obj is not None \
           and actions[action_index].target_obj.obj_id == obj_id

    action_index = 2
    lane = 1
    lon = 1
    obj_id = 11

    cell = (lane, lon)
    assert actions[action_index].cell == cell and actions[action_index].target_obj is not None \
           and actions[action_index].target_obj.obj_id == obj_id


def test_enumerate_actions_egoAtRoadEdge_filterOnlyValidActions(state_with_sorrounding_objects, testable_map_api,
                                                                state_with_ego_on_right_lane,
                                                                state_with_ego_on_left_lane):
    logger = AV_Logger.get_logger('Nov demo - semantic occupancy grid')
    map_api = testable_map_api
    predictor = RoadFollowingPredictor(map_api=map_api)

    policy = NovDemoPolicy(logger=logger, policy_config=None, predictor=predictor, map_api=map_api)


    # Check that when car is on right lane we get only 2 valid actions
    state = state_with_ego_on_right_lane
    behavioral_state = NovDemoBehavioralState.create_from_state(state=state, map_api=map_api, logger=logger)
    actions = policy._enumerate_actions(behavioral_state=behavioral_state)

    action_index = 0
    lane = 0
    lon = 1
    cell = (lane, lon)
    assert actions[action_index].cell == cell

    action_index = 1
    lane = 1
    lon = 1
    cell = (lane, lon)
    assert actions[action_index].cell == cell

    assert len(actions) == 2

    # Check that when car is on left lane we get only 2 valid actions
    state = state_with_ego_on_left_lane
    behavioral_state = NovDemoBehavioralState.create_from_state(state=state, map_api=map_api, logger=logger)
    actions = policy._enumerate_actions(behavioral_state=behavioral_state)

    action_index = 0
    lane = -1
    lon = 1
    cell = (lane, lon)
    assert actions[action_index].cell == cell

    action_index = 1
    lane = 0
    lon = 1
    cell = (lane, lon)
    assert actions[action_index].cell == cell

    assert len(actions) == 2
