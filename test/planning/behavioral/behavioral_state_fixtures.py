from typing import List

import numpy as np
import pytest

from decision_making.src.planning.behavioral.policies.november_demo_semantic_policy import NovDemoBehavioralState, \
    NovDemoPolicy
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticAction, SemanticActionType
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import OccupancyState, State, EgoState, DynamicObject, ObjectSize, RoadLocalization
from mapping.src.model.map_api import MapAPI
from rte.python.logger.AV_logger import AV_Logger
import pytest

@pytest.fixture(scope='function')
def state_with_sorrounding_objects(testable_map_api: MapAPI):
    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_road_id = 1
    ego_road_lon = 15.0
    ego_road_lat = 4.5

    ego_pos, ego_yaw = testable_map_api._convert_road_to_global_coordinates(road_id=ego_road_id, lon=ego_road_lon,
                                                                            lat=ego_road_lat)

    road_localization = DynamicObject.compute_road_localization(global_pos=ego_pos, global_yaw=ego_yaw,
                                                                map_api=testable_map_api)
    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_pos[0], y=ego_pos[1], z=ego_pos[2], yaw=ego_yaw,
                         size=car_size, confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=road_localization)

    # Generate objects at the following locations:
    obj_id = 1
    obj_road_id = 1
    obj_road_lons = [5.0, 10.0, 15.0, 20.0, 25.0]
    obj_road_lats = [1.5, 4.5, 6.0]

    dynamic_objects: List[DynamicObject] = list()
    for obj_road_lon in obj_road_lons:
        for obj_road_lat in obj_road_lats:

            if obj_road_lon == ego_road_lon and obj_road_lat == ego_road_lat:
                # Don't create an object where the ego is
                continue

            obj_pos, obj_yaw = testable_map_api._convert_road_to_global_coordinates(road_id=obj_road_id,
                                                                                    lon=obj_road_lon,
                                                                                    lat=obj_road_lat)

            road_localization = DynamicObject.compute_road_localization(global_pos=obj_pos, global_yaw=obj_yaw,
                                                                        map_api=testable_map_api)
            dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_pos[0], y=obj_pos[1], z=obj_pos[2],
                                           yaw=obj_yaw, size=car_size, confidence=1.0, v_x=0.0, v_y=0.0,
                                           acceleration_lon=0.0, omega_yaw=0.0, road_localization=road_localization)

            dynamic_objects.append(dynamic_object)
            obj_id += 1

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


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


@pytest.fixture(scope='function')
def nov_demo_state():
    ego_state = EgoState(obj_id=0, timestamp=0, x=15.0, y=0.0, z=0.0, yaw=0.0,
                         size=ObjectSize(length=2.5, width=1.5, height=1.0), confidence=1.0, v_x=7.0, v_y=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, steering_angle=0.0,
                         road_localization=RoadLocalization(full_lat=4.5, intra_lane_lat=1.5, intra_lane_yaw=0.0,
                                                            lane_num=1, road_id=1, road_lon=15.0))

    obj = DynamicObject(acceleration_lon=0.0, confidence=1.0, obj_id=9, omega_yaw=0.0,
                        road_localization=RoadLocalization(full_lat=1.5, intra_lane_lat=1.5, intra_lane_yaw=0.0,
                                                           lane_num=0, road_id=1, road_lon=20.0),
                        size=ObjectSize(height=1.0, length=2.5, width=1.5), timestamp=0, v_x=10.0, v_y=0.0, x=20.0,
                        y=-3.0, yaw=0.0, z=0.0)

    yield State(None, [obj], ego_state)


@pytest.fixture(scope='function')
def nov_demo_semantic_behavioral_state(nov_demo_state: State):
    obj = nov_demo_state.dynamic_objects[0]
    yield NovDemoBehavioralState({(-1, 1): [obj]}, nov_demo_state.ego_state)


@pytest.fixture(scope='function')
def nov_demo_semantic_follow_action(nov_demo_semantic_behavioral_state: NovDemoBehavioralState):
    obj = nov_demo_semantic_behavioral_state.road_occupancy_grid[(-1, 1)][0]
    yield SemanticAction((-1, 1), obj, SemanticActionType.FOLLOW)


@pytest.fixture(scope='function')
def nov_demo_policy(testable_map_api: MapAPI):
    logger = AV_Logger.get_logger('Nov demo - semantic occupancy grid')
    policy = NovDemoPolicy(logger, None, RoadFollowingPredictor(testable_map_api, logger=logger), testable_map_api)
    yield policy
