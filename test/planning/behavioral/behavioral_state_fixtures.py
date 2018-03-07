from typing import List

import numpy as np
import pytest

from decision_making.src.planning.behavioral.policies.semantic_actions_grid_policy import SemanticActionsGridPolicy
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_state import \
    SemanticActionsGridState
from decision_making.src.planning.behavioral.policies.semantic_actions_policy import SemanticAction, SemanticActionType
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import OccupancyState, State, EgoState, DynamicObject, ObjectSize
from rte.python.logger.AV_logger import AV_Logger


@pytest.fixture(scope='function')
def state_with_sorrounding_objects(testable_map_api):
    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_road_id = 1
    ego_road_lon = 15.0
    ego_road_lat = 4.5

    ego_pos, ego_yaw = testable_map_api.convert_road_to_global_coordinates(road_id=ego_road_id, lon=ego_road_lon,
                                                                            lat=ego_road_lat)

    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_pos[0], y=ego_pos[1], z=ego_pos[2], yaw=ego_yaw,
                         size=car_size, confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0)

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

            obj_pos, obj_yaw = testable_map_api.convert_road_to_global_coordinates(road_id=obj_road_id,
                                                                                    lon=obj_road_lon,
                                                                                    lat=obj_road_lat)

            dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_pos[0], y=obj_pos[1], z=obj_pos[2],
                                           yaw=obj_yaw, size=car_size, confidence=1.0, v_x=0.0, v_y=0.0,
                                           acceleration_lon=0.0, omega_yaw=0.0)

            dynamic_objects.append(dynamic_object)
            obj_id += 1

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_ego_on_right_lane(testable_map_api):
    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_road_id = 1
    ego_road_lon = 15.0
    ego_road_lat = 1.5

    ego_pos, ego_yaw = testable_map_api.convert_road_to_global_coordinates(road_id=ego_road_id, lon=ego_road_lon,
                                                                   lat=ego_road_lat)

    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_pos[0], y=ego_pos[1], z=ego_pos[2], yaw=ego_yaw,
                         size=car_size, confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0)

    dynamic_objects = []

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_ego_on_left_lane(testable_map_api):
    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_road_id = 1
    ego_road_lon = 15.0
    ego_road_lat = 7.5

    ego_pos, ego_yaw = testable_map_api.convert_road_to_global_coordinates(road_id=ego_road_id, lon=ego_road_lon,
                                                                   lat=ego_road_lat)

    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_pos[0], y=ego_pos[1], z=ego_pos[2], yaw=ego_yaw,
                         size=car_size, confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0)

    dynamic_objects = []

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def semantic_state():
    ego_state = EgoState(obj_id=0, timestamp=0, x=15.0, y=0.0, z=0.0, yaw=0.0,
                         size=ObjectSize(length=2.5, width=1.5, height=1.0), confidence=1.0, v_x=7.0, v_y=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, steering_angle=0.0)

    obj = DynamicObject(acceleration_lon=0.0, confidence=1.0, obj_id=9, omega_yaw=0.0,
                        size=ObjectSize(height=1.0, length=2.5, width=1.5), timestamp=0, v_x=10.0, v_y=0.0, x=20.0,
                        y=-3.0, yaw=0.0, z=0.0)

    occupancy_state = OccupancyState(0, np.array([]), np.array([]))
    yield State(occupancy_state, [obj], ego_state)


@pytest.fixture(scope='function')
def semantic_actions_state(semantic_state: State):
    obj = semantic_state.dynamic_objects[0]
    yield SemanticActionsGridState({(-1, 1): [obj]}, semantic_state.ego_state)


@pytest.fixture(scope='function')
def semantic_follow_action(semantic_actions_state: SemanticActionsGridState):
    obj = semantic_actions_state.road_occupancy_grid[(-1, 1)][0]
    yield SemanticAction((-1, 1), obj, SemanticActionType.FOLLOW_VEHICLE)


@pytest.fixture(scope='function')
def semantic_grid_policy():
    logger = AV_Logger.get_logger('Semantic occupancy grid')
    policy = SemanticActionsGridPolicy(logger, RoadFollowingPredictor(logger=logger))
    yield policy
