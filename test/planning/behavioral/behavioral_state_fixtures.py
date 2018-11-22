from typing import List

import numpy as np
import pytest

from decision_making.src.global_constants import EPS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, ActionType, AggressivenessLevel, \
    StaticActionRecipe, RelativeLane, RelativeLongitudinalPosition
from decision_making.src.state.state import OccupancyState, State, ObjectSize, EgoState, DynamicObject
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.model.map_api import MapAPI
from mapping.src.service.map_service import MapService


@pytest.fixture(scope='function')
def pg_map_api():
    MapService.initialize(map_file='TestingGroundMap3Lanes.bin')
    yield MapService.get_instance()


@pytest.fixture(scope='function')
def state_with_sorrounding_objects():
    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    lanes_list = MapUtils.get_lanes_by_road_segment_id(road_id)

    # Ego state
    ego_lon = 50.0
    ego_lat = 0
    ego_lane_id = lanes_list[1]
    ego_vel = 10
    lane_width = MapUtils.get_lane_width(ego_lane_id, 0)

    ego_pos, ego_yaw = MapUtils._convert_from_lane_to_map_coordinates(ego_lane_id, np.array([ego_lon, ego_lat]))

    ego_state = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0,
                                                     cartesian_state=np.array([ego_pos[0], ego_pos[1], ego_yaw, ego_vel, 0.0, 0.0]),
                                                     size=car_size, confidence=1.0)

    # Generate objects at the following locations, relative to the ego lane frame:
    obj_lons = [ego_lon - 20, ego_lon, ego_lon + 20]
    obj_lats = [-lane_width, 0, lane_width]

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1
    for obj_lon in obj_lons:
        for obj_lat in obj_lats:

            if obj_lon == ego_lon and obj_lat == ego_lat:
                # Don't create an object where the ego is
                continue

            obj_pos, obj_yaw = MapUtils._convert_from_lane_to_map_coordinates(ego_lane_id, np.array([obj_lon, obj_lat]))

            dynamic_object = DynamicObject.create_from_cartesian_state(obj_id=obj_id, timestamp=0,
                                                                       cartesian_state=np.array([obj_pos[0], obj_pos[1], obj_yaw, ego_vel, 0.0, 0.0]),
                                                                       size=car_size, confidence=1.0)
            dynamic_objects.append(dynamic_object)
            obj_id += 1

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_tracking_mode():
    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    lanes_list = MapUtils.get_lanes_by_road_segment_id(road_id)

    # Ego state
    ego_lon = 50.0
    ego_lat = 0
    ego_lane_id = lanes_list[1]
    ego_vel = 10

    ego_pos, ego_yaw = MapUtils._convert_from_lane_to_map_coordinates(ego_lane_id, np.array([ego_lon, ego_lat]))

    ego_state = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0,
                                                     cartesian_state=np.array([ego_pos[0], ego_pos[1], ego_yaw, ego_vel, 0.0, 0.0]),
                                                     size=car_size, confidence=1.0)

    # Generate objects at the following locations, relative to the ego lane frame:
    obj_lon = ego_lon + 20
    obj_lat = ego_lat
    obj_vel = 10.2

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    obj_pos, obj_yaw = MapUtils._convert_from_lane_to_map_coordinates(ego_lane_id, np.array([obj_lon, obj_lat]))

    dynamic_object = DynamicObject.create_from_cartesian_state(obj_id=obj_id, timestamp=0,
                                                               cartesian_state=np.array([obj_pos[0], obj_pos[1], obj_yaw, obj_vel, 0.0, 0.0]),
                                                               size=car_size, confidence=1.0)

    dynamic_objects.append(dynamic_object)

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_negative_sT():
    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    lanes_list = MapUtils.get_lanes_by_road_segment_id(road_id)

    # Ego state
    ego_lon = 50.0
    ego_lat = 0
    ego_lane_id = lanes_list[1]
    ego_vel = 10

    ego_pos, ego_yaw = MapUtils._convert_from_lane_to_map_coordinates(ego_lane_id, np.array([ego_lon, ego_lat]))

    ego_state = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0,
                                                     cartesian_state=np.array([ego_pos[0], ego_pos[1], ego_yaw, ego_vel, 0.0, 0.0]),
                                                     size=car_size, confidence=1.0)

    # Generate objects at the following locations, relative to the ego lane frame:
    obj_lon = ego_lon + 3.8
    obj_lat = ego_lat
    obj_vel = 11

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    obj_pos, obj_yaw = MapUtils._convert_from_lane_to_map_coordinates(ego_lane_id, np.array([obj_lon, obj_lat]))

    dynamic_object = DynamicObject.create_from_cartesian_state(obj_id=obj_id, timestamp=0,
                                                               cartesian_state=np.array([obj_pos[0], obj_pos[1], obj_yaw, obj_vel, 0.0, 0.0]),
                                                               size=car_size, confidence=1.0)

    dynamic_objects.append(dynamic_object)

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_too_aggressive(pg_map_api: MapAPI):
    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    lanes_list = MapUtils.get_lanes_by_road_segment_id(road_id)

    # Ego state
    ego_lon = 50.0
    ego_lat = 0
    ego_lane_id = lanes_list[1]
    ego_vel = 10

    ego_pos, ego_yaw = MapUtils._convert_from_lane_to_map_coordinates(ego_lane_id, np.array([ego_lon, ego_lat]))

    ego_state = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0,
                                                     cartesian_state=np.array([ego_pos[0], ego_pos[1], ego_yaw, ego_vel, 0.0, 0.0]),
                                                     size=car_size, confidence=1.0)

    # Generate objects at the following locations:
    obj_lon = ego_lon + 58
    obj_lat = ego_lat
    obj_vel = 30

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    obj_pos, obj_yaw = MapUtils._convert_from_lane_to_map_coordinates(ego_lane_id, np.array([obj_lon, obj_lat]))

    dynamic_object = DynamicObject.create_from_cartesian_state(obj_id=obj_id, timestamp=0,
                                                               cartesian_state=np.array([obj_pos[0], obj_pos[1], obj_yaw, obj_vel, 0.0, 0.0]),
                                                               size=car_size, confidence=1.0)

    dynamic_objects.append(dynamic_object)

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def behavioral_grid_state(state_with_sorrounding_objects: State):
    yield BehavioralGridState.create_from_state(state_with_sorrounding_objects, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_filtering_tracking_mode(
        state_with_objects_for_filtering_tracking_mode: State):
    yield BehavioralGridState.create_from_state(state_with_objects_for_filtering_tracking_mode, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_filtering_negative_sT(state_with_objects_for_filtering_negative_sT: State):
    yield BehavioralGridState.create_from_state(state_with_objects_for_filtering_negative_sT, None)


@pytest.fixture(scope='function')
def behavioral_grid_state_with_objects_for_filtering_too_aggressive(
        state_with_objects_for_filtering_too_aggressive: State):
    yield BehavioralGridState.create_from_state(state_with_objects_for_filtering_too_aggressive, None)


@pytest.fixture(scope='function')
def follow_vehicle_recipes_towards_front_cells():
    yield [DynamicActionRecipe(lane, RelativeLongitudinalPosition.FRONT, ActionType.FOLLOW_VEHICLE, agg)
           for lane in RelativeLane
           for agg in AggressivenessLevel]


@pytest.fixture(scope='function')
def follow_lane_recipes():
    velocity_grid = np.arange(0, 30 + EPS, 6)
    yield [StaticActionRecipe(RelativeLane.SAME_LANE, velocity, agg)
           for velocity in velocity_grid
           for agg in AggressivenessLevel]
