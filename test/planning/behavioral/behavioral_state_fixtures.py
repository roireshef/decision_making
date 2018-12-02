from typing import List

import numpy as np
import pytest

from decision_making.src.global_constants import EPS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, ActionType, AggressivenessLevel, \
    StaticActionRecipe
from decision_making.src.planning.types import CartesianPoint2D, FrenetPoint, FP_SX
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import OccupancyState, State, ObjectSize, EgoState, DynamicObject
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.model.map_api import MapAPI
from mapping.src.service.map_service import MapService


@pytest.fixture(scope='function')
def pg_map_api():
    MapService.initialize(map_file='TestingGroundMap3Lanes.bin')
    yield MapService.get_instance()


@pytest.fixture(scope='function')
def state_with_sorrounding_objects(pg_map_api: MapAPI):
    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = 50.0
    obj_vel = ego_vel = 10
    ego_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), ego_lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Generate objects at the following locations:
    obj_lane_lons = [ego_lane_lon - 20, ego_lane_lon, ego_lane_lon + 20]

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1
    obj_lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_id)
    for obj_lane_lon in obj_lane_lons:
        for obj_lane_id in obj_lane_ids:

            if obj_lane_lon == ego_lane_lon and obj_lane_id == ego_lane_id:
                # Don't create an object where the ego is
                continue

            map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), obj_lane_id)
            dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                            size=car_size, confidence=1.)
            dynamic_objects.append(dynamic_object)
            obj_id += 1

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_tracking_mode(pg_map_api: MapAPI):
    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = 50.0
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Generate objects at the following locations:
    obj_lane_lon = ego_lane_lon + 20
    obj_vel = 10.2

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

    dynamic_objects.append(dynamic_object)

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_negative_sT(pg_map_api: MapAPI):
    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = 50.0
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Generate objects at the following locations:
    obj_lane_lon = ego_lane_lon + 3.8
    obj_vel = 11

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

    dynamic_objects.append(dynamic_object)

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def state_with_objects_for_filtering_too_aggressive(pg_map_api: MapAPI):
    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_lane_lon = 50.0
    ego_vel = 10
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[1]

    map_state = MapState(np.array([ego_lane_lon, ego_vel, 0, 0, 0, 0]), lane_id)
    ego_state = EgoState.create_from_map_state(obj_id=0, timestamp=0, map_state=map_state, size=car_size, confidence=1)

    # Generate objects at the following locations:
    obj_lane_lon = ego_lane_lon + 58
    obj_vel = 30

    dynamic_objects: List[DynamicObject] = list()
    obj_id = 1

    map_state = MapState(np.array([obj_lane_lon, obj_vel, 0, 0, 0, 0]), lane_id)
    dynamic_object = EgoState.create_from_map_state(obj_id=obj_id, timestamp=0, map_state=map_state,
                                                    size=car_size, confidence=1.)

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

