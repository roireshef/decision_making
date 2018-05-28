from typing import List

import numpy as np
import pytest

from decision_making.src.global_constants import EPS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import DynamicActionRecipe, ActionType, AggressivenessLevel, \
    StaticActionRecipe
from decision_making.src.state.state import OccupancyState, State, NewEgoState, DynamicObject, ObjectSize, NewDynamicObject
from mapping.src.api.map_api import MapAPI
from mapping.src.service.map_service import MapService, MapServiceArgs


@pytest.fixture(scope='function')
def pg_map_api():
    MapService.initialize(MapServiceArgs(map_source='TestingGroundMap3Lanes.bin'))
    yield MapService.get_instance()


@pytest.fixture(scope='function')
def state_with_sorrounding_objects(pg_map_api: MapAPI):
    road_id = 20

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    center_lanes_latitudes = pg_map_api.get_center_lanes_latitudes(road_id)

    # Ego state
    ego_road_lon = 50.0
    ego_road_lat = center_lanes_latitudes[1]
    ego_vel = 10

    ego_pos, ego_yaw = pg_map_api.convert_road_to_global_coordinates(road_id=road_id, lon=ego_road_lon,
                                                                     lat=ego_road_lat)

    ego_state = NewEgoState.create_from_cartesian_state(
        obj_id=0, timestamp=0, cartesian_state=np.array([ego_pos[0], ego_pos[1], ego_yaw, ego_vel, 0.0, 0]),
        size=car_size, confidence=1.0)

    # Generate objects at the following locations:
    obj_road_lons = [ego_road_lon - 20, ego_road_lon, ego_road_lon + 20]
    obj_road_lats = center_lanes_latitudes

    dynamic_objects: List[NewDynamicObject] = list()
    obj_id = 1
    for obj_road_lon in obj_road_lons:
        for obj_road_lat in obj_road_lats:

            if obj_road_lon == ego_road_lon and obj_road_lat == ego_road_lat:
                # Don't create an object where the ego is
                continue

            obj_pos, obj_yaw = pg_map_api.convert_road_to_global_coordinates(road_id=road_id,
                                                                             lon=obj_road_lon,
                                                                             lat=obj_road_lat)

            dynamic_object = NewDynamicObject.create_from_cartesian_state(
                obj_id=obj_id, timestamp=0, size=car_size, confidence=1.0,
                cartesian_state=np.array([obj_pos[0],obj_pos[1], obj_yaw, ego_vel, 0.0, 0.0]))

            dynamic_objects.append(dynamic_object)
            obj_id += 1

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


@pytest.fixture(scope='function')
def behavioral_grid_state(state_with_sorrounding_objects: State):
    yield BehavioralGridState.create_from_state(state_with_sorrounding_objects, None)


@pytest.fixture(scope='function')
def follow_vehicle_recipes_towards_front_cells():
    yield [DynamicActionRecipe(lane, RelativeLongitudinalPosition.FRONT, ActionType.FOLLOW_VEHICLE, agg)
           for lane in RelativeLane
           for agg in AggressivenessLevel]


@pytest.fixture(scope='function')
def follow_lane_recipes():
    velocity_grid = np.arange(0, 20 + EPS, 10/3.6)
    yield [StaticActionRecipe(lane, velocity, agg)
           for lane in RelativeLane
           for velocity in velocity_grid
           for agg in AggressivenessLevel]
