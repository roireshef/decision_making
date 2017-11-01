import numpy as np
from typing import List
import pytest

from decision_making.src.planning.behavioral.policies.november_demo_semantic_policy import NovDemoBehavioralState
from decision_making.src.planning.behavioral.semantic_actions_policy import RoadSemanticOccupancyGrid
from decision_making.src.state.state import RoadLocalization, EgoState, ObjectSize, \
    DynamicObject, OccupancyState, State
from mapping.src.model.map_api import MapAPI
from rte.python.logger.AV_logger import AV_Logger
from mapping.test.model.testable_map_fixtures import testable_map_api


@pytest.fixture(scope='function')
def state_with_sorrounding_objects(testable_map_api: MapAPI):
    map_api = testable_map_api

    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    car_size = ObjectSize(length=2.5, width=1.5, height=1.0)

    # Ego state
    ego_road_id = 1
    ego_road_lon = 15.0
    ego_road_lat = 4.5

    ego_pos, ego_yaw = map_api._convert_road_to_global_coordinates(road_id=ego_road_id, lon=ego_road_lon,
                                                                   lat=ego_road_lat)

    road_localization = DynamicObject.compute_road_localization(global_pos=ego_pos, global_yaw=ego_yaw,
                                                                map_api=map_api)
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

            obj_pos, obj_yaw = map_api._convert_road_to_global_coordinates(road_id=obj_road_id, lon=obj_road_lon,
                                                                           lat=obj_road_lat)

            road_localization = DynamicObject.compute_road_localization(global_pos=obj_pos, global_yaw=obj_yaw,
                                                                        map_api=map_api)
            dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=obj_pos[0], y=obj_pos[1], z=obj_pos[2],
                                           yaw=obj_yaw, size=car_size, confidence=1.0, v_x=0.0, v_y=0.0,
                                           acceleration_lon=0.0, omega_yaw=0.0, road_localization=road_localization)

            dynamic_objects.append(dynamic_object)
            obj_id += 1

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


def test_generate_semantic_occupancy_grid_ComplexStateWithFullGrid_carsAreInRightCells(state_with_sorrounding_objects,
                                                                                       testable_map_api):
    """
    Here we generate a state with ego and dynamic obejcts, and verify that
    each object is mapped to the right cell in the grid.
    The implementation assigns only the closest cars in each cell, as detailed in the
    class documentation, therefore we expect to see only the relevant cars assigned.
    :return:
    """
    logger = AV_Logger.get_logger('Nov demo - semantic occupancy grid')

    map_api = testable_map_api
    state = state_with_sorrounding_objects
    occupancy_grid = NovDemoBehavioralState.create_from_state(state=state, map_api=map_api, logger=logger)


    # Assertion tests of objects in grid:

    # Closest cars behind ego:
    lane = -1
    lon = -1
    obj_id = 4

    cell = (lane, lon)
    assert cell in occupancy_grid.road_occupancy_grid and occupancy_grid.road_occupancy_grid[cell][0].obj_id == obj_id

    lane = 0
    lon = -1
    obj_id = 5

    cell = (lane, lon)
    assert cell in occupancy_grid.road_occupancy_grid and occupancy_grid.road_occupancy_grid[cell][0].obj_id == obj_id

    lane = 1
    lon = -1
    obj_id = 6

    cell = (lane, lon)
    assert cell in occupancy_grid.road_occupancy_grid and occupancy_grid.road_occupancy_grid[cell][0].obj_id == obj_id

    # Cars aside ego:
    lane = -1
    lon = 0
    obj_id = 7

    cell = (lane, lon)
    assert cell in occupancy_grid.road_occupancy_grid and occupancy_grid.road_occupancy_grid[cell][0].obj_id == obj_id

    lane = 1
    lon = 0
    obj_id = 8

    cell = (lane, lon)
    assert cell in occupancy_grid.road_occupancy_grid and occupancy_grid.road_occupancy_grid[cell][0].obj_id == obj_id


    # Closest cars in front of ego:
    lane = -1
    lon = 1
    obj_id = 9

    cell = (lane, lon)
    assert cell in occupancy_grid.road_occupancy_grid and occupancy_grid.road_occupancy_grid[cell][0].obj_id == obj_id

    lane = 0
    lon = 1
    obj_id = 10

    cell = (lane, lon)
    assert cell in occupancy_grid.road_occupancy_grid and occupancy_grid.road_occupancy_grid[cell][0].obj_id == obj_id

    lane = 1
    lon = 1
    obj_id = 11

    cell = (lane, lon)
    assert cell in occupancy_grid.road_occupancy_grid and occupancy_grid.road_occupancy_grid[cell][0].obj_id == obj_id
