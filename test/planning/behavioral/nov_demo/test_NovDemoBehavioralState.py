import numpy as np
from typing import List
import pytest

from decision_making.src.planning.behavioral.policies.november_demo_semantic_policy import NovDemoBehavioralState
from decision_making.src.planning.behavioral.semantic_actions_policy import RoadSemanticOccupancyGrid
from decision_making.src.state.state import RoadLocalization, EgoState, ObjectSize, \
    DynamicObject, OccupancyState, State
from rte.python.logger.AV_logger import AV_Logger
from mapping.test.model.testable_map_fixtures import testable_map_api


@pytest.fixture(scope='function')
def state_with_sorrounding_objects():
    # Stub of occupancy grid
    occupancy_state = OccupancyState(0, np.array([]), np.array([]))

    # ego state at (0.0, 0.0, 0.0)
    road_localization = RoadLocalization(road_id=1, lane_num=0, full_lat=0.0, intra_lane_lat=0.0, road_lon=0.0,
                                         intra_lane_yaw=0.0)
    ego_state = EgoState(obj_id=0, timestamp=0, x=0.0, y=0.0, z=0.0, yaw=0.0,
                         size=ObjectSize(length=2.5, width=1.5, height=1.0),
                         confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0, road_localization=road_localization)


    # Generate objects with at the following locations:
    dynamic_objects : List[DynamicObject] = list()
    for obj_id in range(10):
        road_localization = RoadLocalization(road_id=1, lane_num=0, full_lat=1.5, intra_lane_lat=1.5, road_lon=10.0,
                                             intra_lane_yaw=0.0)
        dynamic_object = DynamicObject(obj_id=obj_id, timestamp=0, x=10.0, y=1.0, z=0.0, yaw=0.0,
                                           size=ObjectSize(length=2.5, width=1.5, height=1.0),
                                           confidence=1.0, v_x=0.0, v_y=0.0,
                                           acceleration_lon=0.0, omega_yaw=0.0, road_localization=road_localization)

        dynamic_objects.append(dynamic_object)

    yield State(occupancy_state=occupancy_state, dynamic_objects=dynamic_objects, ego_state=ego_state)


def test_generate_semantic_occupancy_grid_ComplexStateWithFullGrid_carsAreInRightCells(state_with_sorrounding_objects, tes):
    """
    Here we generate a state with ego and dynamic obejcts, and verify that
    each object is mapped to the right cell in the grid.
    The implementation assigns only the closest cars in each cell, as detailed in the
    class documentation, therefore we expect to see only the relevant cars assigned.
    :return:
    """
    logger = AV_Logger.get_logger('Nov demo - semantic occupancy grid')

    state = state_with_sorrounding_objects
    road_occupancy_grid = NovDemoBehavioralState.create_from_state(state=state, map_api=testable_map_api, logger=logger)

    # TODO: write assert conditions
    assert True

