from decision_making.src.planning.behavioral.policies.november_demo_semantic_policy import NovDemoBehavioralState
from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.planning.behavioral.behavioral_state_fixtures import state_with_sorrounding_objects, testable_map_api


def test_generateSemanticOccupancyGrid_ComplexStateWithFullGrid_carsAreInRightCells(state_with_sorrounding_objects,
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

    # Closest cars behind ego: (cars 1-3 are ignored because they are far)
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


    # Closest cars in front of ego: (cars 12-14 are ignored because they are far)
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

def test_specifyAction_followOtherCar_wellSpecified(state_with_sorrounding_objects):
    state_with_sorrounding_objects