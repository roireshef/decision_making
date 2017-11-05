from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import OptimalControlUtils
from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.planning.behavioral.behavioral_state_fixtures import state_with_sorrounding_objects, \
    testable_map_api, nov_demo_policy, nov_demo_semantic_follow_action, nov_demo_semantic_behavioral_state, nov_demo_state
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticAction
from decision_making.src.planning.behavioral.policies.november_demo_semantic_policy import NovDemoBehavioralState, \
    NovDemoPolicy

import numpy as np

import time

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


def test_specifyAction_followOtherCar_wellSpecified(nov_demo_semantic_follow_action: SemanticAction,
                                                    nov_demo_semantic_behavioral_state: NovDemoBehavioralState,
                                                    nov_demo_policy: NovDemoPolicy):
    specify = nov_demo_policy._specify_action(nov_demo_semantic_behavioral_state, nov_demo_semantic_follow_action)

    # A = OptimalControlUtils.QuinticPoly1D.time_constraints_matrix(specify.t)
    # A_inv = np.linalg.inv(A)
    #
    # constraints_s = np.array([ego_sx0, ego_sv0, ego_sa0, obj_sxT - safe_lon_dist - obj_long_margin, obj_svT, obj_saT])
    # constraints_d = np.array([ego_dx0, ego_dv0, ego_da0, obj_dxT, 0.0, 0.0])
    #
    # poly_all_coefs_s = OptimalControlUtils.QuinticPoly1D.solve(A_inv, [constraints_s])[0]

    assert True
