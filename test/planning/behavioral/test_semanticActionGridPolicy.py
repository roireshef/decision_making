from unittest.mock import patch

from decision_making.src.planning.behavioral.constants import SAFE_DIST_TIME_DELAY, EGO_ORIGIN_LON_FROM_REAR
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_policy import SemanticActionsGridPolicy
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_state import \
    SemanticActionsGridState
from decision_making.src.planning.behavioral.policies.semantic_actions_policy import SemanticAction
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.planning.behavioral.behavioral_state_fixtures import semantic_actions_state, \
    semantic_follow_action, semantic_state, semantic_grid_policy, state_with_sorrounding_objects, \
    state_with_ego_on_right_lane, state_with_ego_on_left_lane
from mapping.test.model.testable_map_fixtures import map_api_mock, navigation_fixture, testable_map_api
from rte.python.logger.AV_logger import AV_Logger


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_enumerate_actions_egoAtRoadEdge_filterOnlyValidActions(state_with_sorrounding_objects,
                                                                state_with_ego_on_right_lane,
                                                                state_with_ego_on_left_lane):
    logger = AV_Logger.get_logger('Nov demo - semantic occupancy grid')
    predictor = RoadFollowingPredictor(logger)

    policy = SemanticActionsGridPolicy(logger=logger, predictor=predictor)

    # Check that when car is on right lane we get only 2 valid actions
    state = state_with_ego_on_right_lane
    behavioral_state = SemanticActionsGridState.create_from_state(state=state, logger=logger)
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
    behavioral_state = SemanticActionsGridState.create_from_state(state=state, logger=logger)
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


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_enumerate_actions_gridFull_allActionsEnumerated(state_with_sorrounding_objects):
    logger = AV_Logger.get_logger('Nov demo - semantic occupancy grid')
    state = state_with_sorrounding_objects
    predictor = RoadFollowingPredictor(logger)

    policy = SemanticActionsGridPolicy(logger=logger, predictor=predictor)

    behavioral_state = SemanticActionsGridState.create_from_state(state=state, logger=logger)
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


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_generateSemanticOccupancyGrid_ComplexStateWithFullGrid_carsAreInRightCells(state_with_sorrounding_objects):
    """
    Here we generate a state with ego and dynamic obejcts, and verify that
    each object is mapped to the right cell in the grid.
    The implementation assigns only the closest cars in each cell, as detailed in the
    class documentation, therefore we expect to see only the relevant cars assigned.
    :return:
    """
    logger = AV_Logger.get_logger('Semantic occupancy grid')

    state = state_with_sorrounding_objects
    occupancy_grid = SemanticActionsGridState.create_from_state(state=state, logger=logger)

    # Assertion tests of objects in grid:

    lane = -1
    lon = -1
    obj_id = 1

    cell = (lane, lon)
    assert cell in occupancy_grid.road_occupancy_grid and occupancy_grid.road_occupancy_grid[cell][0].obj_id == obj_id

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

@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_specifyAction_followOtherCar_wellSpecified(semantic_follow_action: SemanticAction,
                                                    semantic_actions_state: SemanticActionsGridState,
                                                    semantic_grid_policy: SemanticActionsGridPolicy,
                                                    navigation_fixture):

    specify = semantic_grid_policy._specify_action(semantic_actions_state, semantic_follow_action,
                                                   navigation_fixture)

    # A = OptimalControlUtils.QuinticPoly1D.time_constraints_matrix(specify.t)
    # A_inv = np.linalg.inv(A)
    #
    # constraints_s = np.array([ego_sx0, ego_sv0, ego_sa0, obj_sxT - safe_lon_dist - obj_long_margin, obj_svT, obj_saT])
    # constraints_d = np.array([ego_dx0, ego_dv0, ego_da0, obj_dxT, 0.0, 0.0])
    #
    # poly_all_coefs_s = OptimalControlUtils.QuinticPoly1D.solve(A_inv, [constraints_s])[0]

    ego_on_road = semantic_actions_state.ego_state.road_localization
    ego_s0 = ego_on_road.road_lon

    obj = semantic_follow_action.target_obj
    obj_on_road = obj.road_localization
    obj_s0 = obj_on_road.road_lon
    obj_v = obj.road_longitudinal_speed
    obj_sT = obj_s0 + specify.t * obj_v
    lon_margin = semantic_actions_state.ego_state.size.length - EGO_ORIGIN_LON_FROM_REAR + obj.size.length / 2

    assert specify.v == obj_v
    assert specify.s_rel + ego_s0 == obj_sT - lon_margin - SAFE_DIST_TIME_DELAY * obj_v
