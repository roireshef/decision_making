from decision_making.src.planning.behavioral.policies.november_demo_semantic_policy import NovDemoPolicy, \
    NovDemoBehavioralState
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from mapping.test.model.testable_map_fixtures import testable_map_api
from decision_making.test.planning.behavioral.nov_demo.test_NovDemoBehavioralState import state_with_sorrounding_objects
from rte.python.logger.AV_logger import AV_Logger


def test_enumerate_actions_gridFull_allActionsEnumerated(state_with_sorrounding_objects, testable_map_api):

    logger = AV_Logger.get_logger('Nov demo - semantic occupancy grid')
    map_api = testable_map_api
    state = state_with_sorrounding_objects
    predictor = RoadFollowingPredictor(map_api=map_api)

    policy = NovDemoPolicy(logger=logger, policy_config=None, predictor=predictor, map_api=map_api)

    behavioral_state = NovDemoBehavioralState.create_from_state(state=state, map_api=map_api, logger=logger)
    actions = policy._enumerate_actions(behavioral_state=behavioral_state)

    action_index = 0
    lane = -1
    lon = 1
    obj_id = 9

    cell = (lane, lon)
    assert actions[action_index].cell == cell and actions[action_index].target_obj is not None\
           and actions[action_index].target_obj.obj_id == obj_id

    action_index = 1
    lane = 0
    lon = 1
    obj_id = 10

    cell = (lane, lon)
    assert actions[action_index].cell == cell and actions[action_index].target_obj is not None\
           and actions[action_index].target_obj.obj_id == obj_id

    action_index = 2
    lane = 1
    lon = 1
    obj_id = 11

    cell = (lane, lon)
    assert actions[action_index].cell == cell and actions[action_index].target_obj is not None\
           and actions[action_index].target_obj.obj_id == obj_id

