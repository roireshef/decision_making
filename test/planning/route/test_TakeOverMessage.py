from decision_making.src.infra.pubsub import PubSub
from rte.python.logger.AV_logger import AV_Logger
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.test.planning.behavioral.mock_behavioral_facade import BehavioralFacadeMock

from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.state.state import EgoState
from decision_making.test.planning.route.scene_fixtures import TakeOverTestData, \
    default_route_plan_for_PG_split_file, construction_scene_for_takeover_test
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split
from decision_making.test.planning.behavioral.behavioral_state_fixtures import \
    ego_state_for_takover_message_default_scene


def test_setTakeoverMessage_defaultScene_noTakeoverFlag(scene_static_pg_split: SceneStatic,
                                                        ego_state_for_takover_message_default_scene: EgoState):

    # Route Plan Data
    route_plan_data = default_route_plan_for_PG_split_file()

    behavior_facade_mock = BehavioralFacadeMock(pubsub=PubSub(), logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING),
                                                trajectory_params=None,
                                                visualization_msg=None, trigger_pos=None)

    takeover_msg = behavior_facade_mock._mock_takeover_message(route_plan_data=route_plan_data,
                                                               ego_state=ego_state_for_takover_message_default_scene,
                                                               scene_static=scene_static_pg_split)

    assert takeover_msg.s_Data.e_b_is_takeover_needed == False


def test_setTakeoverMessage_blockedScene_takeoverFlag(construction_scene_for_takeover_test: TakeOverTestData):
    scene_static = construction_scene_for_takeover_test.scene_static
    route_plan_data = construction_scene_for_takeover_test.route_plan_data
    ego_state = construction_scene_for_takeover_test.ego_state
    expected_takeover = construction_scene_for_takeover_test.expected_takeover


    behavior_facade_mock = BehavioralFacadeMock(pubsub=PubSub(), logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING),
                                                trajectory_params=None,
                                                visualization_msg=None, trigger_pos=None)

    takeover_msg = behavior_facade_mock._mock_takeover_message(route_plan_data=route_plan_data,
                                                               ego_state=ego_state,
                                                               scene_static=scene_static)

    assert takeover_msg.s_Data.e_b_is_takeover_needed == expected_takeover
