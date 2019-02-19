import numpy as np
import pprint
from typing import List

import pytest

from decision_making.src.messages.route_plan_message import RoutePlan, RoutePlanLaneSegment, DataRoutePlan
     
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData

from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher

from decision_making.src.messages.scene_static_message import SceneStatic

from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.state.state import EgoState
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.test.planning.behavioral.behavioral_state_fixtures import ( 
    ego_state_for_takover_message_simple_scene, 
    ego_state_for_takover_message_default_scene )
from decision_making.test.messages.static_scene_fixture import scene_static
from decision_making.test.planning.route.scene_fixtures import ( 
    RoutePlanTestData , 
    construction_scene_and_expected_output, 
    default_route_plan , 
    blocked_scene_and_expected_output)


def test_setTakeoverMessage_defaultScene_noTakeoverFlag(scene_static: SceneStatic , ego_state_for_takover_message_default_scene: EgoState):
    
    # route plan data
    route_plan_data = default_route_plan()

    takeover_msg = BehavioralPlanningFacade.set_takeover_message(route_plan_data = route_plan_data, ego_state= ego_state_for_takover_message_default_scene, \
                                                                 scene_static = scene_static)

    assert takeover_msg.s_Data.e_b_is_takeover_needed == False


def test_setTakeoverMessage_updatedScene_noTakeoverFlag(construction_scene_and_expected_output: RoutePlanTestData, \
                                                        ego_state_for_takover_message_default_scene: EgoState):
    
    scene_static = construction_scene_and_expected_output.scene_static
    route_plan_data = construction_scene_and_expected_output.expected_output

    takeover_msg = BehavioralPlanningFacade.set_takeover_message(route_plan_data = route_plan_data, ego_state= ego_state_for_takover_message_default_scene, \
                                                                 scene_static = scene_static)

    assert takeover_msg.s_Data.e_b_is_takeover_needed == False


def test_setTakeoverMessage_blockedScene_trueTakeoverFlag(blocked_scene_and_expected_output: RoutePlanTestData, \
                                                    ego_state_for_takover_message_default_scene: EgoState):

    scene_static = blocked_scene_and_expected_output.scene_static
    route_plan_data = blocked_scene_and_expected_output.expected_output

    takeover_msg = BehavioralPlanningFacade.set_takeover_message(route_plan_data = route_plan_data, ego_state= ego_state_for_takover_message_default_scene, \
                                                                    scene_static = scene_static)

    assert takeover_msg.s_Data.e_b_is_takeover_needed == True



