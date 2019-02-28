import numpy as numpy
import pytest

from typing import List

from decision_making.src.messages.route_plan_message import RoutePlan, RoutePlanLaneSegment, DataRoutePlan
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.state.state import EgoState
from decision_making.test.messages.static_scene_fixture import scene_static
from decision_making.test.planning.behavioral.behavioral_state_fixtures import ego_state_for_takover_message_default_scene
from decision_making.test.planning.route.scene_fixtures import ( 
    TakeOverTestData , 
    construction_scene_for_takeover_test, 
    default_route_plan )
from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher

def test_setTakeoverMessage_defaultScene_noTakeoverFlag(scene_static: SceneStatic, ego_state_for_takover_message_default_scene: EgoState):
    # Route Plan Data
    route_plan_data = default_route_plan()

    takeover_msg = BehavioralPlanningFacade.set_takeover_message(route_plan_data=route_plan_data,
                                                                 ego_state=ego_state_for_takover_message_default_scene,
                                                                 scene_static=scene_static)

    assert takeover_msg.s_Data.e_b_is_takeover_needed == False

def test_setTakeoverMessage_blockedScene_takeoverFlag(construction_scene_for_takeover_test: TakeOverTestData):
    scene_static = construction_scene_for_takeover_test.scene_static
    route_plan_data = construction_scene_for_takeover_test.route_plan_data
    ego_state = construction_scene_for_takeover_test.ego_state
    expected_takeover = construction_scene_for_takeover_test.expected_takeover

    takeover_msg = BehavioralPlanningFacade.set_takeover_message(route_plan_data=route_plan_data,
                                                                 ego_state=ego_state,
                                                                 scene_static=scene_static)

    assert takeover_msg.s_Data.e_b_is_takeover_needed == expected_takeover
