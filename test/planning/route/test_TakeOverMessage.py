import numpy as np
import pprint
from typing import List

import pytest
from logging import Logger


from decision_making.src.infra.pubsub import PubSub
from rte.python.logger.AV_logger import AV_Logger

from decision_making.src.messages.route_plan_message import RoutePlan, RoutePlanLaneSegment, DataRoutePlan
     
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData

from decision_making.test.messages.static_scene_fixture import scene_static
from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher

from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.state.state import State
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.test.planning.behavioral.behavioral_state_fixtures import state_with_no_sorrounding_objects_for_takover_message


def test_setTakeoverMessage_simpleScene_takeoverFlag(state_with_no_sorrounding_objects_for_takover_message:State):
    
    # route plan input
    num_road_segments = 2

    road_segment_ids = np.array([1, 2])

    num_lane_segments = np.array([2, 2])

    road_segment_1 = [RoutePlanLaneSegment(101,0,0) , RoutePlanLaneSegment(102,0,0) ]
    road_segment_2 = [RoutePlanLaneSegment(201,0,0) , RoutePlanLaneSegment(202,0,0) ]
    route_plan_lane_segments = [road_segment_1 , road_segment_2]

    route_plan_data = DataRoutePlan(True, num_road_segments, road_segment_ids, num_lane_segments,route_plan_lane_segments)

    # setting the scene static for MApUtil functions

    logger = AV_Logger.get_logger("")
    pubsub = PubSub()
    
    scene_static_obj = SceneStaticPublisher(pubsub = pubsub , logger = logger)
    scene_static_data = scene_static_obj._generate_data()

    takeover_msg = BehavioralPlanningFacade.set_takeover_message(route_plan_data = route_plan_data, state= state_with_no_sorrounding_objects_for_takover_message, \
                                                                 scene_static =scene_static_data)

    assert takeover_msg.s_Data.e_b_is_takeover_needed == False




