import numpy as np
import pprint
from typing import List

import pytest
from logging import Logger

from decision_making.src.messages.route_plan_message import RoutePlan, RoutePlanLaneSegment, DataRoutePlan
     
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData

from decision_making.test.messages.static_scene_fixture import scene_static

from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.state.state import State
# form decision_making.test.messages.planning.behavioral.behavioral_state_fixtures import 


def test_setTakeoverMessage_simpleScene_takeoverFlag():

    # route plan input
    num_road_segments = 2

    road_segment_ids = np.array([1, 2])

    num_lane_segments = np.array([2, 2])

    road_segment_1 = [RoutePlanLaneSegment(101,0,0) , RoutePlanLaneSegment(102,0,0) ]
    road_segment_2 = [RoutePlanLaneSegment(201,0,0) , RoutePlanLaneSegment(202,0,0) ]
    route_plan_lane_segments = [road_segment_1 , road_segment_2]

    route_plan = DataRoutePlan(True, num_road_segments, road_segment_ids, num_lane_segments,route_plan_lane_segments)

    #TODO: come up with a way to generate state

    takeover_msg = BehavioralPlanningFacade.set_takeover_message(route_plan = route_plan, state= None)




