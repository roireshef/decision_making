import numpy as np
import pprint
from typing import List

import pytest
from logging import Logger
from decision_making.src.infra.pubsub import PubSub

from decision_making.src.messages.route_plan_message import RoutePlan,RoutePlanLaneSegment, DataRoutePlan

from common_data.interface.Rte_Types.python.sub_structures import TsSYSRoutePlanLaneSegment, TsSYSDataRoutePlan
     
from decision_making.src.planning.route.route_planner import RoutePlanner, RoutePlannerInputData, DataRoutePlan
from decision_making.src.planning.route.cost_based_route_planner import CostBasedRoutePlanner
from decision_making.src.messages.scene_static_enums import RoutePlanLaneSegmentAttr, LaneMappingStatusType, MapLaneDirection, \
     GMAuthorityType, LaneConstructionType

from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher

from rte.python.logger.AV_logger import AV_Logger


def test_plan_twoRoadSegments_routePlanOutput(): 

    logger = AV_Logger.get_logger("")

    pubsub = PubSub()
    
    scene_static_obj = SceneStaticPublisher(pubsub = pubsub , logger = logger)

    scene_static_data = scene_static_obj._generate_data()

    scene_static_base = scene_static_data.s_Data.s_SceneStaticBase

    navigation_plan = scene_static_data.s_Data.s_NavigationPlan

    route_planner_input = RoutePlannerInputData(Scene=scene_static_base,Nav=navigation_plan)

    route_plan_obj = CostBasedRoutePlanner()
    
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # expected outputs:

    exp_num_road_segments = 2

    exp_road_segment_ids = np.array([1, 2])

    exp_num_lane_segments = np.array([2, 2])

    road_segment_1 = [RoutePlanLaneSegment(101,0,0) , RoutePlanLaneSegment(102,0,0) ]
    road_segment_2 = [RoutePlanLaneSegment(201,0,0) , RoutePlanLaneSegment(202,0,0) ]
    exp_route_plan_lane_segments = [road_segment_1 , road_segment_2]

    # assertion
    
    assert route_plan_output.e_Cnt_num_road_segments == exp_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == exp_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == exp_num_lane_segments.all()
    for i in range(len(exp_route_plan_lane_segments)) :
        for j in range(len(exp_route_plan_lane_segments[i])) :
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost == exp_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost == exp_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id == exp_route_plan_lane_segments[i][j].e_i_lane_segment_id
