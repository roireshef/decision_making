import numpy as np
import pprint
from typing import List

import pytest
from logging import Logger
from decision_making.src.infra.pubsub import PubSub
from rte.python.logger.AV_logger import AV_Logger

from decision_making.src.messages.route_plan_message import RoutePlanLaneSegment
from decision_making.src.messages.scene_static_message import (
    SceneStatic,
    SceneStaticBase,
    NavigationPlan)
from decision_making.src.planning.route.route_planner import RoutePlannerInputData
from decision_making.src.planning.route.cost_based_route_planner import CostBasedRoutePlanner
from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher
from decision_making.test.messages.static_scene_fixture import scene_static
from decision_making.test.planning.route.scene_fixtures import (
    RoutePlanTestData,
    construction_scene_and_expected_output)


def test_plan_simpleScene_routePlanOutput():

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

def test_plan_normalScene_accurateRoutePlanOutput(scene_static: SceneStatic):

    scene_static_base = scene_static.s_Data.s_SceneStaticBase

    navigation_plan = scene_static.s_Data.s_NavigationPlan

    route_planner_input = RoutePlannerInputData(Scene=scene_static_base,Nav=navigation_plan)

    route_plan_obj = CostBasedRoutePlanner()

    route_plan_output = route_plan_obj.plan(route_planner_input)

    # expected outputs:
    num_lane_segments = [road_segment.e_Cnt_lane_segment_id_count for road_segment in scene_static_base.as_scene_road_segment]

    exp_num_road_segments = navigation_plan.e_Cnt_num_road_segments
    exp_road_segment_ids = navigation_plan.a_i_road_segment_ids
    exp_num_lane_segments = np.array(num_lane_segments)
    exp_route_plan_lane_segments = [[RoutePlanLaneSegment(lane_segment_id, 0., 0.) for lane_segment_id in road_segment.a_i_lane_segment_ids]
                                    for road_segment in scene_static_base.as_scene_road_segment]

    # assertion
    assert route_plan_output.e_Cnt_num_road_segments == exp_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == exp_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == exp_num_lane_segments.all()
    for i in range(len(exp_route_plan_lane_segments)) :
        for j in range(len(exp_route_plan_lane_segments[i])) :
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost == exp_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost == exp_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id == exp_route_plan_lane_segments[i][j].e_i_lane_segment_id

def test_plan_constructionScenes_accurateRoutePlanOutput(construction_scene_and_expected_output: RoutePlanTestData):
    # Test Data
    scene_static = construction_scene_and_expected_output.scene_static
    expected_output = construction_scene_and_expected_output.expected_output

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData(Scene=scene_static.s_Data.s_SceneStaticBase,
                                                Nav=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = CostBasedRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Assertions
    assert route_plan_output.e_Cnt_num_road_segments == expected_output.e_Cnt_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == expected_output.a_i_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == expected_output.a_Cnt_num_lane_segments.all()

    for i, road_segment in enumerate(route_plan_output.as_route_plan_lane_segments):
        for j, lane_segment in enumerate(road_segment):
            # print("lane_segment_id     = ", lane_segment.e_i_lane_segment_id)
            # print("lane_occupancy_cost = ", lane_segment.e_cst_lane_occupancy_cost)
            # print("lane_end_cost       = ", lane_segment.e_cst_lane_end_cost, "\n")

            assert lane_segment.e_i_lane_segment_id == expected_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id
            assert lane_segment.e_cst_lane_occupancy_cost == expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert lane_segment.e_cst_lane_end_cost == expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost
        
        # print("==========================\n")
