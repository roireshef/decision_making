import numpy as np

from decision_making.src.messages.route_plan_message import RoutePlanLaneSegment
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.planning.route.binary_cost_based_route_planner import BinaryCostBasedRoutePlanner
from decision_making.src.planning.route.route_planner import RoutePlannerInputData
from decision_making.test.planning.route.scene_fixtures import RoutePlanTestData, \
    construction_scene_and_expected_output, map_scene_and_expected_output, \
    gmfa_scene_and_expected_output, lane_direction_scene_and_expected_output, \
    combined_scene_and_expected_output
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split, left_lane_split_scene_static,\
    right_lane_split_scene_static, multiple_lane_split_scene_static
from unittest.mock import patch


def test_plan_normalScene_accurateRoutePlanOutput(scene_static_pg_split: SceneStatic):
    # Test Data
    scene_static_base = scene_static_pg_split.s_Data.s_SceneStaticBase
    navigation_plan = scene_static_pg_split.s_Data.s_NavigationPlan

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static_base, nav_plan=navigation_plan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Expected Outputs
    num_lane_segments = [road_segment.e_Cnt_lane_segment_id_count for road_segment in scene_static_base.as_scene_road_segment]

    exp_num_road_segments = navigation_plan.e_Cnt_num_road_segments
    exp_road_segment_ids = navigation_plan.a_i_road_segment_ids
    exp_num_lane_segments = np.array(num_lane_segments)
    exp_route_plan_lane_segments = [[RoutePlanLaneSegment(lane_segment_id, 0., 0.) for lane_segment_id in road_segment.a_i_lane_segment_ids]
                                    for road_segment in scene_static_base.as_scene_road_segment]

    # Assertions
    assert route_plan_output.e_Cnt_num_road_segments == exp_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == exp_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == exp_num_lane_segments.all()
    for i in range(len(exp_route_plan_lane_segments)):
        for j in range(len(exp_route_plan_lane_segments[i])):
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost == exp_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost == exp_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id == exp_route_plan_lane_segments[i][j].e_i_lane_segment_id


def test_plan_constructionScenes_accurateRoutePlanOutput(construction_scene_and_expected_output: RoutePlanTestData):
    # Test Data
    scene_static = construction_scene_and_expected_output.scene_static
    expected_output = construction_scene_and_expected_output.expected_output

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    print(route_plan_output)

    # Assertions
    assert route_plan_output.e_Cnt_num_road_segments == expected_output.e_Cnt_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == expected_output.a_i_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == expected_output.a_Cnt_num_lane_segments.all()

    for i, road_segment in enumerate(route_plan_output.as_route_plan_lane_segments):
        for j, lane_segment in enumerate(road_segment):
            # print("lane_segment_id     = ", lane_segment.e_i_lane_segment_id)
            # print("lane_occupancy_cost = ", lane_segment.e_cst_lane_occupancy_cost)
            # print("lane_end_cost       = ", lane_segment.e_cst_lane_end_cost, "\n")

            expected_laneseg_id = expected_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id
            expected_lane_occupancy_cost = expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            expected_lane_end_cost = expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert (lane_segment.e_i_lane_segment_id == expected_laneseg_id),\
                "output lane_segment_id:" + str(lane_segment.e_i_lane_segment_id) + "  expected lane_segment_id:" + str(expected_laneseg_id)
            assert(lane_segment.e_cst_lane_occupancy_cost == expected_lane_occupancy_cost),\
                "lane_segment_id:" + str(lane_segment.e_i_lane_segment_id) + \
                "   output lane_occ_cost:" + str(lane_segment.e_cst_lane_occupancy_cost) + "  expected lane_occ_cost:" +\
                str(expected_lane_occupancy_cost)
            assert(lane_segment.e_cst_lane_end_cost == expected_lane_end_cost),\
                "lane_segment_id:" + str(lane_segment.e_i_lane_segment_id) + \
                "   output lane_end_cost:" + str(lane_segment.e_cst_lane_end_cost) + "  expected lane_end_cost:" +\
                str(expected_lane_end_cost)

        # print("==========================\n")


def test_plan_mapScenes_accurateRoutePlanOutput(map_scene_and_expected_output: RoutePlanTestData):
    # Test Data
    scene_static = map_scene_and_expected_output.scene_static
    expected_output = map_scene_and_expected_output.expected_output

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
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


def test_plan_gmfaScenes_accurateRoutePlanOutput(gmfa_scene_and_expected_output: RoutePlanTestData):
    # Test Data
    scene_static = gmfa_scene_and_expected_output.scene_static
    expected_output = gmfa_scene_and_expected_output.expected_output

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
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


def test_plan_laneDirectionScenes_accurateRoutePlanOutput(lane_direction_scene_and_expected_output: RoutePlanTestData):
    # Test Data
    scene_static = lane_direction_scene_and_expected_output.scene_static
    expected_output = lane_direction_scene_and_expected_output.expected_output

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
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


def test_plan_combinedScenes_accurateRoutePlanOutput(combined_scene_and_expected_output: RoutePlanTestData):
    # Test Data
    scene_static = combined_scene_and_expected_output.scene_static
    expected_output = combined_scene_and_expected_output.expected_output

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
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


@patch('decision_making.src.planning.route.binary_cost_based_route_planner.TAKE_SPLIT', True)
def test_plan_laneSplitOnLeft_routePlanTakeSplit(left_lane_split_scene_static: SceneStatic):
    # Test Data
    scene_static_base = left_lane_split_scene_static.s_Data.s_SceneStaticBase
    navigation_plan = left_lane_split_scene_static.s_Data.s_NavigationPlan

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static_base, nav_plan=navigation_plan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Expected Outputs
    num_lane_segments = [road_segment.e_Cnt_lane_segment_id_count for road_segment in scene_static_base.as_scene_road_segment]

    exp_num_road_segments = navigation_plan.e_Cnt_num_road_segments
    exp_road_segment_ids = navigation_plan.a_i_road_segment_ids
    exp_num_lane_segments = np.array(num_lane_segments)
    exp_route_plan_lane_segments = [[RoutePlanLaneSegment(lane_segment_id, 0., 0.) for lane_segment_id in road_segment.a_i_lane_segment_ids]
                                    for road_segment in scene_static_base.as_scene_road_segment]

    exp_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0

    # Assertions
    assert route_plan_output.e_Cnt_num_road_segments == exp_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == exp_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == exp_num_lane_segments.all()
    for i in range(len(exp_route_plan_lane_segments)):
        for j in range(len(exp_route_plan_lane_segments[i])):
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id ==\
                exp_route_plan_lane_segments[i][j].e_i_lane_segment_id


@patch('decision_making.src.planning.route.binary_cost_based_route_planner.TAKE_SPLIT', False)
def test_plan_laneSplitOnLeft_routePlanStayStraight(left_lane_split_scene_static: SceneStatic):
    # Test Data
    scene_static_base = left_lane_split_scene_static.s_Data.s_SceneStaticBase
    navigation_plan = left_lane_split_scene_static.s_Data.s_NavigationPlan

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static_base, nav_plan=navigation_plan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Expected Outputs
    num_lane_segments = [road_segment.e_Cnt_lane_segment_id_count for road_segment in scene_static_base.as_scene_road_segment]

    exp_num_road_segments = navigation_plan.e_Cnt_num_road_segments
    exp_road_segment_ids = navigation_plan.a_i_road_segment_ids
    exp_num_lane_segments = np.array(num_lane_segments)
    exp_route_plan_lane_segments = [[RoutePlanLaneSegment(lane_segment_id, 0., 0.) for lane_segment_id in road_segment.a_i_lane_segment_ids]
                                    for road_segment in scene_static_base.as_scene_road_segment]

    exp_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0

    # Assertions
    assert route_plan_output.e_Cnt_num_road_segments == exp_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == exp_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == exp_num_lane_segments.all()
    for i in range(len(exp_route_plan_lane_segments)):
        for j in range(len(exp_route_plan_lane_segments[i])):
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id ==\
                exp_route_plan_lane_segments[i][j].e_i_lane_segment_id


@patch('decision_making.src.planning.route.binary_cost_based_route_planner.TAKE_SPLIT', True)
def test_plan_laneSplitOnRight_routePlanTakeSplit(right_lane_split_scene_static: SceneStatic):
    # Test Data
    scene_static_base = right_lane_split_scene_static.s_Data.s_SceneStaticBase
    navigation_plan = right_lane_split_scene_static.s_Data.s_NavigationPlan

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static_base, nav_plan=navigation_plan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Expected Outputs
    num_lane_segments = [road_segment.e_Cnt_lane_segment_id_count for road_segment in scene_static_base.as_scene_road_segment]

    exp_num_road_segments = navigation_plan.e_Cnt_num_road_segments
    exp_road_segment_ids = navigation_plan.a_i_road_segment_ids
    exp_num_lane_segments = np.array(num_lane_segments)
    exp_route_plan_lane_segments = [[RoutePlanLaneSegment(lane_segment_id, 0., 0.) for lane_segment_id in road_segment.a_i_lane_segment_ids]
                                    for road_segment in scene_static_base.as_scene_road_segment]

    exp_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0

    # Assertions
    assert route_plan_output.e_Cnt_num_road_segments == exp_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == exp_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == exp_num_lane_segments.all()
    for i in range(len(exp_route_plan_lane_segments)):
        for j in range(len(exp_route_plan_lane_segments[i])):
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id ==\
                exp_route_plan_lane_segments[i][j].e_i_lane_segment_id


@patch('decision_making.src.planning.route.binary_cost_based_route_planner.TAKE_SPLIT', False)
def test_plan_laneSplitOnRight_routePlanStayStraight(right_lane_split_scene_static: SceneStatic):
    # Test Data
    scene_static_base = right_lane_split_scene_static.s_Data.s_SceneStaticBase
    navigation_plan = right_lane_split_scene_static.s_Data.s_NavigationPlan

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static_base, nav_plan=navigation_plan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Expected Outputs
    num_lane_segments = [road_segment.e_Cnt_lane_segment_id_count for road_segment in scene_static_base.as_scene_road_segment]

    exp_num_road_segments = navigation_plan.e_Cnt_num_road_segments
    exp_road_segment_ids = navigation_plan.a_i_road_segment_ids
    exp_num_lane_segments = np.array(num_lane_segments)
    exp_route_plan_lane_segments = [[RoutePlanLaneSegment(lane_segment_id, 0., 0.) for lane_segment_id in road_segment.a_i_lane_segment_ids]
                                    for road_segment in scene_static_base.as_scene_road_segment]

    exp_route_plan_lane_segments[1][0].e_cst_lane_end_cost = 1.0

    # Assertions
    assert route_plan_output.e_Cnt_num_road_segments == exp_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == exp_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == exp_num_lane_segments.all()
    for i in range(len(exp_route_plan_lane_segments)):
        for j in range(len(exp_route_plan_lane_segments[i])):
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id ==\
                exp_route_plan_lane_segments[i][j].e_i_lane_segment_id


@patch('decision_making.src.planning.route.binary_cost_based_route_planner.TAKE_SPLIT', True)
@patch('decision_making.src.planning.route.binary_cost_based_route_planner.PRIORITIZE_RIGHT_SPLIT_OVER_LEFT_SPLIT', True)
def test_plan_laneSplitOnLeftAndRight_routePlanTakeRightSplit(multiple_lane_split_scene_static: SceneStatic):
    # Test Data
    scene_static_base = multiple_lane_split_scene_static.s_Data.s_SceneStaticBase
    navigation_plan = multiple_lane_split_scene_static.s_Data.s_NavigationPlan

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static_base, nav_plan=navigation_plan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Expected Outputs
    num_lane_segments = [road_segment.e_Cnt_lane_segment_id_count for road_segment in scene_static_base.as_scene_road_segment]

    exp_num_road_segments = navigation_plan.e_Cnt_num_road_segments
    exp_road_segment_ids = navigation_plan.a_i_road_segment_ids
    exp_num_lane_segments = np.array(num_lane_segments)
    exp_route_plan_lane_segments = [[RoutePlanLaneSegment(lane_segment_id, 0., 0.) for lane_segment_id in road_segment.a_i_lane_segment_ids]
                                    for road_segment in scene_static_base.as_scene_road_segment]

    exp_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0
    exp_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0

    # Assertions
    assert route_plan_output.e_Cnt_num_road_segments == exp_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == exp_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == exp_num_lane_segments.all()
    for i in range(len(exp_route_plan_lane_segments)):
        for j in range(len(exp_route_plan_lane_segments[i])):
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id ==\
                exp_route_plan_lane_segments[i][j].e_i_lane_segment_id


@patch('decision_making.src.planning.route.binary_cost_based_route_planner.TAKE_SPLIT', True)
@patch('decision_making.src.planning.route.binary_cost_based_route_planner.PRIORITIZE_RIGHT_SPLIT_OVER_LEFT_SPLIT', False)
def test_plan_laneSplitOnLeftAndRight_routePlanTakeLeftSplit(multiple_lane_split_scene_static: SceneStatic):
    # Test Data
    scene_static_base = multiple_lane_split_scene_static.s_Data.s_SceneStaticBase
    navigation_plan = multiple_lane_split_scene_static.s_Data.s_NavigationPlan

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static_base, nav_plan=navigation_plan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Expected Outputs
    num_lane_segments = [road_segment.e_Cnt_lane_segment_id_count for road_segment in scene_static_base.as_scene_road_segment]

    exp_num_road_segments = navigation_plan.e_Cnt_num_road_segments
    exp_road_segment_ids = navigation_plan.a_i_road_segment_ids
    exp_num_lane_segments = np.array(num_lane_segments)
    exp_route_plan_lane_segments = [[RoutePlanLaneSegment(lane_segment_id, 0., 0.) for lane_segment_id in road_segment.a_i_lane_segment_ids]
                                    for road_segment in scene_static_base.as_scene_road_segment]

    exp_route_plan_lane_segments[1][0].e_cst_lane_end_cost = 1.0
    exp_route_plan_lane_segments[1][1].e_cst_lane_end_cost = 1.0

    # Assertions
    assert route_plan_output.e_Cnt_num_road_segments == exp_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == exp_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == exp_num_lane_segments.all()
    for i in range(len(exp_route_plan_lane_segments)):
        for j in range(len(exp_route_plan_lane_segments[i])):
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id ==\
                exp_route_plan_lane_segments[i][j].e_i_lane_segment_id


@patch('decision_making.src.planning.route.binary_cost_based_route_planner.TAKE_SPLIT', False)
def test_plan_laneSplitOnLeftAndRight_routePlanStayStraight(multiple_lane_split_scene_static: SceneStatic):
    # Test Data
    scene_static_base = multiple_lane_split_scene_static.s_Data.s_SceneStaticBase
    navigation_plan = multiple_lane_split_scene_static.s_Data.s_NavigationPlan

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static_base, nav_plan=navigation_plan)
    route_plan_obj = BinaryCostBasedRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Expected Outputs
    num_lane_segments = [road_segment.e_Cnt_lane_segment_id_count for road_segment in scene_static_base.as_scene_road_segment]

    exp_num_road_segments = navigation_plan.e_Cnt_num_road_segments
    exp_road_segment_ids = navigation_plan.a_i_road_segment_ids
    exp_num_lane_segments = np.array(num_lane_segments)
    exp_route_plan_lane_segments = [[RoutePlanLaneSegment(lane_segment_id, 0., 0.) for lane_segment_id in road_segment.a_i_lane_segment_ids]
                                    for road_segment in scene_static_base.as_scene_road_segment]

    exp_route_plan_lane_segments[1][0].e_cst_lane_end_cost = 1.0
    exp_route_plan_lane_segments[1][2].e_cst_lane_end_cost = 1.0

    # Assertions
    assert route_plan_output.e_Cnt_num_road_segments == exp_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == exp_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == exp_num_lane_segments.all()
    for i in range(len(exp_route_plan_lane_segments)):
        for j in range(len(exp_route_plan_lane_segments[i])):
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_end_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost ==\
                exp_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost
            assert route_plan_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id ==\
                exp_route_plan_lane_segments[i][j].e_i_lane_segment_id
