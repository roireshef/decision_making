import numpy as np
from unittest.mock import patch

from decision_making.src.messages.route_plan_message import RoutePlanLaneSegment
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.planning.route.backpropagating_route_planner import BackpropagatingRoutePlanner
from decision_making.src.planning.route.cost_based_route_planner import RoutePlannerInputData
from decision_making.src.scene.scene_static_model import SceneStaticModel

from decision_making.test.planning.route.scene_fixtures import RoutePlanTestData, \
    construction_scene_and_expected_output, map_scene_and_expected_output, \
    gmfa_scene_and_expected_output, lane_direction_scene_and_expected_output, \
    combined_scene_and_expected_output
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split


def test_plan_normalScene_accurateRoutePlanOutput(scene_static_pg_split: SceneStatic):
    # Test Data
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    scene_static_base = scene_static_pg_split.s_Data.s_SceneStaticBase
    navigation_plan = scene_static_pg_split.s_Data.s_NavigationPlan

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static_base, nav_plan=navigation_plan)
    route_plan_obj = BackpropagatingRoutePlanner()
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


@patch('decision_making.src.planning.route.backpropagating_route_planner.BACKPROP_DISCOUNT_FACTOR', 0.5)
def test_plan_constructionScenes_backpropagatedCostsCorrectRelativeToOtherLanes(construction_scene_and_expected_output: RoutePlanTestData):
    """
    In these scenes, the left and middle lanes are blocked on different road segments. Regardless, due to backpropgation,
    we only need to check the first road segment. We should see that the end costs for the left and middle lanes are
    basically equal and greater than the right lane cost. In this way, we're favoring being in the right lane.
    """
    # Test Data
    scene_static = construction_scene_and_expected_output.scene_static
    SceneStaticModel.get_instance().set_scene_static(scene_static)
    expected_output = construction_scene_and_expected_output.expected_binary_output

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = BackpropagatingRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    print(route_plan_output)

    # Assertions
    # Check that outputs other than lane end costs match expected binary output
    assert route_plan_output.e_Cnt_num_road_segments == expected_output.e_Cnt_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == expected_output.a_i_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == expected_output.a_Cnt_num_lane_segments.all()

    for i, road_segment in enumerate(route_plan_output.as_route_plan_lane_segments):
        for j, lane_segment in enumerate(road_segment):
            assert lane_segment.e_i_lane_segment_id == expected_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id
            assert lane_segment.e_cst_lane_occupancy_cost == expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost

    # Check lane end costs relative to other lanes
    assert abs(route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost
               - route_plan_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost) < 0.002
    assert route_plan_output.as_route_plan_lane_segments[0][0].e_cst_lane_end_cost < \
           route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost

@patch('decision_making.src.planning.route.backpropagating_route_planner.BACKPROP_DISCOUNT_FACTOR', 0.5)
def test_plan_mapScenes_backpropagatedCostsCorrectRelativeToOtherLanes(map_scene_and_expected_output: RoutePlanTestData):
    """
    These scenes are described in the if and elif statement blocks below. Due to backpropgation,
    we only need to check the first road segment.
    """
    # Test Data
    scene_static = map_scene_and_expected_output.scene_static
    SceneStaticModel.get_instance().set_scene_static(scene_static)
    expected_output = map_scene_and_expected_output.expected_binary_output
    scene_name = map_scene_and_expected_output.scene_name

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = BackpropagatingRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Assertions
    # Check that outputs other than lane end costs match expected binary output
    assert route_plan_output.e_Cnt_num_road_segments == expected_output.e_Cnt_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == expected_output.a_i_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == expected_output.a_Cnt_num_lane_segments.all()

    for i, road_segment in enumerate(route_plan_output.as_route_plan_lane_segments):
        for j, lane_segment in enumerate(road_segment):
            assert lane_segment.e_i_lane_segment_id == expected_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id
            assert lane_segment.e_cst_lane_occupancy_cost == expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost

    # Check lane end costs relative to other lanes
    if scene_name is "scene_one":
        # In this scene, the left lane on the second road segment is blocked. We should favor the middle and right lane equally.
        assert abs(route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost
                   - route_plan_output.as_route_plan_lane_segments[0][0].e_cst_lane_end_cost) < 0.002
        assert route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost < \
               route_plan_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost
    elif scene_name is "scene_two":
        # In this scene, all lanes on the last road segment are blocked. All lanes are favored equally.
        assert abs(route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost
                   - route_plan_output.as_route_plan_lane_segments[0][0].e_cst_lane_end_cost) < 0.002
        assert abs(route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost
                   - route_plan_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost) < 0.002

@patch('decision_making.src.planning.route.backpropagating_route_planner.BACKPROP_DISCOUNT_FACTOR', 0.5)
def test_plan_gmfaScenes_backpropagatedCostsCorrectRelativeToOtherLanes(gmfa_scene_and_expected_output: RoutePlanTestData):
    """
    In these scenes, the left lane is blocked on different road segments with various GMFA reasons.
    Regardless, due to backpropgation, we only need to check the first road segment. We should see
    that the end costs for the middle and right lanes are basically equal and less than the
    left lane cost. In this way, we're favoring being in the middle and right lanes.
    """
    # Test Data
    scene_static = gmfa_scene_and_expected_output.scene_static
    SceneStaticModel.get_instance().set_scene_static(scene_static)
    expected_output = gmfa_scene_and_expected_output.expected_binary_output

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = BackpropagatingRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Assertions
    # Check that outputs other than lane end costs match expected binary output
    assert route_plan_output.e_Cnt_num_road_segments == expected_output.e_Cnt_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == expected_output.a_i_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == expected_output.a_Cnt_num_lane_segments.all()

    for i, road_segment in enumerate(route_plan_output.as_route_plan_lane_segments):
        for j, lane_segment in enumerate(road_segment):
            assert lane_segment.e_i_lane_segment_id == expected_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id
            assert lane_segment.e_cst_lane_occupancy_cost == expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost

    # Check lane end costs relative to other lanes
    assert abs(route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost
               - route_plan_output.as_route_plan_lane_segments[0][0].e_cst_lane_end_cost) < 0.002
    assert route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost < \
           route_plan_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost


@patch('decision_making.src.planning.route.backpropagating_route_planner.BACKPROP_DISCOUNT_FACTOR', 0.5)
def test_plan_laneDirectionScenes_backpropagatedCostsCorrectRelativeToOtherLanes(lane_direction_scene_and_expected_output: RoutePlanTestData):
    """
    In this scene, the left lane is blocked on the second road segment. Due to backpropgation,
    we only need to check the first road segment. We should see that the end costs for the
    middle and right lanes are basically equal and less than the left lane cost.
    In this way, we're favoring being in the middle and right lanes.
    """
    # Test Data
    scene_static = lane_direction_scene_and_expected_output.scene_static
    SceneStaticModel.get_instance().set_scene_static(scene_static)
    expected_output = lane_direction_scene_and_expected_output.expected_binary_output

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = BackpropagatingRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Assertions
    # Check that outputs other than lane end costs match expected binary output
    assert route_plan_output.e_Cnt_num_road_segments == expected_output.e_Cnt_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == expected_output.a_i_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == expected_output.a_Cnt_num_lane_segments.all()

    for i, road_segment in enumerate(route_plan_output.as_route_plan_lane_segments):
        for j, lane_segment in enumerate(road_segment):
            assert lane_segment.e_i_lane_segment_id == expected_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id
            assert lane_segment.e_cst_lane_occupancy_cost == expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost

    # Check lane end costs relative to other lanes
    assert abs(route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost
               - route_plan_output.as_route_plan_lane_segments[0][0].e_cst_lane_end_cost) < 0.002
    assert route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost < \
           route_plan_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost


@patch('decision_making.src.planning.route.backpropagating_route_planner.BACKPROP_DISCOUNT_FACTOR', 0.5)
def test_plan_combinedScenes_backpropagatedCostsCorrectRelativeToOtherLanes(combined_scene_and_expected_output: RoutePlanTestData):
    """
    These scenes are described in the if and elif statement blocks below. Due to backpropgation,
    only the road segments where the end costs are expected to be different are checked.
    """
    # Test Data
    scene_static = combined_scene_and_expected_output.scene_static
    SceneStaticModel.get_instance().set_scene_static(scene_static)
    expected_output = combined_scene_and_expected_output.expected_binary_output
    scene_name = combined_scene_and_expected_output.scene_name

    # Route Planner Logic
    route_planner_input = RoutePlannerInputData()
    route_planner_input.reformat_input_data(scene=scene_static.s_Data.s_SceneStaticBase,
                                            nav_plan=scene_static.s_Data.s_NavigationPlan)
    route_plan_obj = BackpropagatingRoutePlanner()
    route_plan_output = route_plan_obj.plan(route_planner_input)

    # Assertions
    # Check that outputs other than lane end costs match expected binary output
    assert route_plan_output.e_Cnt_num_road_segments == expected_output.e_Cnt_num_road_segments
    assert route_plan_output.a_i_road_segment_ids.all() == expected_output.a_i_road_segment_ids.all()
    assert route_plan_output.a_Cnt_num_lane_segments.all() == expected_output.a_Cnt_num_lane_segments.all()

    for i, road_segment in enumerate(route_plan_output.as_route_plan_lane_segments):
        for j, lane_segment in enumerate(road_segment):
            assert lane_segment.e_i_lane_segment_id == expected_output.as_route_plan_lane_segments[i][j].e_i_lane_segment_id
            assert lane_segment.e_cst_lane_occupancy_cost == expected_output.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost

    # Check lane end costs relative to other lanes
    if scene_name is "scene_one":
        # In this scene, various lanes are blocked on different road segments. Since the relativity of the backpropagated
        # end costs changes depending on where it's are checked, we are checking it in three different places here. On
        # the first road segment, we should see that the right lane is favored over the other two due to the number of
        # obstructions in the middle and left lanes. However, once these obstructions are passed, we should see that
        # the left and middle lane are now favored do to a blocked right lane downstream.

        # Check on first road segment
        assert route_plan_output.as_route_plan_lane_segments[0][0].e_cst_lane_end_cost < \
               route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost < \
               route_plan_output.as_route_plan_lane_segments[0][2].e_cst_lane_end_cost

        # Check on second road segment
        assert route_plan_output.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost == \
               route_plan_output.as_route_plan_lane_segments[1][2].e_cst_lane_end_cost
        assert route_plan_output.as_route_plan_lane_segments[1][0].e_cst_lane_end_cost < \
               route_plan_output.as_route_plan_lane_segments[1][1].e_cst_lane_end_cost

        # Check on fourth road segment after all initial obstructions are passed
        assert route_plan_output.as_route_plan_lane_segments[3][1].e_cst_lane_end_cost == \
               route_plan_output.as_route_plan_lane_segments[3][2].e_cst_lane_end_cost
        assert route_plan_output.as_route_plan_lane_segments[3][0].e_cst_lane_end_cost > \
               route_plan_output.as_route_plan_lane_segments[3][1].e_cst_lane_end_cost
    elif scene_name is "scene_two":
        # See the description in the scene_fixtures file for scene details.

        # Check to see that right lane is favored in the beginning
        assert route_plan_output.as_route_plan_lane_segments[0][0].e_cst_lane_end_cost < \
               route_plan_output.as_route_plan_lane_segments[0][1].e_cst_lane_end_cost

        # Check to see that exit is required
        assert route_plan_output.as_route_plan_lane_segments[2][1].e_cst_lane_end_cost == \
               route_plan_output.as_route_plan_lane_segments[2][2].e_cst_lane_end_cost
        assert route_plan_output.as_route_plan_lane_segments[2][0].e_cst_lane_end_cost < \
               route_plan_output.as_route_plan_lane_segments[2][1].e_cst_lane_end_cost

        # Check to see we're staying left at the fork in order to reach goal
        assert route_plan_output.as_route_plan_lane_segments[3][1].e_cst_lane_end_cost < \
               route_plan_output.as_route_plan_lane_segments[3][0].e_cst_lane_end_cost
