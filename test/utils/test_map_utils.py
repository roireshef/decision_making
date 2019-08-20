import numpy as np
from unittest.mock import patch
import pytest

from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.messages.scene_static_message import SceneStatic, StaticTrafficFlowControl, \
    RoadObjectType
from decision_making.src.messages.route_plan_message import RoutePlanLaneSegment
from decision_making.src.messages.scene_static_enums import NominalPathPoint
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FP_SX, FP_DX, FS_SX, FS_DX, SIGN_S
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GFF_Type
from decision_making.test.planning.behavioral.behavioral_state_fixtures import \
    behavioral_grid_state_with_objects_for_filtering_too_aggressive, state_with_objects_for_filtering_too_aggressive, \
    route_plan_20_30, create_route_plan_msg, route_plan_lane_splits_offset
from decision_making.test.planning.custom_fixtures import route_plan_1_2, route_plan_1_2_3
from decision_making.src.exceptions import NavigationPlanDoesNotFitMap, NavigationPlanTooShort, DownstreamLaneNotFound, \
    UpstreamLaneNotFound, ValidLaneAheadTooShort
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split, right_lane_split_scene_static, \
    left_right_lane_split_scene_static, scene_static_short_testable, scene_static_left_lane_ends, scene_static_right_lane_ends, \
    left_lane_split_scene_static, scene_static_lane_split_on_left_ends, scene_static_lane_splits_on_left_and_right_offset
from decision_making.src.global_constants import PLANNING_LOOKAHEAD_DIST, MAX_HORIZON_DISTANCE


MAP_SPLIT = "PG_split.bin"
SMALL_DISTANCE_ERROR = 0.01


def test_getStaticTrafficFlowControlsS_findsSingleStopIdx(scene_static_pg_split: SceneStatic,
                                                          behavioral_grid_state_with_objects_for_filtering_too_aggressive):

    gff = behavioral_grid_state_with_objects_for_filtering_too_aggressive.extended_lane_frames[RelativeLane.SAME_LANE]

    # A Frenet-Frame trajectory: a numpy matrix of FrenetState2D [:, [FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA]]
    gff_state = np.array([[12.0, 0., 0., 0., 0., 0.]])
    lane_id, segment_states = gff.convert_to_segment_states(gff_state)
    segment_s = segment_states[0][0]

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    stop_sign = StaticTrafficFlowControl(e_e_road_object_type=RoadObjectType.StopSign, e_l_station=segment_s,
                                         e_Pct_confidence=1.0)
    yield_sign = StaticTrafficFlowControl(e_e_road_object_type=RoadObjectType.Yield, e_l_station=segment_s-1.0,
                                         e_Pct_confidence=1.0)
    stop_bar = StaticTrafficFlowControl(e_e_road_object_type=RoadObjectType.StopBar_Right, e_l_station=segment_s-2.0,
                                         e_Pct_confidence=1.0)
    MapUtils.get_lane(lane_id).as_static_traffic_flow_control.append(stop_sign)
    MapUtils.get_lane(lane_id).as_static_traffic_flow_control.append(yield_sign)
    MapUtils.get_lane(lane_id).as_static_traffic_flow_control.append(stop_bar)
    gff = behavioral_grid_state_with_objects_for_filtering_too_aggressive.extended_lane_frames[RelativeLane.SAME_LANE]
    actual = MapUtils.get_stop_bar_and_stop_sign(gff)
    assert len(actual) == 2
    assert actual[0][SIGN_S] == 10.0
    assert actual[1][SIGN_S] == 12.0

def test_getRoadSegmentIdFromLaneId_correct(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    lane_id = 222
    expected_result = 22
    actual_result = MapUtils.get_road_segment_id_from_lane_id(lane_id)
    assert actual_result == expected_result


def test_getAdjacentLanes_adjacentOfRightestAndSecondLanes_accurate(scene_static_pg_split):
    """
    test method get_adjacent_lane_ids for the current map;
    check adjacent lanes of the rightest and the second-from-right lanes
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_ids = MapUtils.get_road_segment_ids()

    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[2])
    right_to_rightest = MapUtils.get_adjacent_lane_ids(lane_ids[0], RelativeLane.RIGHT_LANE)
    assert len(right_to_rightest) == 0
    left_to_rightest = MapUtils.get_adjacent_lane_ids(lane_ids[0], RelativeLane.LEFT_LANE)
    assert left_to_rightest == lane_ids[1:]
    right_to_second = MapUtils.get_adjacent_lane_ids(lane_ids[1], RelativeLane.RIGHT_LANE)
    assert right_to_second == [lane_ids[0]]
    left_to_second = MapUtils.get_adjacent_lane_ids(lane_ids[1], RelativeLane.LEFT_LANE)
    assert left_to_second == lane_ids[2:]
    left_to_leftmost = MapUtils.get_adjacent_lane_ids(lane_ids[-1], RelativeLane.LEFT_LANE)
    assert len(left_to_leftmost) == 0


def test_getDistToLaneBorders_rightLane_equalToHalfLaneWidth(scene_static_pg_split):
    """
    test method get_dist_to_lane_borders:
        in the current map the lanes have a constant lane width and all lanes have the same width;
        therefore it should return half lane width
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_ids = MapUtils.get_road_segment_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])
    dist_to_right, dist_to_left = MapUtils.get_dist_to_lane_borders(lane_ids[0], 0)
    assert dist_to_right == dist_to_left


def test_getLookaheadFrenetFrameByCost_onEndingLane_PartialGFFCreated(scene_static_left_lane_ends, route_plan_1_2):
    """
    Make sure a partial GFF is created when the left lane suddenly ends
    :param scene_static_left_lane_ends:
    :return:
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_left_lane_ends)

    starting_lon = 800
    starting_lane = 12

    del route_plan_1_2.s_Data.as_route_plan_lane_segments[1][2]
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[1] -= 1

    gff_dict = MapUtils.get_lookahead_frenet_frame_by_cost(starting_lane, starting_lon, route_plan_1_2)
    # check partial SAME_LANE
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [12])
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFF_Type.Partial


def test_getLookaheadFrenetFrameByCost_onFullLane_NormalGFFCreated(scene_static_right_lane_ends, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(scene_static_right_lane_ends)

    starting_lon = 800
    starting_lane = 11

    gff_dict = MapUtils.get_lookahead_frenet_frame_by_cost(starting_lane, starting_lon, route_plan_1_2)
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11,21])
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFF_Type.Normal


def test_getLookaheadFrenetFrameByCost_LeftSplitAugmentedGFFCreated(left_lane_split_scene_static, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(left_lane_split_scene_static)

    starting_lon = 700.
    starting_lane = 11
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: False}

    gff_dict = MapUtils.get_lookahead_frenet_frame_by_cost(starting_lane, starting_lon, route_plan_1_2, can_augment = can_augment)

    # check same_lane
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFF_Type.Normal
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11, 21])
    # check augmented right lane
    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFF_Type.Augmented
    assert np.array_equal(gff_dict[RelativeLane.LEFT_LANE].segment_ids, [11, 22])


def test_getLookaheadFrenetFrameByCost_RightSplitAugmentedGFFCreated(right_lane_split_scene_static, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(right_lane_split_scene_static)

    starting_lon = 700.
    starting_lane = 11
    can_augment = {RelativeLane.LEFT_LANE: False, RelativeLane.RIGHT_LANE: True}

    gff_dict = MapUtils.get_lookahead_frenet_frame_by_cost(starting_lane, starting_lon, route_plan_1_2, can_augment=can_augment)

    # check same_lane
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFF_Type.Normal
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11, 21])
    # check augmented right lane
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFF_Type.Augmented
    assert np.array_equal(gff_dict[RelativeLane.RIGHT_LANE].segment_ids, [11, 20])


def test_getLookaheadFrenetFrameByCost_LeftRightSplitAugmentedGFFsCreated(left_right_lane_split_scene_static, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(left_right_lane_split_scene_static)
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: True}

    # Modify the route plan
    # In order to match the scene static data, the left and right lane in the first road segment needs to be deleted since
    # it does not exist in left_right_lane_split_scene_static.
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[0][0]
    # delete index [0][1] instead of [0][2] since the first delete shifts all the indicies
    del route_plan_1_2.s_Data.as_route_plan_lane_segments[0][1]
    route_plan_1_2.s_Data.a_Cnt_num_lane_segments[0] = 1

    gff_dict = MapUtils.get_lookahead_frenet_frame_by_cost(11, 600, route_plan_1_2, can_augment=can_augment)

    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFF_Type.Augmented
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFF_Type.Augmented
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFF_Type.Normal
    assert gff_dict[RelativeLane.LEFT_LANE].has_segment_id(22)
    assert gff_dict[RelativeLane.RIGHT_LANE].has_segment_id(20)
    assert gff_dict[RelativeLane.SAME_LANE].has_segment_id(21)


def test_getLookaheadFrenetFrameByCost_CanAugmentButNoSplit_NoAugmentedCreated(scene_static_short_testable, route_plan_1_2):
    SceneStaticModel.get_instance().set_scene_static(scene_static_short_testable)
    starting_lon = 700.
    starting_lane = 11
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: True}

    gff_dict = MapUtils.get_lookahead_frenet_frame_by_cost(starting_lane, starting_lon, route_plan_1_2, can_augment=can_augment)

    # check same_lane
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFF_Type.Normal
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [11, 21])
    assert RelativeLane.LEFT_LANE not in gff_dict
    assert RelativeLane.RIGHT_LANE not in gff_dict


def test_getLookaheadFrenetFrameByCost_OffsetSplits_BothAugmentedCreated(scene_static_lane_splits_on_left_and_right_offset, route_plan_lane_splits_offset):
    SceneStaticModel.get_instance().set_scene_static(scene_static_lane_splits_on_left_and_right_offset)
    starting_lon = 10.
    starting_lane = 211
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: True}

    gff_dict = MapUtils.get_lookahead_frenet_frame_by_cost(starting_lane, starting_lon, route_plan_lane_splits_offset, can_augment=can_augment)

    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFF_Type.Normal
    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFF_Type.Augmented
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFF_Type.Augmented
    assert np.array_equal(gff_dict[RelativeLane.SAME_LANE].segment_ids, [201, 211, 221, 231, 241])
    assert np.array_equal(gff_dict[RelativeLane.LEFT_LANE].segment_ids, [201, 211, 222, 232, 242])
    assert np.array_equal(gff_dict[RelativeLane.RIGHT_LANE].segment_ids, [201, 211, 221, 230, 240])


def test_getLookaheadFrenetFrameByCost_frenetStartsBehindAndEndsAheadOfCurrentLane_accurateFrameStartAndLength(
        scene_static_pg_split, route_plan_20_30):
    """
    test method get_lookahead_frenet_frame_by_cost:
        the frame starts and ends on arbitrary points.
    verify that final length, offset of GFF and conversion of an arbitrary point are accurate
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    station = 50.0
    arbitrary_fpoint = np.array([450., 1.])

    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])
    lane_id = lane_ids[current_ordinal]
    gff = MapUtils.get_lookahead_frenet_frame_by_cost(lane_id, station, route_plan_20_30)[RelativeLane.SAME_LANE]

    # validate the length of the obtained frenet frame
    assert abs(gff.s_max - (PLANNING_LOOKAHEAD_DIST + MAX_HORIZON_DISTANCE)) < SMALL_DISTANCE_ERROR
    # calculate cartesian state of the origin of lane_id using GFF and using original frenet of lane_id and compare them
    gff_cpoint = gff.fpoint_to_cpoint(np.array([PLANNING_LOOKAHEAD_DIST - station, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(lane_id).fpoint_to_cpoint(np.array([0, 0]))
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < SMALL_DISTANCE_ERROR

    # calculate cartesian state of some point using GFF and using original frenet (from the map) and compare them
    gff_cpoint = gff.fpoint_to_cpoint(arbitrary_fpoint)
    segment_id, segment_fstate = gff.convert_to_segment_state(np.array([arbitrary_fpoint[FP_SX], 0, 0,
                                                                        arbitrary_fpoint[FP_DX], 0, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(segment_id).fpoint_to_cpoint(segment_fstate[[FS_SX, FS_DX]])
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < SMALL_DISTANCE_ERROR


@patch('decision_making.src.utils.map_utils.MAX_HORIZON_DISTANCE', 900)
def test_getLookaheadFrenet_AugmentedPartialCreatedWhenSplitEnds(left_right_lane_split_scene_static, route_plan_1_2_3):
    """
    Make sure that partial/augmentedPartial GFFS are created when the lookahead distance is set to be very far ahead
    :param left_right_lane_split_scene_static:
    :param route_plan_1_2_3:
    :return:
    """
    SceneStaticModel.get_instance().set_scene_static(left_right_lane_split_scene_static)
    can_augment = {RelativeLane.LEFT_LANE: True, RelativeLane.RIGHT_LANE: True}

    # Modify the route plan
    # In order to match the scene static data, the left and right lane in the first road segment needs to be deleted since
    # it does not exist in left_right_lane_split_scene_static.
    del route_plan_1_2_3.s_Data.as_route_plan_lane_segments[0][0]
    # delete index [0][1] instead of [0][2] since the first delete shifts all the indicies
    del route_plan_1_2_3.s_Data.as_route_plan_lane_segments[0][1]
    route_plan_1_2_3.s_Data.a_Cnt_num_lane_segments[0] = 1

    gff_dict = MapUtils.get_lookahead_frenet_frame_by_cost(11, 900, route_plan_1_2_3, can_augment=can_augment)

    assert gff_dict[RelativeLane.LEFT_LANE].gff_type == GFF_Type.AugmentedPartial
    assert gff_dict[RelativeLane.RIGHT_LANE].gff_type == GFF_Type.AugmentedPartial
    assert gff_dict[RelativeLane.SAME_LANE].gff_type == GFF_Type.Partial
    assert gff_dict[RelativeLane.LEFT_LANE].has_segment_id(22)
    assert gff_dict[RelativeLane.RIGHT_LANE].has_segment_id(20)
    assert gff_dict[RelativeLane.SAME_LANE].has_segment_id(21)


def test_advanceByCost_planFiveOutOfTenSegments_validateTotalLengthAndOrdinal(scene_static_pg_split, route_plan_20_30):
    """
    test the method _advance_by_cost
        validate that total length of output sub segments == lookahead_dist;
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    lookahead_dist = 500.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])[current_ordinal]
    sub_segments, is_partial, is_augmented = MapUtils._advance_by_cost(starting_lane_id, starting_lon, lookahead_dist, route_plan_20_30)[RelativeLane.SAME_LANE]
    assert len(sub_segments) == 5
    for seg in sub_segments:
        assert MapUtils.get_lane_ordinal(seg.e_i_SegmentID) == current_ordinal
    tot_length = sum([seg.e_i_SEnd - seg.e_i_SStart for seg in sub_segments])
    assert np.isclose(tot_length, lookahead_dist)
    assert is_partial == False
    assert is_augmented == False


def test_advanceByCost_navPlanDoesNotFitMap_partialLookahead(scene_static_pg_split, route_plan_20_30):
    """
    test the method _advance_by_cost
        add additional segment to nav_plan that does not exist on the map; validate a partial lookahead is done
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    lookahead_dist = 600.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[current_road_idx])[current_ordinal]
    wrong_road_segment_id = 1234
    nav_plan_length = 8

    # Modify route plan for this test case
    route_plan = route_plan_20_30
    route_plan.s_Data.e_Cnt_num_road_segments = nav_plan_length + 1
    route_plan.s_Data.a_i_road_segment_ids = np.array(route_plan.s_Data.a_i_road_segment_ids[:nav_plan_length].tolist() +
                                                      [wrong_road_segment_id])
    route_plan.s_Data.a_Cnt_num_lane_segments = route_plan.s_Data.a_Cnt_num_lane_segments[:(nav_plan_length + 1)]
    route_plan.s_Data.as_route_plan_lane_segments = route_plan.s_Data.as_route_plan_lane_segments[:(nav_plan_length + 1)]

    lane_number = 1

    for lane_segment in route_plan.s_Data.as_route_plan_lane_segments[-1]:
        lane_segment.e_i_lane_segment_id = wrong_road_segment_id + lane_number
        lane_number += 1

    # test navigation plan fitting the lookahead distance, and add non-existing road at the end of the plan
    # validate getting the relevant exception
    subsegs, is_partial, is_augmented = MapUtils._advance_by_cost(starting_lane_id, starting_lon, lookahead_dist, route_plan)[RelativeLane.SAME_LANE]

    subseg_ids = [subseg.e_i_SegmentID for subseg in subsegs]

    # verify the wrong road segment is not added
    # make sure that the non-existent road segment is not contained in the GFF
    assert np.all([MapUtils.get_road_segment_id_from_lane_id(subseg.e_i_SegmentID) != wrong_road_segment_id for subseg in subsegs])
    # make sure that the previous existing road segments were used
    assert len(subsegs) == nav_plan_length - current_road_idx
    # make sure the lanes are in the correct order
    assert np.array_equal(subseg_ids, [231, 241, 251, 261, 271])
    # make sure the GFF created was of type Partial since it should not extend the entire route plan
    assert is_partial == True


def test_advanceByCost_navPlanTooShort_validateRelevantException(scene_static_pg_split, route_plan_20_30):
    """
    test the method _advance_by_cost
        test exception for too short nav plan; validate the relevant exception
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    lookahead_dist = 500.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[current_road_idx])[current_ordinal]
    nav_plan_length = 7

    # Modify route plan for this test case
    route_plan = route_plan_20_30
    route_plan.s_Data.e_Cnt_num_road_segments = nav_plan_length
    route_plan.s_Data.a_i_road_segment_ids = route_plan.s_Data.a_i_road_segment_ids[:nav_plan_length]
    route_plan.s_Data.a_Cnt_num_lane_segments = route_plan.s_Data.a_Cnt_num_lane_segments[:nav_plan_length]
    route_plan.s_Data.as_route_plan_lane_segments = route_plan.s_Data.as_route_plan_lane_segments[:nav_plan_length]

    # test the case when the navigation plan is too short; validate the relevant exception
    try:
        MapUtils._advance_by_cost(starting_lane_id, starting_lon, lookahead_dist, route_plan)
        assert False
    except NavigationPlanTooShort:
        assert True


def test_advanceByCost_lookAheadDistLongerThanMap_validatePartialLookahead(scene_static_pg_split, route_plan_20_30):
    """
    test the method _advance_by_cost
        test exception for too short map but nav_plan is long enough; validate the relevant exception
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 9
    current_ordinal = 1
    starting_lon = 50.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[current_road_idx])[current_ordinal]
    lookadhead_dist = 1000

    # Modify route plan to make it extend past the last lane in the scene_static fixture
    route_plan = route_plan_20_30
    route_plan.s_Data.e_Cnt_num_road_segments += 1
    route_plan.s_Data.a_i_road_segment_ids = np.append(route_plan.s_Data.a_i_road_segment_ids, 30)
    route_plan.s_Data.a_Cnt_num_lane_segments = np.append(route_plan.s_Data.a_Cnt_num_lane_segments,
                                                          route_plan.s_Data.a_Cnt_num_lane_segments[-1])
    route_plan.s_Data.as_route_plan_lane_segments.append([RoutePlanLaneSegment(300,0,0),
                                                          RoutePlanLaneSegment(301,0,0),
                                                          RoutePlanLaneSegment(302,0,0)])

    # test the case when the map is too short; validate partial lookahead is done
    subsegs, is_partial, is_augmented = MapUtils._advance_by_cost(starting_lane_id, starting_lon, lookadhead_dist, route_plan)[RelativeLane.SAME_LANE]
    subseg_ids = [subseg.e_i_SegmentID for subseg in subsegs]

    # make sure the subsegments are in the correct order
    assert np.array_equal(subseg_ids, [291])
    # make sure the the gff is marked as partial
    assert is_partial == True
    assert is_augmented == False


def test_getUpstreamLanesFromDistance_upstreamFiveOutOfTenSegments_validateLength(scene_static_pg_split):
    """
     test the method _get_upstream_lanes_from_distance
         validate that total length of output sub segments == lookahead_dist; validate lanes' ordinal
     """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 7
    current_ordinal = 1
    starting_lon = 20.
    backward_dist = 500.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[current_road_idx])[current_ordinal]
    lane_ids, final_lon = MapUtils._get_upstream_lanes_from_distance(starting_lane_id, starting_lon, backward_dist)
    tot_length = starting_lon - final_lon
    # validate: total length of the segments equals to backward_dist and correctness of the segments' ordinal
    for lane_id in lane_ids[1:]:  # exclude the starting lane
        assert MapUtils.get_lane_ordinal(lane_id) == current_ordinal
        tot_length += MapUtils.get_lane_length(lane_id)
    assert np.isclose(tot_length, backward_dist)


def test_getUpstreamLanesFromDistance_smallBackwardDist_validateLaneIdAndLength(scene_static_pg_split):
    """
     test the method _get_upstream_lanes_from_distance
        test small backward_dist ending on the same lane; validate the same lane_id and final longitude
     """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 7
    current_ordinal = 1
    starting_lon = 20.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[current_road_idx])[current_ordinal]

    # test small backward_dist ending on the same lane
    small_backward_dist = 1
    lane_ids, final_lon = MapUtils._get_upstream_lanes_from_distance(starting_lane_id, starting_lon,
                                                                     backward_dist=small_backward_dist)
    assert lane_ids == [starting_lane_id]
    assert final_lon == starting_lon - small_backward_dist


def test_getUpstreamLanesFromDistance_backwardDistForFullMap_validateSegmentsNumberAndFinalLon(scene_static_pg_split):
    """
     test the method _get_upstream_lanes_from_distance
         try lookahead_dist until start of the map; validate there are no exceptions and segments number
     """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_ordinal = 1
    # test from the end until start of the map: verify no exception is thrown
    cumulative_distance = 0
    for road_id in road_segment_ids:
        lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[current_ordinal]
        cumulative_distance += MapUtils.get_lane_length(lane_id)
    last_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[-1])[current_ordinal]
    last_lane_length = MapUtils.get_lane_length(last_lane_id)
    lane_ids, final_lon = MapUtils._get_upstream_lanes_from_distance(last_lane_id, last_lane_length, cumulative_distance)
    # validate the number of segments and final longitude
    assert len(lane_ids) == len(road_segment_ids)
    assert final_lon == 0


def test_getUpstreamLanesFromDistance_tooLongBackwardDist_validateRelevantException(scene_static_pg_split):
    """
     test the method _get_upstream_lanes_from_distance
         validate the relevant exception
     """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 7
    current_ordinal = 1
    starting_lon = 20.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])[current_ordinal]
    backward_dist = 1000
    # test the case when the map is too short
    try:
        MapUtils._get_upstream_lanes_from_distance(starting_lane_id, starting_lon, backward_dist=backward_dist)
        assert False
    except UpstreamLaneNotFound:
        assert True


def test_getUpstreamLanes_emptyOnFirstSegment(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    current_lane_id = 202
    upstream_lane_ids = MapUtils.get_upstream_lane_ids(lane_id=current_lane_id)
    assert len(upstream_lane_ids) == 0


def test_getDownstreamLanes_emptyOnLastSegment(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    current_lane_id = 292
    downstream_lane_ids = MapUtils.get_downstream_lane_ids(lane_id=current_lane_id)
    print(downstream_lane_ids)
    assert len(downstream_lane_ids) == 0


def test_getUpstreamLanes_upstreamMatch(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    current_lane_id = 222
    upstream_of_current = 212
    upstream_lane_ids = MapUtils.get_upstream_lane_ids(lane_id=current_lane_id)
    assert upstream_lane_ids[0] == upstream_of_current


def test_getDownstreamLanes_downstreamMatch(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    current_lane_id = 212
    downstream_of_current = 222
    downstream_lane_ids = MapUtils.get_downstream_lane_ids(lane_id=current_lane_id)
    assert downstream_lane_ids[0] == downstream_of_current


def test_getClosestLane_multiLaneRoad_findRightestAndLeftestLanesByPoints(scene_static_pg_split):
    """
    test method get_closest_lane:
        find the most left and the most right lanes by points inside these lanes
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[0])
    # take the rightest lane
    lane_id = lane_ids[0]
    frenet = MapUtils.get_lane_frenet_frame(lane_id)
    closest_lane_id = MapUtils.get_closest_lane(frenet.points[1])
    assert lane_id == closest_lane_id
    # take the leftmost lane
    lane_id = lane_ids[-1]
    frenet = MapUtils.get_lane_frenet_frame(lane_id)
    closest_lane_id = MapUtils.get_closest_lane(frenet.points[-2])
    assert lane_id == closest_lane_id


def test_getClosestLane_nearLanesSeam_closestPointIsInternal(scene_static_pg_split):
    # take a far input point, such that there are two closest lanes from the point and
    # the closest point in one of them is internal point (not start/end point)
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[0])
    # take the rightest lane
    lane_id1 = lane_ids[0]
    lane_id2 = MapUtils.get_downstream_lane_ids(lane_id1)[0]
    x_index = NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value
    y_index = NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value
    seam_point = MapUtils.get_lane_geometry(lane_id2).a_nominal_path_points[0]
    point_xy = seam_point[[x_index, y_index]]
    yaw = seam_point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value]
    distance_to_point = 1000
    normal_angle = yaw + np.pi/2  # normal to yaw
    cpoint = point_xy + distance_to_point * np.array([np.cos(normal_angle), np.sin(normal_angle)])
    lane = MapUtils.get_closest_lane(cpoint)
    MapUtils.get_lane_frenet_frame(lane).cpoint_to_fpoint(cpoint)  # verify that the conversion does not crash


def test_getClosestLane_nearLanesSeam_laneAccordingToYaw(scene_static_pg_split):
    # take an input point close to the lanes seam, such that there are two closest lanes from the point
    # sharing the same closest (non-internal) lane-point
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[0])
    # take the rightest lane
    lane_id1 = lane_ids[0]
    lane_id2 = MapUtils.get_downstream_lane_ids(lane_id1)[0]
    x_index = NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value
    y_index = NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value
    seam_point = MapUtils.get_lane_geometry(lane_id2).a_nominal_path_points[0]
    point_xy = seam_point[[x_index, y_index]]
    yaw = seam_point[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value]
    distance_to_point = 0.2
    yaw1 = yaw + 2  # obtuse angle with yaw
    cpoint1 = point_xy + distance_to_point * np.array([np.cos(yaw1), np.sin(yaw1)])
    lane1 = MapUtils.get_closest_lane(cpoint1)
    assert lane1 == lane_id1
    MapUtils.get_lane_frenet_frame(lane1).cpoint_to_fpoint(cpoint1)  # verify that the conversion does not crash
    yaw2 = yaw + 1  # acute angle with yaw
    cpoint2 = point_xy + distance_to_point * np.array([np.cos(yaw2), np.sin(yaw2)])
    lane2 = MapUtils.get_closest_lane(cpoint2)
    assert lane2 == lane_id2
    MapUtils.get_lane_frenet_frame(lane2).cpoint_to_fpoint(cpoint2)  # verify that the conversion does not crash


def test_getLanesIdsFromRoadSegmentId_multiLaneRoad_validateIdsConsistency(scene_static_pg_split):
    """
    test method get_lanes_ids_from_road_segment_id
        validate consistency between road segment ids and lane ids
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    road_segment_id = road_segment_ids[0]
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)
    assert road_segment_id == MapUtils.get_road_segment_id_from_lane_id(lane_ids[0])
    assert road_segment_id == MapUtils.get_road_segment_id_from_lane_id(lane_ids[-1])


def test_getUpstreamLaneSubsegments_backwardHorizonOnLane_NoUpstreamLaneSubsegments(scene_static_pg_split: SceneStatic):
    """
    Test _get_upstream_lane_subsegments
    The distance to travel backwards is small enough that it is still on the same lane. This should result in no upstream lane subsegments
    returned.
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    upstream_lane_subsegments = MapUtils._get_upstream_lane_subsegments(200, 100, 50)
    assert upstream_lane_subsegments == []


def test_getUpstreamLaneSubsegments_backwardHorizonPassesBeginningOfLane_CorrectUpstreamLaneSubsegments(scene_static_pg_split: SceneStatic):
    """
    Test _get_upstream_lane_subsegments
    This tests the scenario where the backwards horizon extends to upstream lanes and they exist. The result should be the "normal" output
    with subsegments extending back as far as the provided horizon with start and end stations assigned accordingly. The expected values
    were calculated as follows:

        station_on_220 = 5
        backward_distance = 150
        length_of_200 = 120.84134201631973  (from pickle file)
        length_of_210 = 119.64304560784024  (from pickle file)

        start_station_on_200 = (length_of_200 + length_of_210 + station_on_220) - backward_distance
        end_station_on_200 = length_of_200

        start_station_on_210 = 0.0  (beginning of lane segment)
        end_station_on_210 = length_of_210
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    upstream_lane_subsegments = MapUtils._get_upstream_lane_subsegments(220, 5, 150)

    # Check size
    assert len(upstream_lane_subsegments) == 2

    # Check order
    assert upstream_lane_subsegments[0].e_i_SegmentID == 200
    assert upstream_lane_subsegments[1].e_i_SegmentID == 210

    # Check start and end stations
    assert upstream_lane_subsegments[0].e_i_SStart == 95.48438762415998
    assert upstream_lane_subsegments[0].e_i_SEnd == 120.84134201631973

    assert upstream_lane_subsegments[1].e_i_SStart == 0.0
    assert upstream_lane_subsegments[1].e_i_SEnd == 119.64304560784024


def test_getUpstreamLaneSubsegments_NoUpstreamLane_CorrectUpstreamLaneSubsegments(scene_static_pg_split: SceneStatic):
    """
    Test _get_upstream_lane_subsegments
    This tests the scenario where the backwards horizon extends to upstream lanes but an upstream lane doesn't exist at some point. With
    starting close to the beginning of lane 210 and going backwards 150 m, the beginning of lane 200 is passed. Since lane 200 doesn't
    have any upstream lanes, the search should end and the lane subsegment should include the entire length of lane 200.
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    upstream_lane_subsegments = MapUtils._get_upstream_lane_subsegments(210, 5, 150)

    # Check size
    assert len(upstream_lane_subsegments) == 1

    # Check ID
    assert upstream_lane_subsegments[0].e_i_SegmentID == 200

    # Check start and end stations
    assert upstream_lane_subsegments[0].e_i_SStart == 0.0
    assert upstream_lane_subsegments[0].e_i_SEnd == 120.84134201631973  # Lane length taken from pickle file


