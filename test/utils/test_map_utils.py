from decision_making.test.planning.behavioral.behavioral_state_fixtures import create_route_plan_msg
from decision_making.test.planning.behavioral.behavioral_state_fixtures import \
    behavioral_grid_state_with_objects_for_filtering_too_aggressive, state_with_objects_for_filtering_too_aggressive,\
    route_plan_20_30
from unittest.mock import patch

import numpy as np

from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.messages.scene_static_message import SceneStatic, StaticTrafficFlowControl, \
    RoadObjectType
from decision_making.src.messages.scene_static_enums import NominalPathPoint
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.types import FP_SX, FP_DX, FS_SX, FS_DX
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.exceptions import NavigationPlanDoesNotFitMap, NavigationPlanTooShort, DownstreamLaneNotFound, \
    UpstreamLaneNotFound
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split


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
    assert actual[0].s == 10.0
    assert actual[1].s == 12.0


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


def test_getLookaheadFrenetFrame_frenetStartsBehindAndEndsAheadOfCurrentLane_accurateFrameStartAndLength(
        scene_static_pg_split):
    """
    test method get_lookahead_frenet_frame:
        the current map has only one road segment;
        the frame starts and ends on arbitrary points.
    verify that final length, offset of GFF and conversion of an arbitrary point are accurate
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = -200.
    lookahead_dist = 500.
    arbitrary_fpoint = np.array([450., 1.])

    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])
    lane_id = lane_ids[current_ordinal]
    gff = MapUtils.get_lookahead_frenet_frame(lane_id, starting_lon, lookahead_dist,
                                              create_route_plan_msg(np.array(road_ids)))

    # validate the length of the obtained frenet frame
    assert abs(gff.s_max - lookahead_dist) < SMALL_DISTANCE_ERROR
    # calculate cartesian state of the origin of lane_id using GFF and using original frenet of lane_id and compare them
    gff_cpoint = gff.fpoint_to_cpoint(np.array([-starting_lon, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(lane_id).fpoint_to_cpoint(np.array([0, 0]))
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < SMALL_DISTANCE_ERROR

    # calculate cartesian state of some point using GFF and using original frenet (from the map) and compare them
    gff_cpoint = gff.fpoint_to_cpoint(arbitrary_fpoint)
    segment_id, segment_fstate = gff.convert_to_segment_state(np.array([arbitrary_fpoint[FP_SX], 0, 0,
                                                                        arbitrary_fpoint[FP_DX], 0, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(segment_id).fpoint_to_cpoint(segment_fstate[[FS_SX, FS_DX]])
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < SMALL_DISTANCE_ERROR


def test_advanceOnPlan_planFiveOutOfTenSegments_validateTotalLengthAndOrdinal(scene_static_pg_split):
    """
    test the method _advance_on_plan
        validate that total length of output sub segments == lookahead_dist;
    """

    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    lookahead_dist = 500.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])[current_ordinal]
    sub_segments = MapUtils._advance_on_plan(starting_lane_id, starting_lon, lookahead_dist,
                                             create_route_plan_msg(np.array(road_ids)).s_Data.a_i_road_segment_ids)
    assert len(sub_segments) == 5
    for seg in sub_segments:
        assert MapUtils.get_lane_ordinal(seg.e_i_SegmentID) == current_ordinal
    tot_length = sum([seg.e_i_SEnd - seg.e_i_SStart for seg in sub_segments])
    assert np.isclose(tot_length, lookahead_dist)


def test_advanceOnPlan_navPlanDoesNotFitMap_relevantException(scene_static_pg_split):
    """
    test the method _advance_on_plan
        add additional segment to nav_plan that does not exist on the map; validate getting the relevant exception
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
    # test navigation plan fitting the lookahead distance, and add non-existing road at the end of the plan
    # validate getting the relevant exception
    try:
        MapUtils._advance_on_plan(starting_lane_id, starting_lon, lookahead_dist,
                                  create_route_plan_msg(
                                      np.array(road_segment_ids[:nav_plan_length] + [wrong_road_segment_id])).s_Data.a_i_road_segment_ids)
        assert False
    except NavigationPlanDoesNotFitMap:
        assert True


def test_advanceOnPlan_lookaheadCoversFullMap_validateNoException(scene_static_pg_split):
    """
    test the method _advance_on_plan
        run lookahead_dist from the beginning until end of the map
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_ordinal = 1
    # test lookahead distance until the end of the map: verify no exception is thrown
    cumulative_distance = 0
    for road_id in road_segment_ids:
        lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[current_ordinal]
        cumulative_distance += MapUtils.get_lane_length(lane_id)
    first_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[0])[current_ordinal]
    sub_segments = MapUtils._advance_on_plan(first_lane_id, 0, cumulative_distance,
                                             create_route_plan_msg(np.array(road_segment_ids)).s_Data.a_i_road_segment_ids)
    assert len(sub_segments) == len(road_segment_ids)


def test_advanceOnPlan_navPlanTooShort_validateRelevantException(scene_static_pg_split):
    """
    test the method _advance_on_plan
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
    # test the case when the navigation plan is too short; validate the relevant exception
    try:
        MapUtils._advance_on_plan(starting_lane_id, starting_lon, lookahead_dist,
                                  create_route_plan_msg(np.array(road_segment_ids[:nav_plan_length])).s_Data.a_i_road_segment_ids)
        assert False
    except NavigationPlanTooShort:
        assert True


def test_advanceOnPlan_lookAheadDistLongerThanMap_validateException(scene_static_pg_split):
    """
    test the method _advance_on_plan
        test exception for too short map but nav_plan is long enough; validate the relevant exception
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_ids[current_road_idx])[current_ordinal]
    wrong_road_id = 1234
    lookadhead_dist = 1000
    # test the case when the map is too short; validate the relevant exception
    try:
        MapUtils._advance_on_plan(starting_lane_id, starting_lon, lookahead_distance=lookadhead_dist,
                                  route_plan_road_ids=create_route_plan_msg(
                                      np.array(road_segment_ids + [wrong_road_id])).s_Data.a_i_road_segment_ids)
        assert False
    except DownstreamLaneNotFound:
        assert True


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
    upstream_lanes = MapUtils.get_upstream_lanes(lane_id=current_lane_id)
    assert len(upstream_lanes) == 0

def test_getDownstreamLanes_emptyOnLastSegment(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    current_lane_id = 292
    downstream_lanes = MapUtils.get_downstream_lanes(lane_id=current_lane_id)
    print(downstream_lanes)
    assert len(downstream_lanes) == 0

def test_getUpstreamLanes_upstreamMatch(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    current_lane_id = 222
    upstream_of_current = 212
    upstream_lanes = MapUtils.get_upstream_lanes(lane_id=current_lane_id)
    assert upstream_lanes[0] == upstream_of_current


def test_getDownstreamLanes_downstreamMatch(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    current_lane_id = 212
    downstream_of_current = 222
    downstream_lanes = MapUtils.get_downstream_lanes(lane_id=current_lane_id)
    assert downstream_lanes[0] == downstream_of_current


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
    lane_id2 = MapUtils.get_downstream_lanes(lane_id1)[0]
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
    lane_id2 = MapUtils.get_downstream_lanes(lane_id1)[0]
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


def test_doesMapExistBackward_longBackwardDist_validateRelevantException(scene_static_pg_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)
    road_segment_ids = MapUtils.get_road_segment_ids()
    road_segment_id = road_segment_ids[2]
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[0]
    assert MapUtils.does_map_exist_backward(lane_id, 200)
    assert not MapUtils.does_map_exist_backward(lane_id, 400)
