import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.exceptions import NavigationPlanTooShort, DownstreamLaneNotFound, UpstreamLaneNotFound, \
    NavigationPlanDoesNotFitMap, RoadNotFound
from mapping.src.service.map_service import MapService
from decision_making.src.planning.types import FS_SX, FS_DX, FP_SX, FP_DX

MAP_SPLIT = "PG_split.bin"
SMALL_DISTANCE_ERROR = 0.01


def test_getAdjacentLanes_adjacentOfRightestAndSecondLanes_accurate():
    """
    test method get_adjacent_lanes for the current map;
    check adjacent lanes of the rightest and the second-from-right lanes
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[2])
    right_to_rightest = MapUtils.get_adjacent_lanes(lane_ids[0], RelativeLane.RIGHT_LANE)
    assert len(right_to_rightest) == 0
    left_to_rightest = MapUtils.get_adjacent_lanes(lane_ids[0], RelativeLane.LEFT_LANE)
    assert left_to_rightest == lane_ids[1:]
    right_to_second = MapUtils.get_adjacent_lanes(lane_ids[1], RelativeLane.RIGHT_LANE)
    assert right_to_second == [lane_ids[0]]
    left_to_second = MapUtils.get_adjacent_lanes(lane_ids[1], RelativeLane.LEFT_LANE)
    assert left_to_second == lane_ids[2:]
    left_to_leftmost = MapUtils.get_adjacent_lanes(lane_ids[-1], RelativeLane.LEFT_LANE)
    assert len(left_to_leftmost) == 0


def test_getDistToLaneBorders_rightLane_equalToHalfLaneWidth():
    """
    test method get_dist_to_lane_borders:
        in the current map the lanes have a constant lane width and all lanes have the same width;
        therefore it should return half lane width
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])
    dist_to_right, dist_to_left = MapUtils.get_dist_to_lane_borders(lane_ids[0], 0)
    assert dist_to_right == dist_to_left
    assert dist_to_right == MapService.get_instance().get_road(road_ids[0]).lane_width/2


def test_getDistToRoadBorders_rightLane_equalToDistFromRoadBorder():
    """
    test method get_dist_from_lane_center_to_road_borders:
        in the current map the lanes have a constant lane width and all lanes have the same width
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])
    dist_to_right, dist_to_left = MapUtils.get_dist_to_road_borders(lane_ids[0], 0)
    lane_width = MapService.get_instance().get_road(road_ids[0]).lane_width
    assert dist_to_right == lane_width/2
    assert dist_to_left == lane_width * (len(lane_ids) - 0.5)


def test_getLookaheadFrenetFrame_frenetStartsBehindAndEndsAheadOfCurrentLane_accurateFrameStartAndLength():
    """
    test method get_lookahead_frenet_frame:
        the current map has only one road segment;
        the frame starts and ends on arbitrary points.
    verify that final length, offset of GFF and conversion of an arbitrary point are accurate
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = -200.
    lookahead_dist = 500.
    arbitrary_fpoint = np.array([450., 1.])

    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])
    lane_id = lane_ids[current_ordinal]
    gff = MapUtils.get_lookahead_frenet_frame(lane_id, starting_lon, lookahead_dist, NavigationPlanMsg(np.array(road_ids)))

    # validate the length of the obtained frenet frame
    assert abs(gff.s_max - lookahead_dist) < SMALL_DISTANCE_ERROR
    # calculate cartesian state of the origin of lane_id using GFF and using original frenet of lane_id and compare them
    gff_cpoint = gff.fpoint_to_cpoint(np.array([-starting_lon, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(lane_id).fpoint_to_cpoint(np.array([0, 0]))
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < SMALL_DISTANCE_ERROR

    # calculate cartesian state of some point using GFF and using original frenet (from the map) and compare them
    gff_cpoint = gff.fpoint_to_cpoint(arbitrary_fpoint)
    segment_id, segment_fstate = gff.convert_to_segment_state(np.array([arbitrary_fpoint[FP_SX], 0, 0, arbitrary_fpoint[FP_DX], 0, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(segment_id).fpoint_to_cpoint(segment_fstate[[FS_SX, FS_DX]])
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < SMALL_DISTANCE_ERROR


def test_advanceOnPlan_planFiveOutOfTenSegments_validateTotalLengthAndOrdinal():
    """
    test the method _advance_on_plan
        validate that total length of output sub segments == lookahead_dist;
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    lookahead_dist = 500.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])[current_ordinal]
    sub_segments = MapUtils._advance_on_plan(starting_lane_id, starting_lon, lookahead_dist, NavigationPlanMsg(np.array(road_ids)))
    assert len(sub_segments) == 5
    for seg in sub_segments:
        assert MapUtils.get_lane_ordinal(seg.segment_id) == current_ordinal
    tot_length = sum([seg.s_end - seg.s_start for seg in sub_segments])
    assert np.isclose(tot_length, lookahead_dist)


def test_advanceOnPlan_navPlanDoesNotFitMap_relevantException():
    """
    test the method _advance_on_plan
        add additional segment to nav_plan that does not exist on the map; validate getting the relevant exception
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    lookahead_dist = 600.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])[current_ordinal]
    wrong_road_segment_id = 1234
    nav_plan_length = 8
    # test navigation plan fitting the lookahead distance, and add non-existing road at the end of the plan
    # validate getting the relevant exception
    try:
        MapUtils._advance_on_plan(starting_lane_id, starting_lon, lookahead_dist,
                                  NavigationPlanMsg(np.array(road_ids[:nav_plan_length] + [wrong_road_segment_id])))
        assert False
    except NavigationPlanDoesNotFitMap:
        assert True


def test_advanceOnPlan_lookaheadCoversFullMap_validateNoException():
    """
    test the method _advance_on_plan
        run lookahead_dist from the beginning until end of the map
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    current_ordinal = 1
    # test lookahead distance until the end of the map: verify no exception is thrown
    cumulative_distance = 0
    for road_id in road_ids:
        lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[current_ordinal]
        cumulative_distance += MapUtils.get_lane_length(lane_id)
    first_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])[current_ordinal]
    sub_segments = MapUtils._advance_on_plan(first_lane_id, 0, cumulative_distance, NavigationPlanMsg(np.array(road_ids)))
    assert len(sub_segments) == len(road_ids)


def test_advanceOnPlan_navPlanTooShort_validateRelevantException():
    """
    test the method _advance_on_plan
        test exception for too short nav plan; validate the relevant exception
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    lookahead_dist = 500.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])[current_ordinal]
    nav_plan_length = 7
    # test the case when the navigation plan is too short; validate the relevant exception
    try:
        MapUtils._advance_on_plan(starting_lane_id, starting_lon, lookahead_dist,
                                  NavigationPlanMsg(np.array(road_ids[:nav_plan_length])))
        assert False
    except NavigationPlanTooShort:
        assert True


def test_advanceOnPlan_lookAheadDistLongerThanMap_validateException():
    """
    test the method _advance_on_plan
        test exception for too short map but nav_plan is long enough; validate the relevant exception
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = 20.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])[current_ordinal]
    wrong_road_id = 1234
    lookadhead_dist = 1000
    # test the case when the map is too short; validate the relevant exception
    try:
        MapUtils._advance_on_plan(starting_lane_id, starting_lon, lookahead_distance=lookadhead_dist,
                                  navigation_plan=NavigationPlanMsg(np.array(road_ids + [wrong_road_id])))
        assert False
    except DownstreamLaneNotFound:
        assert True


def test_getUpstreamLanesFromDistance_upstreamFiveOutOfTenSegments_validateLength():
    """
     test the method _get_upstream_lanes_from_distance
         validate that total length of output sub segments == lookahead_dist; validate lanes' ordinal
     """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    current_road_idx = 7
    current_ordinal = 1
    starting_lon = 20.
    backward_dist = 500.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])[current_ordinal]
    lane_ids, final_lon = MapUtils._get_upstream_lanes_from_distance(starting_lane_id, starting_lon, backward_dist)
    tot_length = starting_lon - final_lon
    # validate: total length of the segments equals to backward_dist and correctness of the segments' ordinal
    for lane_id in lane_ids[1:]:  # exclude the starting lane
        assert MapUtils.get_lane_ordinal(lane_id) == current_ordinal
        tot_length += MapUtils.get_lane_length(lane_id)
    assert np.isclose(tot_length, backward_dist)


def test_getUpstreamLanesFromDistance_smallBackwardDist_validateLaneIdAndLength():
    """
     test the method _get_upstream_lanes_from_distance
        test small backward_dist ending on the same lane; validate the same lane_id and final longitude
     """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    current_road_idx = 7
    current_ordinal = 1
    starting_lon = 20.
    starting_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])[current_ordinal]

    # test small backward_dist ending on the same lane
    small_backward_dist = 1
    lane_ids, final_lon = MapUtils._get_upstream_lanes_from_distance(starting_lane_id, starting_lon,
                                                                     backward_dist=small_backward_dist)
    assert lane_ids == [starting_lane_id]
    assert final_lon == starting_lon - small_backward_dist


def test_getUpstreamLanesFromDistance_backwardDistForFullMap_validateSegmentsNumberAndFinalLon():
    """
     test the method _get_upstream_lanes_from_distance
         try lookahead_dist until start of the map; validate there are no exceptions and segments number
     """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    current_ordinal = 1
    # test from the end until start of the map: verify no exception is thrown
    cumulative_distance = 0
    for road_id in road_ids:
        lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[current_ordinal]
        cumulative_distance += MapUtils.get_lane_length(lane_id)
    last_lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[-1])[current_ordinal]
    last_lane_length = MapUtils.get_lane_length(last_lane_id)
    lane_ids, final_lon = MapUtils._get_upstream_lanes_from_distance(last_lane_id, last_lane_length, cumulative_distance)
    # validate the number of segments and final longitude
    assert len(lane_ids) == len(road_ids)
    assert final_lon == 0


def test_getUpstreamLanesFromDistance_tooLongBackwardDist_validateRelevantException():
    """
     test the method _get_upstream_lanes_from_distance
         validate the relevant exception
     """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
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


def test_getClosestLane_multiLaneRoad_findRightestAndLeftestLanesByPoints():
    """
    test method get_closest_lane:
        find the most left and the most right lanes by points inside these lanes
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])
    # find the rightest lane
    lane_id = lane_ids[0]
    frenet = MapUtils.get_lane_frenet_frame(lane_id)
    closest_lane_id = MapUtils.get_closest_lane(frenet.points[1])
    assert lane_id == closest_lane_id
    # find the leftmost lane
    lane_id = lane_ids[-1]
    frenet = MapUtils.get_lane_frenet_frame(lane_id)
    closest_lane_id = MapUtils.get_closest_lane(frenet.points[-2])
    assert lane_id == closest_lane_id
    closest_lane_id = MapUtils.get_closest_lane(frenet.points[-2], road_ids[0])
    assert lane_id == closest_lane_id


def test_getClosestLane_multiLaneRoad_testExceptionOnWrongRoadId():
    """
    test method get_closest_lane:
        validate relevant exception on wrong road_segment_id
    """
    MapService.initialize(MAP_SPLIT)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])
    # find the rightest lane
    lane_id = lane_ids[0]
    frenet = MapUtils.get_lane_frenet_frame(lane_id)
    wrong_road_segment_id = 28
    try:
        MapUtils.get_closest_lane(frenet.points[-2], wrong_road_segment_id)
        assert False
    except RoadNotFound:
        assert True


def test_getLanesIdsFromRoadSegmentId_multiLaneRoad_validateIdsConsistency():
    """
    test method get_lanes_ids_from_road_segment_id
        validate consistency between road segment ids and lane ids
    """
    MapService.initialize(MAP_SPLIT)
    road_segment_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    road_segment_id = road_segment_ids[0]
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)
    assert len(lane_ids) == MapService.get_instance().get_road(road_segment_id).lanes_num
    assert road_segment_id == MapUtils.get_road_segment_id_from_lane_id(lane_ids[0])
    assert road_segment_id == MapUtils.get_road_segment_id_from_lane_id(lane_ids[-1])


def test_doesMapExistBackward_longBackwardDist_validateRelevantException():
    MapService.initialize(MAP_SPLIT)
    road_segment_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    road_segment_id = road_segment_ids[2]
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[0]
    assert MapUtils.does_map_exist_backward(lane_id, 200)
    assert not MapUtils.does_map_exist_backward(lane_id, 400)
