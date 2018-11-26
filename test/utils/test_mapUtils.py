from unittest.mock import patch

import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.state.state import DynamicObject
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.src.service.map_service import MapService
from mapping.test.model.testable_map_fixtures import map_api_mock
from decision_making.test.planning.custom_fixtures import dyn_obj_outside_road, dyn_obj_on_road
from decision_making.src.planning.types import FS_SX, FS_DX, FP_SX, FP_DX


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_isObjectOnRoad_objectOffOfRoad_False(dyn_obj_outside_road: DynamicObject):
    """
    Checking functionality of _is_object_on_road for an object that is off the road.
    """
    actual_result = MapUtils.is_object_on_road(dyn_obj_outside_road.map_state)
    expected_result = False
    assert expected_result == actual_result


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_isObjectOnRoad_objectOnRoad_True(dyn_obj_on_road: DynamicObject):
    """
    Checking functionality of _is_object_on_road for an object that is on the road.
    """
    actual_result = MapUtils.is_object_on_road(dyn_obj_on_road.map_state)
    expected_result = True
    assert expected_result == actual_result


def test_getAdjacentLanes_adjacentOfRightestAndSecondLanes_accurate():
    """
    test method get_adjacent_lanes for the current map;
    check adjacent lanes of the rightest and the second-from-right lanes
    """
    MapService.initialize()
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
    test method get_dist_from_lane_center_to_lane_borders:
        in the current map the lanes have a constant lane width and all lanes have the same width;
        therefore it should return half lane width
    """
    MapService.initialize()
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
    MapService.initialize()
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])
    dist_to_right, dist_to_left = MapUtils.get_dist_to_road_borders(lane_ids[0], 0)
    lane_width = MapService.get_instance().get_road(road_ids[0]).lane_width
    assert dist_to_right == lane_width/2
    assert dist_to_left == lane_width * (len(lane_ids) - 0.5)


def test_getLongitudinalDistance():
    """
    test method get_longitudinal_distance:
        validate distance between two points on different road segments
    """
    MapService.initialize()
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    road_idx1 = 1
    road_idx2 = 7
    ordinal = 1
    lon1 = 10.
    lon2 = 20.
    lane_id1 = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[road_idx1])[ordinal]
    lane_id2 = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[road_idx2])[ordinal]
    cumulative_distance = -lon1 + lon2
    for rid in range(road_idx1, road_idx2):
        lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[rid])[ordinal]
        cumulative_distance += MapUtils.get_lane_length(lane_id)
    dist = MapUtils.get_longitudinal_distance(lane_id1, lane_id2, lon1, lon2, max_depth=road_idx2-road_idx1)
    assert np.isclose(dist, cumulative_distance)


def test_getLateralDistanceInLaneUnits_lanesFromDifferentRoadSegments_accordingToOrdinals():
    """
    test method get_lateral_distance_in_lane_units:
        the lateral distance in lane units between two lanes on different road segments
    """
    road_idx1 = 1
    road_idx2 = 7
    ordinal1 = 0
    ordinal2 = 2
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_id1 = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[road_idx1])[ordinal1]
    lane_id2 = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[road_idx2])[ordinal2]
    assert MapUtils.get_lateral_distance_in_lane_units(lane_id1, lane_id2, max_depth=road_idx2-road_idx1) == \
           ordinal2 - ordinal1


def test_getLookaheadFrenetFrame_frenetStartsBehindAndEndsAheadOfCurrentLane_accurateFrameStartAndLength():
    """
    test method get_lookahead_frenet_frame:
        the current map has only one road segment;
        the frame starts and ends on arbitrary points.
    verify that final length, offset of GFF and conversion of an arbitrary point are accurate
    """
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    current_road_idx = 3
    current_ordinal = 1
    starting_lon = -200.
    lookahead_dist = 500.
    small_dist_err = 0.01
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[current_road_idx])
    lane_id = lane_ids[current_ordinal]
    gff = MapUtils.get_lookahead_frenet_frame(lane_id, starting_lon, lookahead_dist, NavigationPlanMsg(np.array(road_ids)))
    # validate the length of the obtained frenet frame
    assert abs(gff.s_max - lookahead_dist) < small_dist_err
    # calculate cartesian state of the origin of lane_id using GFF and using original frenet of lane_id and compare them
    gff_cpoint = gff.fpoint_to_cpoint(np.array([-starting_lon, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(lane_id).fpoint_to_cpoint(np.array([0, 0]))
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < small_dist_err
    # calculate cartesian state of some point using GFF and using original frenet (from the map) and compare them
    fpoint = np.array([450., 1.])
    gff_cpoint = gff.fpoint_to_cpoint(fpoint)
    segment_id, segment_fstate = gff.convert_to_segment_state(np.array([fpoint[FP_SX], 0, 0, fpoint[FP_DX], 0, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(segment_id).fpoint_to_cpoint(segment_fstate[[FS_SX, FS_DX]])
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < small_dist_err


def test_getClosestLane_multiLaneRoad_findRightestAndLeftestLanesByPoints():
    """
    test method get_closest_lane:
        find the most left and the most right lanes by points inside these lanes
    """
    MapService.initialize()
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


def test_getLanesIdsFromRoadSegmentId_multiLaneRoad_validateIdsConsistency():
    """
    test method get_lanes_ids_from_road_segment_id
        validate consistency between road segment ids and lane ids
    """
    MapService.initialize()
    road_segment_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    road_segment_id = road_segment_ids[0]
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)
    assert len(lane_ids) == MapService.get_instance().get_road(road_segment_id).lanes_num
    assert road_segment_id == MapUtils.get_road_segment_id_from_lane_id(lane_ids[0])
    assert road_segment_id == MapUtils.get_road_segment_id_from_lane_id(lane_ids[-1])
