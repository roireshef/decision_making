import numpy as np
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.utils.map_utils import MapUtils
from mapping.src.service.map_service import MapService


def test_getAdjacentLanes_adjacentOfRightestAndSecondLanes_accurate():
    """
    test method get_adjacent_lanes for the current map;
    check adjacent lanes of the rightest and the second-from-right lanes
    """
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])
    right_to_rightest = MapUtils.get_adjacent_lanes(lane_ids[0], RelativeLane.RIGHT_LANE)
    assert len(right_to_rightest) == 0
    left_to_rightest = MapUtils.get_adjacent_lanes(lane_ids[0], RelativeLane.LEFT_LANE)
    assert left_to_rightest == lane_ids[1:]
    right_to_second = MapUtils.get_adjacent_lanes(lane_ids[1], RelativeLane.RIGHT_LANE)
    assert right_to_second == [lane_ids[0]]
    left_to_second = MapUtils.get_adjacent_lanes(lane_ids[1], RelativeLane.LEFT_LANE)
    assert left_to_second == lane_ids[2:]


def test_getDistFromLaneCenterToLaneBorders_rightLane_equalToHalfLaneWidth():
    """
    test method get_dist_from_lane_center_to_lane_borders:
        in the current map the lanes have a constant lane width and all lanes have the same width;
        therefore it should return half lane width
    """
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])
    dist_to_right, dist_to_left = MapUtils.get_dist_from_lane_center_to_lane_borders(lane_ids[0], 0)
    assert dist_to_right == dist_to_left
    assert dist_to_right == MapService.get_instance().get_road(road_ids[0]).lane_width/2


def test_getDistFromLaneCenterToRoadBorders_rightLane_equalToDistFromRoadBorder():
    """
    test method get_dist_from_lane_center_to_road_borders:
        in the current map the lanes have a constant lane width and all lanes have the same width
    """
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])
    dist_to_right, dist_to_left = MapUtils.get_dist_from_lane_center_to_road_borders(lane_ids[0], 0)
    lane_width = MapService.get_instance().get_road(road_ids[0]).lane_width
    assert dist_to_right == lane_width/2
    assert dist_to_left == lane_width * (len(lane_ids) - 0.5)


def test_getLateralDistanceInLaneUnits_rightestFromLeftmost_equalToLanesNumMinusOne():
    """
    test method get_lateral_distance_in_lane_units:
        in the current map the lanes have a constant lane width and all lanes have the same width;
    the distance from the rightest to the leftmost should be equal to num_lanes - 1
    """
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])
    assert MapUtils.get_lateral_distance_in_lane_units(lane_ids[0], lane_ids[-1]) == len(lane_ids)-1


def test_getLookaheadFrenetFrame_frenetStartsBehindAndEndsAheadOfCurrentLane_accurateFrameStartAndLength():
    """
    test method get_lookahead_frenet_frame:
        the current map has only one road segment;
        the frame starts and ends on arbitrary points.
    verify that final length and offset of GFF are accurate
    """
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[3])
    lane_id = lane_ids[0]
    starting_lon = -200
    lookahead_dist = 500
    small_dist_err = 0.1
    gff = MapUtils.get_lookahead_frenet_frame(lane_id, starting_lon, lookahead_dist, NavigationPlanMsg(np.array(road_ids)))
    assert abs(gff.s_max - lookahead_dist) < small_dist_err
    # calculate the origin of lane_id using GFF and using its own frenet
    gff_cpoint = gff.fpoint_to_cpoint(np.array([-starting_lon, 0]))
    ff_cpoint = MapUtils.get_lane_frenet_frame(lane_id).fpoint_to_cpoint(np.array([0, 0]))
    assert np.linalg.norm(gff_cpoint - ff_cpoint) < small_dist_err
