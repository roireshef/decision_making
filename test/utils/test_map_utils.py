import numpy as np
from typing import List
from unittest.mock import patch
import pytest
from decision_making.src.messages.scene_tcd_message import DynamicTrafficControlDeviceStatus
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame

from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.messages.scene_static_message import SceneStatic, TrafficControlBar, \
    StaticTrafficControlDevice, DynamicTrafficControlDevice
from decision_making.src.messages.scene_static_enums import NominalPathPoint, StaticTrafficControlDeviceType, \
    DynamicTrafficControlDeviceType, TrafficSignalState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RoadSignRestriction
from decision_making.src.scene.scene_traffic_control_devices_status_model import SceneTrafficControlDevicesStatusModel
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.exceptions import UpstreamLaneNotFound

from decision_making.test.planning.behavioral.behavioral_state_fixtures import \
    behavioral_grid_state_with_objects_for_filtering_too_aggressive, state_with_objects_for_filtering_too_aggressive, \
    route_plan_20_30, create_route_plan_msg, route_plan_lane_splits_on_left_and_right_left_first, \
    route_plan_lane_splits_on_left_and_right_right_first
from decision_making.test.planning.custom_fixtures import route_plan_1_2, route_plan_1_2_3
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split, right_lane_split_scene_static, \
    left_right_lane_split_scene_static, scene_static_short_testable, scene_static_left_lane_ends, scene_static_right_lane_ends, \
    left_lane_split_scene_static, scene_static_lane_split_on_left_ends, scene_static_lane_splits_on_left_and_right_left_first, \
    scene_static_lane_splits_on_left_and_right_right_first


MAP_SPLIT = "PG_split.bin"

STOP_BAR_ID = 1
STOP_SIGN_ID = 1
TRAFFIC_LIGHT_ID = 2


def _setup_stop_bars_in_map(scene_static: SceneStatic, gff: GeneralizedFrenetSerretFrame, static_tcds_id: List[int],
                            dynamic_tcds_id: List[int]) -> (int, int):

    # A Frenet-Frame trajectory: a numpy matrix of FrenetState2D [:, [FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA]]
    gff_state = np.array([[12.0, 0., 0., 0., 0., 0.]])
    lane_id, segment_states = gff.convert_to_segment_states(gff_state)
    segment_s = segment_states[0][0]

    for lane_segment in scene_static.s_Data.s_SceneStaticBase.as_scene_lane_segments:
        lane_segment.as_traffic_control_bar = []
    SceneStaticModel.get_instance().set_scene_static(scene_static)
    # add TCDs to the closest bar
    stop_bar1 = TrafficControlBar(e_i_traffic_control_bar_id=STOP_BAR_ID, e_l_station=segment_s - 2,
                                  e_i_static_traffic_control_device_id=static_tcds_id,
                                  e_i_dynamic_traffic_control_device_id=dynamic_tcds_id)
    stop_bar2 = TrafficControlBar(e_i_traffic_control_bar_id=2, e_l_station=segment_s - 1,
                                  e_i_static_traffic_control_device_id=[], e_i_dynamic_traffic_control_device_id=[])
    stop_bar3 = TrafficControlBar(e_i_traffic_control_bar_id=3, e_l_station=segment_s,
                                  e_i_static_traffic_control_device_id=[], e_i_dynamic_traffic_control_device_id=[])
    MapUtils.get_lane(lane_id).as_traffic_control_bar.append(stop_bar1)
    MapUtils.get_lane(lane_id).as_traffic_control_bar.append(stop_bar2)
    MapUtils.get_lane(lane_id).as_traffic_control_bar.append(stop_bar3)
    return lane_id, segment_s


def _setup_tcds_in_map(scene_static: SceneStatic, static_tcds: List[StaticTrafficControlDevice], dynamic_tcds: List[DynamicTrafficControlDevice]):
    scene_static.s_Data.s_SceneStaticBase.as_static_traffic_control_device = []
    for tcd in static_tcds:
        scene_static.s_Data.s_SceneStaticBase.as_static_traffic_control_device.append(tcd)
    scene_static.s_Data.s_SceneStaticBase.as_dynamic_traffic_control_device = []
    for tcd in dynamic_tcds:
        scene_static.s_Data.s_SceneStaticBase.as_dynamic_traffic_control_device.append(tcd)


def test_getTrafficControlBarsS_findsClosestStop(scene_static_pg_split,
                                                 behavioral_grid_state_with_objects_for_filtering_too_aggressive):
    gff = behavioral_grid_state_with_objects_for_filtering_too_aggressive.extended_lane_frames[RelativeLane.SAME_LANE]
    _setup_stop_bars_in_map(scene_static_pg_split, gff, [STOP_SIGN_ID], [TRAFFIC_LIGHT_ID])

    actual = MapUtils.get_traffic_control_bars_s(gff, 0)
    assert len(actual) == 3
    assert actual[0].s == 10.0
    assert actual[0].id == STOP_BAR_ID
    assert actual[1].s == 11.0
    assert actual[2].s == 12.0


def test_getTrafficControlBarsS_IgnoresTcbsBehindStartOffset(scene_static_pg_split,
                                                 behavioral_grid_state_with_objects_for_filtering_too_aggressive):
    gff = behavioral_grid_state_with_objects_for_filtering_too_aggressive.extended_lane_frames[RelativeLane.SAME_LANE]
    _setup_stop_bars_in_map(scene_static_pg_split, gff, [STOP_SIGN_ID], [TRAFFIC_LIGHT_ID])

    actual = MapUtils.get_traffic_control_bars_s(gff, 11.5)
    assert len(actual) == 1
    assert actual[0].s == 12.0


def test_getTrafficControlDevices_findsDevices(scene_static_pg_split,
                                                 behavioral_grid_state_with_objects_for_filtering_too_aggressive):
    gff = behavioral_grid_state_with_objects_for_filtering_too_aggressive.extended_lane_frames[RelativeLane.SAME_LANE]
    lane_id, segment_s = _setup_stop_bars_in_map(scene_static_pg_split, gff, [STOP_SIGN_ID], [TRAFFIC_LIGHT_ID])

    MapUtils.get_traffic_control_bars_s(gff, 0)

    stop_sign = StaticTrafficControlDevice(object_id=STOP_SIGN_ID,e_e_traffic_control_device_type=StaticTrafficControlDeviceType.STOP,
                               e_Pct_confidence=1.0, e_i_controlled_lane_segment_id=[lane_id], e_l_station=segment_s, e_l_lateral_offset=0)
    traffic_light = DynamicTrafficControlDevice(object_id=TRAFFIC_LIGHT_ID,e_e_traffic_control_device_type=DynamicTrafficControlDeviceType.TRAFFIC_LIGHT,
                               e_i_controlled_lane_segment_id=[lane_id], e_l_station=segment_s, e_l_lateral_offset=0)
    _setup_tcds_in_map(scene_static=scene_static_pg_split, static_tcds=[stop_sign], dynamic_tcds=[traffic_light])

    traffic_light_status = {traffic_light.object_id: DynamicTrafficControlDeviceStatus(
        e_i_dynamic_traffic_control_device_id=traffic_light.object_id, a_e_status=[TrafficSignalState.GREEN],
        a_Pct_status_confidence=[1.0])}
    SceneTrafficControlDevicesStatusModel.get_instance().set_traffic_control_devices_status(traffic_control_devices_status=traffic_light_status)

    static_tcds, dynamic_tcds = MapUtils.get_traffic_control_devices()
    assert len(static_tcds) == 1
    assert static_tcds[STOP_SIGN_ID].id == 1
    assert static_tcds[STOP_SIGN_ID].sign_type == StaticTrafficControlDeviceType.STOP
    assert len(dynamic_tcds) == 1
    assert dynamic_tcds[TRAFFIC_LIGHT_ID].id == 2
    assert dynamic_tcds[TRAFFIC_LIGHT_ID].sign_type == DynamicTrafficControlDeviceType.TRAFFIC_LIGHT
    assert dynamic_tcds[TRAFFIC_LIGHT_ID].status[0] == TrafficSignalState.GREEN


def test_getTCDsForBar_findsStaticAndDynamicDevicesAndAssociatedRestrictions(scene_static_pg_split,
                                                 behavioral_grid_state_with_objects_for_filtering_too_aggressive):
    gff = behavioral_grid_state_with_objects_for_filtering_too_aggressive.extended_lane_frames[RelativeLane.SAME_LANE]
    lane_id, segment_s = _setup_stop_bars_in_map(scene_static_pg_split, gff, [STOP_SIGN_ID], [TRAFFIC_LIGHT_ID])

    actual = MapUtils.get_traffic_control_bars_s(gff, 0)

    stop_sign = StaticTrafficControlDevice(object_id=STOP_SIGN_ID,e_e_traffic_control_device_type=StaticTrafficControlDeviceType.STOP,
                               e_Pct_confidence=1.0, e_i_controlled_lane_segment_id=[lane_id], e_l_station=segment_s, e_l_lateral_offset=0)
    traffic_light = DynamicTrafficControlDevice(object_id=TRAFFIC_LIGHT_ID,e_e_traffic_control_device_type=DynamicTrafficControlDeviceType.TRAFFIC_LIGHT,
                               e_i_controlled_lane_segment_id=[lane_id], e_l_station=segment_s, e_l_lateral_offset=0)
    _setup_tcds_in_map(scene_static=scene_static_pg_split, static_tcds=[stop_sign], dynamic_tcds=[traffic_light])

    # dynamic TCDs status should be set before TCDs are retrieved
    traffic_light_status = {traffic_light.object_id: DynamicTrafficControlDeviceStatus(
        e_i_dynamic_traffic_control_device_id=traffic_light.object_id, a_e_status=[TrafficSignalState.GREEN],
        a_Pct_status_confidence=[1.0])}
    SceneTrafficControlDevicesStatusModel.get_instance().set_traffic_control_devices_status(traffic_control_devices_status=traffic_light_status)
    static_tcds, dynamic_tcds = MapUtils.get_traffic_control_devices()

    active_static_tcds, active_dynamic_tcds = MapUtils.get_TCDs_for_bar(actual[0], static_tcds, dynamic_tcds)
    assert len(active_static_tcds) == 1
    assert len(active_dynamic_tcds) == 1

    road_signs_restriction = MapUtils.resolve_restriction_of_road_sign(active_static_tcds, active_dynamic_tcds)
    assert road_signs_restriction == RoadSignRestriction.STOP

    should_stop = MapUtils.should_stop_at_stop_bar(road_signs_restriction)
    assert should_stop


def test_getTCDsForBar_findsStaticOnlyDevicesAndAssociatedRestrictions(scene_static_pg_split,
                                                 behavioral_grid_state_with_objects_for_filtering_too_aggressive):
    gff = behavioral_grid_state_with_objects_for_filtering_too_aggressive.extended_lane_frames[RelativeLane.SAME_LANE]
    lane_id, segment_s = _setup_stop_bars_in_map(scene_static_pg_split, gff, [STOP_SIGN_ID],[])

    actual = MapUtils.get_traffic_control_bars_s(gff, 0)

    stop_sign = StaticTrafficControlDevice(object_id=STOP_SIGN_ID,e_e_traffic_control_device_type=StaticTrafficControlDeviceType.STOP,
                               e_Pct_confidence=1.0, e_i_controlled_lane_segment_id=[lane_id], e_l_station=segment_s, e_l_lateral_offset=0)
    _setup_tcds_in_map(scene_static=scene_static_pg_split, static_tcds=[stop_sign], dynamic_tcds=[])

    SceneTrafficControlDevicesStatusModel.get_instance().set_traffic_control_devices_status(traffic_control_devices_status={})
    static_tcds, dynamic_tcds = MapUtils.get_traffic_control_devices()

    active_static_tcds, active_dynamic_tcds = MapUtils.get_TCDs_for_bar(actual[0], static_tcds, dynamic_tcds)
    assert len(active_static_tcds) == 1
    assert len(active_dynamic_tcds) == 0

    road_signs_restriction = MapUtils.resolve_restriction_of_road_sign(active_static_tcds, active_dynamic_tcds)
    assert road_signs_restriction == RoadSignRestriction.STOP

    should_stop = MapUtils.should_stop_at_stop_bar(road_signs_restriction)
    assert should_stop


def test_getTCDsForBar_findsDynamicOnlyDevicesAndAssociatedRestrictions(scene_static_pg_split,
                                                 behavioral_grid_state_with_objects_for_filtering_too_aggressive):
    gff = behavioral_grid_state_with_objects_for_filtering_too_aggressive.extended_lane_frames[RelativeLane.SAME_LANE]
    lane_id, segment_s = _setup_stop_bars_in_map(scene_static_pg_split, gff, [], [TRAFFIC_LIGHT_ID])

    actual = MapUtils.get_traffic_control_bars_s(gff, 0)

    traffic_light = DynamicTrafficControlDevice(object_id=TRAFFIC_LIGHT_ID,e_e_traffic_control_device_type=DynamicTrafficControlDeviceType.TRAFFIC_LIGHT,
                               e_i_controlled_lane_segment_id=[lane_id], e_l_station=segment_s, e_l_lateral_offset=0)
    _setup_tcds_in_map(scene_static=scene_static_pg_split, static_tcds=[], dynamic_tcds=[traffic_light])

    # dynamic TCDs status should be set before TCDs are retrieved
    traffic_light_status = {traffic_light.object_id: DynamicTrafficControlDeviceStatus(
        e_i_dynamic_traffic_control_device_id=traffic_light.object_id, a_e_status=[TrafficSignalState.GREEN],
        a_Pct_status_confidence=[1.0])}
    SceneTrafficControlDevicesStatusModel.get_instance().set_traffic_control_devices_status(traffic_control_devices_status=traffic_light_status)
    static_tcds, dynamic_tcds = MapUtils.get_traffic_control_devices()

    active_static_tcds, active_dynamic_tcds = MapUtils.get_TCDs_for_bar(actual[0], static_tcds, dynamic_tcds)
    assert len(active_static_tcds) == 0
    assert len(active_dynamic_tcds) == 1

    road_signs_restriction = MapUtils.resolve_restriction_of_road_sign(active_static_tcds, active_dynamic_tcds)
    assert road_signs_restriction == RoadSignRestriction.NONE

    should_stop = MapUtils.should_stop_at_stop_bar(road_signs_restriction)
    assert not should_stop


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


