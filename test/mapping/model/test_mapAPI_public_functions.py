import pytest
import os
import pickle
import numpy as np
from decision_making.src.global_constants import DM_MANAGER_NAME_FOR_LOGGING
from decision_making.src.mapping.model.localization import RoadCoordinatesDifference, RoadLocalization
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.map_exceptions import MapCellNotFound, LongitudeOutOfRoad, RoadNotFound, LaneNotFound
from decision_making.src.mapping.model.map_api import MapAPI
from rte.python.logger.AV_logger import AV_Logger
import common_data as COMMON_DATA_ROOT
from decision_making.test.mapping.model.testable_map_fixtures import map_api_mock, ROAD_WIDTH, MAP_INFLATION_FACTOR, \
    navigation_fixture, testable_map_api


@pytest.fixture()
def map_fixture():
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)

    map_model_filename = os.path.join(os.path.dirname(__file__), 'TestMap.bin')
    map_model = pickle.load(open(map_model_filename, "rb"))

    trans_roads_data = {}
    for rid, road in map_model._roads_data.items():
        trans_roads_data[rid] = road
    map_model._roads_data = trans_roads_data
    yield MapAPI(map_model, logger)


def test_convertGlobalToRoadCoordinates_OutOfRoad_exception(testable_map_api):
    try:
        map_fixture = testable_map_api
        map_fixture.convert_global_to_road_coordinates(-1.0, 0.0, np.pi / 4.0)
        # must get out-of-road exception
        assert False
    except (MapCellNotFound, LongitudeOutOfRoad) as e:
        assert True

def test_convertGeoToMapCoordinates_MapWithFrameOriginAndGivenCoordinates_ReturnsCorrectCoordinates(testable_map_api):

    # Frame origin is in (0,0) map coordinates
    frame_origin = testable_map_api._cached_map_model._frame_origin
    frame_origin_x, frame_origin_y, utm_zone = testable_map_api.convert_geo_to_map_coordinates(frame_origin[0], frame_origin[1])
    assert (frame_origin_x, frame_origin_y) == (0.0, -0.0)

    # A location that has lat,lon > frame origin will have x>0, y<0
    pos_x, pos_y, utm_zone = testable_map_api.convert_geo_to_map_coordinates(
        frame_origin[0] + 0.1, frame_origin[1] + 0.1)
    assert (pos_x, pos_y) == pytest.approx((11176.977744052187, -9333.2862594889011))

    # Edge case - conversion of Lat/Long (0,0)
    zero_x, zero_y, utm_zone = testable_map_api.convert_geo_to_map_coordinates(0, 0)
    assert (zero_x, zero_y) == pytest.approx((-3540872.5313944165, 428436.02022310655))


# def test_convertGlobalToRoadCoordinates_OnRoad_exact(testable_map_api):
#     map_fixture = testable_map_api
#
#     # Check that a point on the first road is exactly localized
#     closest_road_id, lon, lat, yaw, is_within_road_latitudes = map_fixture.convert_global_to_road_coordinates(10.0, 0.0,
#                                                                                                          np.pi / 4.0)
#
#     assert closest_road_id == 1
#     assert lon == 10.0
#     assert lat == ROAD_WIDTH / 2.0
#     assert yaw == np.pi / 4.0
#     assert is_within_road_latitudes
#
#     # Check that a point on the second road is exactly localized
#     closest_road_id, lon, lat, yaw, is_within_road_latitudes = map_fixture.convert_global_to_road_coordinates(
#         MAP_INFLATION_FACTOR - 10.0, MAP_INFLATION_FACTOR, np.pi + np.pi/6.0)
#
#     assert closest_road_id == 2
#     assert lon == 10.0
#     assert lat == ROAD_WIDTH / 2.0
#     assert np.math.isclose(yaw, np.pi / 6.0)
#     assert is_within_road_latitudes

# TODO: should be tested to work
# def test_computeRoadLocalization_OnRoad_exact(testable_map_api):
#     road_local: RoadLocalization = testable_map_api.compute_road_localization(
#         np.array([10.0, 0.0]), np.pi / 4.0)
#
#     assert road_local.road_id == 1
#     assert road_local.road_lon == 10.0
#     assert road_local.intra_road_lat == ROAD_WIDTH / 2.0
#     assert road_local.intra_road_yaw == np.pi / 4.0
#
#     # Check that a point on the second road is exactly localized
#     road_local: RoadLocalization = testable_map_api.compute_road_localization(
#         np.array([MAP_INFLATION_FACTOR - 10.0, MAP_INFLATION_FACTOR]), np.pi + np.pi/6.0)
#
#     assert road_local.road_id == 2
#     assert road_local.road_lon == 10.0
#     assert road_local.intra_road_lat == ROAD_WIDTH / 2.0
#     assert np.math.isclose(road_local.intra_road_yaw, np.pi / 6.0)

# TODO: should be tested to work
def test_computeRoadLocalizationsDiff_OnRoad_exact(testable_map_api):
    road_local: RoadLocalization = testable_map_api.compute_road_localization(
        np.array([10.0, 0.0]), np.pi / 4.0)

    ego_road_local = RoadLocalization(1, 0, 0, 0, 5.0, np.pi/2)
    object_road_local = RoadLocalization(1, 0, 0.5, 0.5, 10.0, np.pi/4)

    realtive_local: RoadCoordinatesDifference = testable_map_api.compute_road_localizations_diff(
        reference_localization=ego_road_local,
        object_localization=object_road_local,
        navigation_plan=NavigationPlanMsg(np.array([1]))
    )

    assert realtive_local.rel_lat == 0.5
    assert realtive_local.rel_lon == 5.0
    assert realtive_local.rel_yaw == -np.pi/4


# def test_convertGlobalToRoadCoordinates_OutOfRoadLat_precise(testable_map_api):
#     map_fixture = testable_map_api
#     closest_road_id, lon, lat, yaw, is_within_road_latitudes = map_fixture.convert_global_to_road_coordinates(10.0, -6.0, np.pi / 4.0)
#
#     assert closest_road_id == 1
#     assert lon == 10.0
#     assert lat == ROAD_WIDTH / 2.0 - 6.0
#     assert yaw == np.pi / 4.0
#     assert not is_within_road_latitudes


# def test_convertGlobalToRoadCoordinates_TestPointInFunnel_precise(testable_map_api):
#     # special case where the point is in the funnel that is created by the normals of two segments
#     map_fixture = testable_map_api
#     closest_road_id, lon, lat, yaw, is_within_road_latitudes = map_fixture.convert_global_to_road_coordinates(
#         2.0*MAP_INFLATION_FACTOR + 1.0, 0 - 1.0, np.pi / 4.0)
#
#     assert closest_road_id == 1
#     assert lon == 2.0 * MAP_INFLATION_FACTOR
#     # We are projected on the end of the first segment, therefore our lat, yaw are relative to it
#     assert np.math.isclose(lat, ROAD_WIDTH / 2.0 - np.sqrt(2.0))
#     assert np.math.isclose(yaw, np.pi/2.0 - np.pi / 4.0)
#     assert is_within_road_latitudes


def test_convertGlobalToRoadCoordinates_OutOfRoadLat_mapCellNotFoundException(testable_map_api):
    try:
        map_fixture = testable_map_api
        closest_road_id, lon, lat, yaw, is_within_road_latitudes, is_within_road_longitudes = \
            map_fixture.convert_global_to_road_coordinates(10.0, -16.0, np.pi/4.0)

    except MapCellNotFound as e:
        assert True


def test_findRoadsContainingPoint_testDifferentPointsOnTwoRoad(map_fixture):
    correct_road_ids_list = [1, 2]
    road_ids = map_fixture._find_roads_containing_point(-30, 50)
    np.testing.assert_array_equal(correct_road_ids_list, road_ids)

    correct_road_ids_list = [1]
    road_ids = map_fixture._find_roads_containing_point(-20, 50)
    np.testing.assert_array_equal(correct_road_ids_list, road_ids)

    correct_road_ids_list = [1, 2]
    road_ids = map_fixture._find_roads_containing_point(30, -50)
    np.testing.assert_array_equal(correct_road_ids_list, road_ids)

    correct_road_ids_list = [2]
    road_ids = map_fixture._find_roads_containing_point(-20, -50)
    np.testing.assert_array_equal(correct_road_ids_list, road_ids)


def test_getCenterLanesLatitudes_checkLatitudesOfAllLanes(map_fixture):
    road_id = 1
    lanes_num = map_fixture.get_road(road_id).lanes_num
    width = map_fixture.get_road(road_id).road_width
    lane_wid = width / lanes_num
    correct_lat_list = np.arange(lane_wid / 2, width, lane_wid)
    lanes_lat = map_fixture.get_center_lanes_latitudes(road_id)
    np.testing.assert_array_almost_equal(correct_lat_list, lanes_lat)


# TODO: expect certain values
def test_getPointRelativeLongitude_differentRoads(map_fixture):
    navigation_plan = NavigationPlanMsg(np.array([1, 2]))
    length = map_fixture.get_road(1).length
    from_lon_in_road = 20
    to_lon_in_road = 10
    total_lon_distance = map_fixture.get_longitudinal_difference(initial_road_id=1, initial_lon=from_lon_in_road,
                                                                 final_road_id=2, final_lon=to_lon_in_road,
                                                                 navigation_plan=navigation_plan)
    assert total_lon_distance == length - from_lon_in_road + to_lon_in_road


def test_getPointRelativeLongitude_sameRoad(map_fixture):
    navigation_plan = NavigationPlanMsg(np.array([1, 2]))
    from_lon_in_road = 10
    to_lon_in_road = 170
    total_lon_distance = map_fixture.get_longitudinal_difference(initial_road_id=2, initial_lon=from_lon_in_road,
                                                                 final_road_id=2, final_lon=to_lon_in_road,
                                                                 navigation_plan=navigation_plan)
    assert to_lon_in_road - from_lon_in_road - 10 < total_lon_distance < to_lon_in_road - from_lon_in_road + 10


# def test_GetPathLookahead_almostUntilEndOfUpperHorizontalPath_validatePathLengthAndConstantY(map_fixture,
#                                                                                              navigation_fixture):
#     points = map_fixture.get_road(1)._points
#     width = map_fixture.get_road(1).road_width
#     navigation_plan = navigation_fixture
#     lookahead_distance = 49.9  # test short path
#     road_id = 1
#     lon = 10
#     lat = 2  # right lane
#     path1, _ = map_fixture.get_lookahead_points(road_id, lon, lookahead_distance, lat, navigation_plan)
#     path_x = path1[:, 0]
#     path_y = path1[:, 1]
#     assert path_x[0] == -30 + lon
#     np.testing.assert_almost_equal(path_x[-1], -30 + lon + lookahead_distance)
#
#     y = points[0, 1] - width / 2 + lat
#     for p in range(path1.shape[0]):
#         assert path_y[p] == y


def test_getPathLookahead_testLongPathOnTwoRoadsOnRightLane_validateLengthIsABitLessThanLookaheadDistance(map_fixture,
                                                                                                          navigation_fixture):
    navigation_plan = navigation_fixture
    road_id = 1
    lon = 10
    lat = 2  # right lane
    lookahead_distance = 200  # test path including two road_ids
    path2, _ = map_fixture.get_lookahead_points(road_id, lon, lookahead_distance, lat, navigation_plan)
    segments = np.diff(path2, axis=0)
    segments_norm = np.linalg.norm(segments, axis=1)
    path_length_small_radius = np.sum(segments_norm)
    assert lookahead_distance - 10 < path_length_small_radius < lookahead_distance


def test_getPathLookahead_testLongPathOnTwoRoadsOnLeftLane_validateLengthIsABitMoreThanLookaheadDistance(map_fixture,
                                                                                                         navigation_fixture):
    navigation_plan = navigation_fixture
    road_id = 1
    lon = 10
    lat = 4  # left lane
    lookahead_distance = 200  # test path including two road_ids
    path3, _ = map_fixture.get_lookahead_points(road_id, lon, lookahead_distance, lat, navigation_plan)
    segments = np.diff(path3, axis=0)
    segments_norm = np.linalg.norm(segments, axis=1)
    path_length_large_radius = np.sum(segments_norm)
    assert lookahead_distance < path_length_large_radius < lookahead_distance + 10


# def test_convertRoadToGlobalCoordinates_simpleRoad_accurateConversion(testable_map_api):
#     lon_of_first_segment = 10.2
#     point_on_first_segment = np.array([10.2, -ROAD_WIDTH / 2])
#     point_lateral_shift = 9.
#     right_edge_position, world_yaw = testable_map_api.convert_road_to_global_coordinates(road_id=1, lat=0.0,
#                                                                                          lon=lon_of_first_segment)
#     shifted_world_position, world_yaw = testable_map_api.convert_road_to_global_coordinates(road_id=1,
#                                                                                             lat=point_lateral_shift,
#                                                                                             lon=lon_of_first_segment)
#
#     np.testing.assert_array_almost_equal(right_edge_position[0:2], point_on_first_segment)
#     np.testing.assert_array_almost_equal(shifted_world_position[0:2],
#                                          point_on_first_segment + np.array([0., point_lateral_shift]))
#
#     lon_of_second_segment = MAP_INFLATION_FACTOR * 2 + 10.2
#     point_on_second_segment = np.array([MAP_INFLATION_FACTOR * 2 + ROAD_WIDTH / 2, 10.2])
#     point_lateral_shift = 9.
#     right_edge_position, world_yaw = testable_map_api.convert_road_to_global_coordinates(road_id=1, lat=0.0,
#                                                                                          lon=lon_of_second_segment)
#     shifted_world_position, world_yaw = testable_map_api.convert_road_to_global_coordinates(road_id=1,
#                                                                                             lat=point_lateral_shift,
#                                                                                             lon=lon_of_second_segment)
#
#     np.testing.assert_array_almost_equal(right_edge_position[0:2], point_on_second_segment)
#     np.testing.assert_array_almost_equal(shifted_world_position[0:2],
#                                          point_on_second_segment + np.array([-point_lateral_shift, 0.]))
#
#     # Uncomment below to see the road structure
#     # road = testable_map_api._cached_map_model.get_road_data(1)
#     # road_points = road.points
#     # plt.plot(road_points[:, 0], road_points[:, 1], '-b')
#     # plt.plot(right_edge_position[0], right_edge_position[1], '*c')
#     # plt.plot(shifted_world_position[0], shifted_world_position[1], '*r')
#     # plt.show()


def test_getLaneWidth_UnknownRoadId_RaisesRoadIdException(testable_map_api):
    with pytest.raises(RoadNotFound):
        testable_map_api.get_lane_width(4, 0, 0)

def test_getLaneWidth_UnknownLane_RaisesLaneIdException(testable_map_api):
    with pytest.raises(LaneNotFound):
        testable_map_api.get_lane_width(1, -1, 0)

    with pytest.raises(LaneNotFound):
        testable_map_api.get_lane_width(1, 10, 0)

def test_getLaneWidth_LongitudeOutOfRange_RaisesLongitudeException(testable_map_api):
    with pytest.raises(LongitudeOutOfRoad):
        testable_map_api.get_lane_width(1, 0, -1)

    with pytest.raises(LongitudeOutOfRoad):
        testable_map_api.get_lane_width(1, 0, 2000)

def test_getLaneWidth_ValidValues_ExpectedWidth(testable_map_api):
    assert testable_map_api.get_lane_width(1, 0, 0), map_fixture.get_road(0)._lane_width

