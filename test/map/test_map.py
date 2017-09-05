import pytest

from rte.python.logger.AV_logger import AV_Logger
from spav.decision_making.paths import Paths
from spav.decision_making.src.map.map_api import *
from spav.decision_making.src.map.naive_cache_map import NaiveCacheMap

import pickle

@pytest.fixture()
def map_fixture():
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)

    map_model_filename = Paths.get_resource_absolute_path_filename("maps/CustomMapPickle.bin")
    map_model = pickle.load(open(map_model_filename, "rb"))

    # TODO: temporal fix
    map_model.xy2road_tile_size = ROADS_MAP_TILE_SIZE
    trans_roads_data = {}
    for rid, road in map_model.roads_data.items():
        road.points = np.transpose(road.points)
        trans_roads_data[rid] = road
    map_model.roads_data = trans_roads_data
    yield NaiveCacheMap(map_model, logger)


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
    lanes_num = map_fixture._get_road(road_id).lanes_num
    width = map_fixture._get_road(road_id).width
    lane_wid = width / lanes_num
    correct_lat_list = np.arange(lane_wid / 2, width, lane_wid)
    lanes_lat = map_fixture.get_center_lanes_latitudes(road_id)
    np.testing.assert_array_almost_equal(correct_lat_list, lanes_lat)

# TODO: expect certain values
def test_getPointRelativeLongitude_differentRoads(map_fixture):
    navigation_plan = NavigationPlanMsg([1, 2])
    length = map_fixture._get_road(1).longitudes[-1]
    from_lon_in_road = 20
    to_lon_in_road = 10
    total_lon_distance = map_fixture.get_longitudinal_difference(init_road_id=1, init_lon=from_lon_in_road,
                                                                 final_road_id=2, final_lon=to_lon_in_road,
                                                                 navigation_plan=navigation_plan)
    assert total_lon_distance == length - from_lon_in_road + to_lon_in_road

def test_getPointRelativeLongitude_sameRoad(map_fixture):
    navigation_plan = NavigationPlanMsg([1, 2])
    from_lon_in_road = 10
    to_lon_in_road = 170
    total_lon_distance = map_fixture.get_longitudinal_difference(init_road_id=2, init_lon=from_lon_in_road,
                                                                 final_road_id=2, final_lon=to_lon_in_road,
                                                                 navigation_plan=navigation_plan)
    assert to_lon_in_road - from_lon_in_road - 10 < total_lon_distance < to_lon_in_road - from_lon_in_road + 10


def test_GetPathLookahead_upperPath(map_fixture):
    points = map_fixture._get_road(1).points
    width = map_fixture._get_road(1).width
    navigation_plan = NavigationPlanMsg([1, 2])
    lookahead_distance = 50  # test short path
    road_id = 1
    lon = 10
    lat = 2  # right lane
    path1 = map_fixture.get_lookahead_points(road_id, lon, lookahead_distance, lat, navigation_plan)
    path_x = path1[0]
    path_y = path1[1]
    assert path_x[0] == -30 + lon
    assert path_x[-1] == -30 + lon + lookahead_distance
    for p in range(path1.shape[1]):
        assert path_y[p] == points[1][0] - width / 2 + lat

def test_GetPathLookahead_testPathOnTwoRoadsOnRightLane(map_fixture):
    navigation_plan = NavigationPlanMsg([1, 2])
    road_id = 1
    lon = 10
    lat = 2  # right lane
    lookahead_distance = 200  # test path including two road_ids
    path2 = map_fixture.get_path_lookahead(road_id, lon, lat, lookahead_distance, navigation_plan)
    segments = np.diff(path2, axis=1)
    segments_norm = np.linalg.norm(segments, axis=0)
    path_length_small_radius = np.sum(segments_norm)
    assert lookahead_distance - 10 < path_length_small_radius < lookahead_distance

def test_GetPathLookahead_testPathOnTwoRoadsOnLeftLane(map_fixture):
    navigation_plan = NavigationPlanMsg([1, 2])
    road_id = 1
    lon = 10
    lat = 4  # left lane
    lookahead_distance = 200  # test path including two road_ids
    path3 = map_fixture.get_path_lookahead(road_id, lon, lat, lookahead_distance, navigation_plan)
    segments = np.diff(path3, axis=1)
    segments_norm = np.linalg.norm(segments, axis=0)
    path_length_large_radius = np.sum(segments_norm)
    assert lookahead_distance < path_length_large_radius < lookahead_distance + 10


def test_getUniformPathLookahead_testUniformityOnTwoRoadsOnRightLane(map_fixture):
    navigation_plan = NavigationPlanMsg([1, 2])
    road_id = 1
    lon = 10
    lat = 2
    lon_step = 10
    steps_num = 20
    max_lookahead_distance = lon_step * steps_num
    path = map_fixture.get_uniform_path_lookahead(road_id, lat, lon, lon_step=lon_step, steps_num=steps_num,
                                                  navigation_plan=navigation_plan)
    segments = np.diff(path, axis=1)
    segments_norm = np.linalg.norm(segments, axis=0)
    path_length_small_radius = np.sum(segments_norm)
    assert max_lookahead_distance - 10 < path_length_small_radius < max_lookahead_distance
    required_norm = path_length_small_radius / (steps_num - 1)
    for i in range(segments_norm.shape[0]):
        assert np.abs(segments_norm[i] - required_norm) < 0.2
