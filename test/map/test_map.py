from decision_making.src.global_constants import DM_MANAGER_NAME_FOR_LOGGING
from spav.decision_making.src.map.map_api import *
from spav.decision_making.src.map.naive_cache_map import NaiveCacheMap
from rte.python.logger.AV_logger import AV_Logger

def test_find_roads_containing_point(map):
    correct_road_ids_list = [1, 2]
    road_ids = map.find_roads_containing_point(0, -30, 50)
    np.testing.assert_array_equal(correct_road_ids_list, road_ids)

    correct_road_ids_list = [1]
    road_ids = map.find_roads_containing_point(0, -20, 50)
    np.testing.assert_array_equal(correct_road_ids_list, road_ids)

    correct_road_ids_list = [1, 2]
    road_ids = map.find_roads_containing_point(0, 30, -50)
    np.testing.assert_array_equal(correct_road_ids_list, road_ids)

    correct_road_ids_list = [2]
    road_ids = map.find_roads_containing_point(0, -20, -50)
    np.testing.assert_array_equal(correct_road_ids_list, road_ids)


def test_get_center_lanes_latitudes(map):
    road_id = 1
    lanes_num, width, length, points = map.get_road_main_details(road_id)
    lane_wid = width/lanes_num
    correct_lat_list = np.arange(lane_wid/2, width, lane_wid)
    lanes_lat = map.get_center_lanes_latitudes(road_id)
    np.testing.assert_array_almost_equal(correct_lat_list, lanes_lat)


def test_convert_world_to_lat_lon(map):
    points = [[20, 51], [20, -49], [20, 71]]
    lat_lon = [[1, 50], [2, 10],  []]

    for i, p in enumerate(points):
        try:
            road_id, lane, full_lat, lane_lat, lon, yaw_in_road = map.convert_world_to_lat_lon(p[0], p[1], 0, 0)
            assert lane_lat==lat_lon[i][0] and lon==lat_lon[i][1]
        except:
            assert len(lat_lon[i]) == 0
            print("the point is outside any road")


def test_get_point_relative_longitude(map):
    navigation_plan = NavigationPlanMsg([1,2])

    lanes_num, width, length, points = map.get_road_main_details(1)
    from_lon_in_road = 20
    to_lon_in_road = 10
    max_lookahead_distance = 500
    total_lon_distance = map.get_point_relative_longitude(from_road_id=1, from_lon_in_road=from_lon_in_road,
                                                                            to_road_id=2, to_lon_in_road=to_lon_in_road,
                                                                            max_lookahead_distance=max_lookahead_distance,
                                                                            navigation_plan=navigation_plan)
    assert total_lon_distance == length - from_lon_in_road + to_lon_in_road

    from_lon_in_road = 10
    to_lon_in_road = 170
    max_lookahead_distance = 150  # less than the distance between the given points
    total_lon_distance = map.get_point_relative_longitude(from_road_id=2, from_lon_in_road=from_lon_in_road,
                                                                            to_road_id=2, to_lon_in_road=to_lon_in_road,
                                                                            max_lookahead_distance=max_lookahead_distance,
                                                                            navigation_plan=navigation_plan)
    assert to_lon_in_road - from_lon_in_road - 10 < total_lon_distance < to_lon_in_road - from_lon_in_road + 10


def test_get_path_lookahead(map):
    lanes_num, width, length, points = map.get_road_main_details(1)
    navigation_plan = NavigationPlanMsg([1, 2])
    max_lookahead_distance = 50  # test short path
    road_id = 1
    lon = 10
    lat = 2  # right lane
    path1 = map.get_path_lookahead(road_id, lon, lat, max_lookahead_distance, navigation_plan)
    path_x = path1[0]
    path_y = path1[1]
    assert path_x[0] == -30 + lon
    assert path_x[-1] == -30 + lon + max_lookahead_distance
    for p in range(path1.shape[1]):
        assert path_y[p] == points[1][0] - width / 2 + lat

    max_lookahead_distance = 200  # test path including two road_ids
    path2 = map.get_path_lookahead(road_id, lon, lat, max_lookahead_distance, navigation_plan)
    segments = np.diff(path2, axis=1)
    segments_norm = np.linalg.norm(segments, axis=0)
    path_length_small_radius = np.sum(segments_norm)
    assert max_lookahead_distance - 10 < path_length_small_radius < max_lookahead_distance

    lat = 4  # left lane
    path3 = map.get_path_lookahead(road_id, lon, lat, max_lookahead_distance, navigation_plan)
    segments = np.diff(path3, axis=1)
    segments_norm = np.linalg.norm(segments, axis=0)
    path_length_large_radius = np.sum(segments_norm)
    assert max_lookahead_distance < path_length_large_radius < max_lookahead_distance + 10


def test_get_uniform_path_lookahead(map):
    lanes_num, width, length, points = map.get_road_main_details(1)
    navigation_plan = NavigationPlanMsg([1, 2])
    road_id = 1
    lon = 10
    lat = 2
    lon_step = 10
    steps_num = 20
    max_lookahead_distance = lon_step*steps_num
    path = map.get_uniform_path_lookahead(road_id, lat, lon, lon_step=lon_step, steps_num=steps_num,
                                          navigation_plan=navigation_plan)
    segments = np.diff(path, axis=1)
    segments_norm = np.linalg.norm(segments, axis=0)
    path_length_small_radius = np.sum(segments_norm)
    assert max_lookahead_distance - 10 < path_length_small_radius < max_lookahead_distance
    required_norm = path_length_small_radius / (steps_num - 1)
    for i in range(segments_norm.shape[0]):
        assert abs(segments_norm[i] - required_norm) < 0.2


def test_map():
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    map = NaiveCacheMap("../../resources/maps/CustomMapPickle.bin", logger)

    test_find_roads_containing_point(map)
    test_get_center_lanes_latitudes(map)
    test_convert_world_to_lat_lon(map)
    test_get_point_relative_longitude(map)
    test_get_path_lookahead(map)
    test_get_uniform_path_lookahead(map)


test_map()
