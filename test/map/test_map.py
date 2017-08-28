from decision_making.src.global_constants import DM_MANAGER_NAME_FOR_LOGGING
from decision_making.src.map.map_api import *
from decision_making.src.map.naive_cache_map import NaiveCacheMap
from rte.python.logger.AV_logger import AV_Logger

def test_map():
    # TODO: add concrete assert statements

    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    map = NaiveCacheMap("../../resources/maps/testingGroundMap.bin", logger)
    points = np.array([[0, 51], [2, 51]]).transpose()
    shifted_points = map._shift_road_vector_in_lat(points, 1)
    road_id = 1
    pnt_ind = 1
    lon = 40
    lat = 2
    navigation_plan = NavigationPlanMsg([1,2])
    road_id, length, right_point, lat_vec, pnt_ind, lon = map._convert_lon_to_world(road_id, pnt_ind, lon, navigation_plan)
    world_pnt = map._convert_lat_lon_to_world(road_id, lat, lon, navigation_plan)
    lat_dist, lon1 = map._convert_world_to_lat_lon_for_given_road(25, 48, road_id)
    closest_lat, closest_lon, closest_id = map._find_closest_road(25., -51., [1,2])
    print("")

test_map()
