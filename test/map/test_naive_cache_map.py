import numpy as np
from decision_making.src.map.naive_cache_map import NaiveCacheMap
from decision_making.src.planning.navigation.navigation_plan import NavigationPlan
from rte.python.logger.AV_logger import AV_Logger

def test_init():
    logger = AV_Logger.get_logger("test_cache_map")
    naive_cache_map = NaiveCacheMap()
    naive_cache_map.load_map("/home/vlad/dev/python_git/decision_making_sim/test/world_generator/CustomMapPickle.bin")
    lanes_num, width, length, points = naive_cache_map.get_road_details(1)
    logger.info("lanes_num=%d, width=%f length=%f", lanes_num, width, length)
    road_id, lane, full_lat, lane_lat, lon, yaw_in_road = naive_cache_map.get_point_in_road_coordinates(-51, 20, 0, -np.pi/2)
    logger.info("road_id=%d lane=%d, full_lat=%f lon=%f yaw=%f", road_id, lane, full_lat, lon, yaw_in_road)

    navigation_plan = NavigationPlan([1,2])
    logger.info("get_uniform_path_lookahead")
    points, lat_vecs = naive_cache_map.get_uniform_path_lookahead(1, 2, 20, 5, 80, navigation_plan)
    for p in points.transpose():
        logger.info("%d,%d", p[0], p[1])

    logger.info("get_point_relative_longitude")
    total_lon_distance, found_connection = naive_cache_map.get_point_relative_longitude(1, 10, 2, 30, 400, navigation_plan)
    logger.info("dist=%d found=%d", total_lon_distance, found_connection)

    logger.info("get_path_lookahead")
    points = naive_cache_map.get_path_lookahead(1, 10, 1, 200, navigation_plan, direction=1)
    for p in points.transpose():
        logger.info("%d,%d", p[0], p[1])

test_init()
