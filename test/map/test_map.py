from decision_making.src.global_constants import DM_MANAGER_NAME_FOR_LOGGING
from spav.decision_making.src.map.map_api import *
from spav.decision_making.src.map.naive_cache_map import NaiveCacheMap
from rte.python.logger.AV_logger import AV_Logger

def test_map():
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    map = NaiveCacheMap("../../../../spcog/decision_making_sim/test/world_generator/CustomMapPickle.bin", logger)
    road_ids = map.find_roads_containing_point(0, -30, 50)
    lanes = map.get_center_lanes_latitudes(1)
    lanes_num, width, length, points = map.get_road_main_details(1)
    road_id, lane, full_lat, lane_lat, lon, yaw_in_road = map.convert_world_to_lat_lon(20, 51, 0, 0)
    navigation_plan = NavigationPlanMsg([1,2])
    total_lon_distance, found_connection = map.get_point_relative_longitude(1, 30, 2, 10, 300, navigation_plan)
    print(road_ids)

test_map()
