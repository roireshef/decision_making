from decision_making.src.global_constants import DM_MANAGER_NAME_FOR_LOGGING
from spav.decision_making.src.map.map_api import *
from spav.decision_making.src.map.naive_cache_map import NaiveCacheMap
from rte.python.logger.AV_logger import AV_Logger

def test_map():
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    map = NaiveCacheMap("../../../../spcog/decision_making_sim/test/world_generator/CustomMapPickle.bin", logger)
    road_ids = map.find_roads_containing_point(0, -30, 50)
    lanes = map.get_center_lanes_latitudes(1)
    print(road_ids)

test_map()
