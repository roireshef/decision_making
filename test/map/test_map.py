from decision_making.src.global_constants import DM_MANAGER_NAME_FOR_LOGGING
from spav.decision_making.src.map.map_api import *
from spav.decision_making.src.map.naive_cache_map import NaiveCacheMap
from rte.python.logger.AV_logger import AV_Logger

def test_map():
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    map = NaiveCacheMap("../../../../spcog/decision_making_sim/test/world_generator/CustomMapPickle.bin", logger)
    closest_lat, closest_sign, closest_lon, closest_yaw, closest_id = map.__find_closest_road(25, -50, [1])
    print("")

test_map()
