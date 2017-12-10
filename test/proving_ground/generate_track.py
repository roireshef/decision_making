from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from mapping.src.model.map_api import MapAPI
from mapping.src.service.map_service import MapService, MapServiceArgs
from rte.python.logger.AV_logger import AV_Logger

INIT_LAT = 32.218577
INIT_LON = 34.835475

GOAL_LAT = 32.212071
GOAL_LON = 34.83709

MapService.initialize(MapServiceArgs(map_source='TestingGroundMap3Lanes.bin'))
map: MapAPI = MapService.get_instance()

init_x, init_y = map.convert_geo_to_map_coordinates(INIT_LAT, INIT_LON)
goal_x, goal_y = map.convert_geo_to_map_coordinates(GOAL_LAT, GOAL_LON)

planner = WerlingPlanner(AV_Logger.get_logger('werling'), )
