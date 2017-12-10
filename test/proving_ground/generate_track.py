from decision_making.src.global_constants import WERLING_TIME_RESOLUTION
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.types import FP_SX, FP_DX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from mapping.src.model.map_api import MapAPI
from mapping.src.service.map_service import MapService, MapServiceArgs
from rte.python.logger.AV_logger import AV_Logger

import numpy as np

INIT_LAT = 32.218577
INIT_LON = 34.835475
INIT_VEL = 20  # [m/s]

GOAL_LAT = 32.212071
GOAL_LON = 34.83709
GOAL_VEL = 20  # [m/s]

ROAD_ID = 20

MapService.initialize(MapServiceArgs(map_source='TestingGroundMap3Lanes.bin'))
map: MapAPI = MapService.get_instance()
road = map.get_road(ROAD_ID)

cinit = np.array(map.convert_geo_to_map_coordinates(INIT_LAT, INIT_LON))
cgoal = np.array(map.convert_geo_to_map_coordinates(GOAL_LAT, GOAL_LON))

frenet = FrenetSerret2DFrame(road._points)

finit = frenet.cpoint_to_fpoint(cinit)
init_constraints = FrenetConstraints(sx=finit[FP_SX], sv=INIT_VEL, sa=0, dx=finit[FP_DX], dv=0, da=0)

fgoal = frenet.cpoint_to_fpoint(cgoal)
goal_constraints = FrenetConstraints(sx=fgoal[FP_SX], sv=GOAL_VEL, sa=0, dx=fgoal[FP_DX], dv=0, da=0)

T = 2 * np.linalg.norm(finit - fgoal) / (GOAL_VEL + INIT_VEL)
time_points = np.arange(0.0, T, WERLING_TIME_RESOLUTION)

ftrajectories = WerlingPlanner._solve_optimization(init_constraints, goal_constraints, T, time_points)


