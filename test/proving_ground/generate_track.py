from decision_making.src.global_constants import WERLING_TIME_RESOLUTION
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.types import FP_SX, FP_DX, C_V, C_A
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.test.planning.trajectory.utils import WerlingVisualizer
from mapping.src.model.map_api import MapAPI
from mapping.src.service.map_service import MapService, MapServiceArgs
from rte.python.logger.AV_logger import AV_Logger

import numpy as np
import matplotlib.pyplot as plt

NORTH_ST_LAT = 32.218577
NORTH_ST_LON = 34.835475

SOUTH_ST_LAT = 32.212071
SOUTH_ST_LON = 34.83709

INIT_VEL = 20/3.6  # [m/s]
GOAL_VEL = 50/3.6  # [m/s]

ROAD_ID = 20

fig = plt.figure()
pmap = fig.add_subplot(211)
pvel = fig.add_subplot(223)
pacc = fig.add_subplot(224)

MapService.initialize(MapServiceArgs(map_source='TestingGroundMap3LanesOld.bin'))
map: MapAPI = MapService.get_instance()
road = map.get_road(ROAD_ID)

cnorth = np.array(map.convert_geo_to_map_coordinates(NORTH_ST_LAT, NORTH_ST_LON))
csouth = np.array(map.convert_geo_to_map_coordinates(SOUTH_ST_LAT, SOUTH_ST_LON))

WerlingVisualizer.plot_route(pmap, road._points)
pmap.plot(cnorth[0], cnorth[1], '.b')
pmap.plot(csouth[0], csouth[1], '.r')

frenet = FrenetSerret2DFrame(road._points)

finit = frenet.cpoint_to_fpoint(csouth)
init_constraints = FrenetConstraints(sx=finit[FP_SX], sv=INIT_VEL, sa=0, dx=finit[FP_DX], dv=0, da=0)

# T = 2 * np.linalg.norm(finit - fgoal) / (GOAL_VEL + INIT_VEL)
T = 10  # [sec]
time_points = np.arange(0.0, T, WERLING_TIME_RESOLUTION)

goal_constraints = FrenetConstraints(sx=init_constraints._sx + (GOAL_VEL + INIT_VEL) / 2 * T, sv=GOAL_VEL,
                                     sa=0, dx=0, dv=0, da=0)

ftrajectories = WerlingPlanner._solve_optimization(init_constraints, goal_constraints, T, time_points)

ctrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories)


WerlingVisualizer.plot_best(pmap, ctrajectories[0])

WerlingVisualizer.plot_route(pvel, np.c_[time_points, ctrajectories[0, :, C_V]])
WerlingVisualizer.plot_route(pacc, np.c_[time_points, ctrajectories[0, :, C_A]])

fig.show()
fig.clear()

np.savetxt('traj_straight_from_south_20kmh_to_50kmh_10sec.txt', ctrajectories[0], delimiter=', ', newline='\n', fmt='%1.8f')