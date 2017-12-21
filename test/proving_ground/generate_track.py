from decision_making.src.global_constants import WERLING_TIME_RESOLUTION
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.types import FP_SX, FP_DX, C_V, C_A, C_K
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.test.planning.trajectory.utils import WerlingVisualizer
from mapping.src.model.map_api import MapAPI
from mapping.src.service.map_service import MapService, MapServiceArgs
from rte.python.logger.AV_logger import AV_Logger

import numpy as np
import matplotlib.pyplot as plt

### MAP INIT ###

NORTH_ST_LAT = 32.218577
NORTH_ST_LON = 34.835475

SOUTH_ST_LAT = 32.212071
SOUTH_ST_LON = 34.83709

ROAD_ID = 20

MapService.initialize(MapServiceArgs(map_source='TestingGroundMap3LanesOld.bin'))
map: MapAPI = MapService.get_instance()
road = map.get_road(ROAD_ID)

cnorth = np.array(map.convert_geo_to_map_coordinates(NORTH_ST_LAT, NORTH_ST_LON))
csouth = np.array(map.convert_geo_to_map_coordinates(SOUTH_ST_LAT, SOUTH_ST_LON))

frenet = FrenetSerret2DFrame(road._points)

def plan_action(init_constraints: FrenetConstraints, T: float, delta_dx: float, goal_sv: float):
    """Given initial state, planning-time and desired change in (dx, sv) - generate trajectory (return in cartesian)"""
    time_points = np.arange(0.0, T+WERLING_TIME_RESOLUTION, WERLING_TIME_RESOLUTION)
    goal_state = FrenetConstraints(sx=init_constraints._sx + (goal_sv + init_constraints._sv) / 2 * T, sv=goal_sv,
                                         sa=0, dx=init_constraints._dx + delta_dx, dv=0, da=0)
    ftrajectories, _ = WerlingPlanner._solve_optimization(init_constraints, goal_state, T, time_points)
    ctrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories)
    return ctrajectories[0], goal_state


INIT_VEL = 20/3.6  # [m/s]
GOAL_VEL = 50/3.6  # [m/s]

CRUISE_VEL = 20/3.6  # [m/s]

finit = frenet.cpoint_to_fpoint(cnorth)
init_fstate_vec = np.array([finit[FP_SX], INIT_VEL, 0, finit[FP_DX], 0, 0])
init_state = FrenetConstraints.from_state(init_fstate_vec)

# list of (T [sec], delta_dx [m], goal_sv [m/sec]) intermediate goals
cruise_in_lane_time = 7.0
lane_change_time = 1.0
interm_action = [
    [15.0, 0, CRUISE_VEL],
    [lane_change_time, road.lane_width, CRUISE_VEL],
    [cruise_in_lane_time, 0, CRUISE_VEL],
    [lane_change_time, -road.lane_width, CRUISE_VEL],
    [cruise_in_lane_time, 0, CRUISE_VEL],
    [lane_change_time, road.lane_width, CRUISE_VEL],
    [cruise_in_lane_time, 0, CRUISE_VEL],
    [lane_change_time, -road.lane_width, CRUISE_VEL],
    [cruise_in_lane_time, 0, CRUISE_VEL],
    [lane_change_time, road.lane_width, CRUISE_VEL],
    [cruise_in_lane_time, 0, CRUISE_VEL]
]

# compose a trajectory from intermediate actions
trajectory = [frenet.ftrajectory_to_ctrajectory(np.array([init_fstate_vec]))[0]]
total_time = 0.0
current_state = init_state
for action in interm_action:
    interm_traj, goal_state = plan_action(current_state, action[0], action[1], action[2])

    trajectory = np.concatenate((trajectory, interm_traj[1:, :]), axis=0)  # trim first point to remove duplicates

    current_state = goal_state
    total_time += action[0]

time_points = np.arange(0.0, total_time+WERLING_TIME_RESOLUTION, WERLING_TIME_RESOLUTION)

### VIZ ###

fig = plt.figure()
pmap = fig.add_subplot(211)
pvel = fig.add_subplot(234)
pacc = fig.add_subplot(235)
pcurv = fig.add_subplot(236)

WerlingVisualizer.plot_route(pmap, road._points)
pmap.plot(cnorth[0], cnorth[1], '.b')
pmap.plot(csouth[0], csouth[1], '.r')

WerlingVisualizer.plot_best(pmap, trajectory)

WerlingVisualizer.plot_route(pvel, np.c_[time_points, trajectory[:, C_V]])
WerlingVisualizer.plot_route(pacc, np.c_[time_points, trajectory[:, C_A]])
WerlingVisualizer.plot_route(pcurv, np.c_[time_points, trajectory[:, C_K]])

fig.show()
fig.clear()

# assert no consecutive duplicates in positions within a trajectory
assert np.all(np.greater(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=-1), 0))

np.savetxt('lane_changes_from_north_50kmh_change_1sec_cruise_7sec.txt', trajectory, delimiter=', ', newline='\n', fmt='%1.8f')


