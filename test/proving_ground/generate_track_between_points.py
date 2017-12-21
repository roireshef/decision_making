import matplotlib.pyplot as plt
import numpy as np

from decision_making.src.global_constants import WERLING_TIME_RESOLUTION
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.types import C_V, C_A, C_K
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.test.planning.trajectory.utils import WerlingVisualizer
from mapping.src.model.map_api import MapAPI
from mapping.src.service.map_service import MapService, MapServiceArgs

### MAP INIT ###

NORTH_ST_LAT = 32.218577
NORTH_ST_LON = 34.835475

SOUTH_ST_LAT = 32.212071
SOUTH_ST_LON = 34.83709

ROAD_ID = 20

MapService.initialize(MapServiceArgs(map_source='TestingGroundMap3LanesOld.bin'))
map: MapAPI = MapService.get_instance()
road = map.get_road(ROAD_ID)

frenet = FrenetSerret2DFrame(road._points)

T = 13  # [sec]
time_points = np.arange(0.0, T+WERLING_TIME_RESOLUTION, WERLING_TIME_RESOLUTION)

starting_point = frenet.ctrajectory_to_ftrajectory(
    np.array([[1055.9186431031828, -48.83041200760383, -2.3861229999999995, 10.059331349065568, 0, 0]]))[0]
init_state = FrenetConstraints.from_state(starting_point)

end_point = frenet.ctrajectory_to_ftrajectory(
    np.array([[868.3829118232477, -174.1275141301752, -2.770566, 11.580967587451124, 0, 0]]))[0]
goal_state = FrenetConstraints.from_state(end_point)

ftrajectories, _ = WerlingPlanner._solve_optimization(init_state, goal_state, T, time_points)
trajectory = frenet.ftrajectories_to_ctrajectories(ftrajectories)[0]

### VIZ ###

fig = plt.figure()
pmap = fig.add_subplot(211)
pvel = fig.add_subplot(234)
pacc = fig.add_subplot(235)
pcurv = fig.add_subplot(236)

WerlingVisualizer.plot_route(pmap, road._points)
pmap.plot(starting_point[0], starting_point[1], '.b')

WerlingVisualizer.plot_best(pmap, trajectory)

WerlingVisualizer.plot_route(pvel, np.c_[time_points, trajectory[:, C_V]])
WerlingVisualizer.plot_route(pacc, np.c_[time_points, trajectory[:, C_A]])
WerlingVisualizer.plot_route(pcurv, np.c_[time_points, trajectory[:, C_K]])

fig.show()
fig.clear()

# assert no consecutive duplicates in positions within a trajectory
assert np.all(np.greater(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=-1), 0))

np.savetxt('trajectory_from_recording_2017_11_08_run2.txt', trajectory, delimiter=', ', newline='\n', fmt='%1.8f')


