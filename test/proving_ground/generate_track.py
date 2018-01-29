from decision_making.src.global_constants import WERLING_TIME_RESOLUTION
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.types import FP_SX, FP_DX, C_V, C_A, C_K
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.test.planning.trajectory.utils import WerlingVisualizer
from mapping.src.model.map_api import MapAPI
from mapping.src.service.map_service import MapService, MapServiceArgs

import numpy as np
import matplotlib.pyplot as plt


class OfflineTrajectoryGenerator:
    def __init__(self, road_id, map_file_name: str):
        MapService.initialize(MapServiceArgs(map_source=map_file_name))
        self.map: MapAPI = MapService.get_instance()
        self.road = self.map.get_road(road_id)
        self.frenet = FrenetSerret2DFrame(self.road._points)

    def plan(self, geo_coordinates: np.array, init_velocity: float, interm_goals: list):
        map_coordinates = np.array(self.map.convert_geo_to_map_coordinates(geo_coordinates[0], geo_coordinates[1]))

        finit = self.frenet.cpoint_to_fpoint(map_coordinates)
        init_fstate_vec = np.array([finit[FP_SX], init_velocity, 0, finit[FP_DX], 0, 0])
        init_state = FrenetConstraints.from_state(init_fstate_vec)

        # compose a trajectory from intermediate actions
        trajectory = [self.frenet.ftrajectory_to_ctrajectory(np.array([init_fstate_vec]))[0]]
        current_state = init_state
        for action in interm_goals:
            interm_traj, goal_state = self._plan_segment(current_state, action[0], action[1] * self.road.lane_width, action[2])

            # trim first point to remove duplicates
            trajectory = np.concatenate((trajectory, interm_traj[1:, :]), axis=0)

            current_state = goal_state

        return trajectory

    def _plan_segment(self, init_constraints: FrenetConstraints, T: float, delta_dx: float, goal_sv: float):
        """Given initial state, planning-time and desired change in (dx, sv) - generate trajectory (return in cartesian)"""
        time_points = np.arange(0.0, T+WERLING_TIME_RESOLUTION, WERLING_TIME_RESOLUTION)
        goal_state = FrenetConstraints(sx=init_constraints._sx + (goal_sv + init_constraints._sv) / 2 * T, sv=goal_sv,
                                       sa=0, dx=init_constraints._dx + delta_dx, dv=0, da=0)
        ftrajectories, _ = WerlingPlanner._solve_optimization(init_constraints, goal_state, T, time_points)
        ctrajectories = self.frenet.ftrajectories_to_ctrajectories(ftrajectories)
        return ctrajectories[0], goal_state


def main():
    MAP_FILE_NAME = 'TestingGroundMap3LanesOld.bin'
    NORTH_POINT_LAT_LON = [32.218577, 34.835475]
    SOUTH_POINT_LAT_LON = [32.212071, 34.83709]
    ROAD_ID = 20

    # initial velocity at the initial point
    INIT_VEL = 20 / 3.6  # [m/s]

    # list of intermediate goals [time to goal, goal lateral deviation in lanes, goal velocity]
    CRUISE_IN_LANE_TIME = 7.0
    LANE_CHANGE_TIME = 1.0
    CRUISE_VEL = 20 / 3.6  # [m/s]
    interm_goals = [
        [15.0, 0, CRUISE_VEL],
        [LANE_CHANGE_TIME, 1, CRUISE_VEL],
        [CRUISE_IN_LANE_TIME, 0, CRUISE_VEL],
        [LANE_CHANGE_TIME, -1, CRUISE_VEL],
        [CRUISE_IN_LANE_TIME, 0, CRUISE_VEL],
        [LANE_CHANGE_TIME, 1, CRUISE_VEL],
        [CRUISE_IN_LANE_TIME, 0, CRUISE_VEL],
        [LANE_CHANGE_TIME, -1, CRUISE_VEL],
        [CRUISE_IN_LANE_TIME, 0, CRUISE_VEL],
        [LANE_CHANGE_TIME, 1, CRUISE_VEL],
        [CRUISE_IN_LANE_TIME, 0, CRUISE_VEL]
    ]

    generator = OfflineTrajectoryGenerator(ROAD_ID, MAP_FILE_NAME)

    init_geo_coordinate = NORTH_POINT_LAT_LON
    init_geo_name = 'north'

    trajectory = generator.plan(geo_coordinates=init_geo_coordinate, init_velocity=INIT_VEL, interm_goals=interm_goals)

    epsilon = np.finfo(np.float16).eps
    time_points = np.arange(0.0, len(trajectory)*WERLING_TIME_RESOLUTION + epsilon, WERLING_TIME_RESOLUTION)

    ### VIZ ###

    fig = plt.figure()
    pmap = fig.add_subplot(211)
    pvel = fig.add_subplot(234)
    pacc = fig.add_subplot(235)
    pcurv = fig.add_subplot(236)

    WerlingVisualizer.plot_route(pmap, generator.road._points)
    pmap.plot(init_geo_coordinate[0], init_geo_coordinate[1], '.b')

    WerlingVisualizer.plot_best(pmap, trajectory)

    WerlingVisualizer.plot_route(pvel, np.c_[time_points, trajectory[:, C_V]])
    WerlingVisualizer.plot_route(pacc, np.c_[time_points, trajectory[:, C_A]])
    WerlingVisualizer.plot_route(pcurv, np.c_[time_points, trajectory[:, C_K]])

    fig.show()
    fig.clear()

    # assert no consecutive duplicates in positions within a trajectory
    assert np.all(np.greater(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=-1), 0))

    np.savetxt('lane_changes_from_%s_%skmh_change_%ssec_cruise_%ssec.txt' %
               (init_geo_name, CRUISE_VEL, LANE_CHANGE_TIME, CRUISE_IN_LANE_TIME),
               trajectory, delimiter=', ', newline='\n', fmt='%1.8f')



