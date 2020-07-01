import matplotlib
import numpy as np

from decision_making.src.global_constants import WERLING_TIME_RESOLUTION
from decision_making.src.planning.trajectory.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.types import FP_SX, FP_DX, C_V, C_A, C_K, CartesianExtendedTrajectory
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.test.planning.trajectory.utils import WerlingVisualizer
from gm_ctm.deprecated import CtmService


class OfflineTrajectoryGenerator:
    """ This class takes specifications for maneuvers and generates a trajectory that
    goes according to them on a specific road using Werling planner """
    def __init__(self, frenet: FrenetSerret2DFrame):
        """
        :param frenet: the frenet frame that corresponds to the RHS of the road
        """
        self.frenet = frenet

    def plan(self, geo_coordinates: np.array, init_velocity: float, interm_goals: list, lane_width: float) -> CartesianExtendedTrajectory:
        """
        Given the initial state of the vehicle (in geo_coordinates and init_velocity) and a list of intermediate goals
        (each specifies a maneuver), generate a single and unified trajectory that goes through those goals.
        :param geo_coordinates: Geographic coordinates of the vehicle's initial position
        :param init_velocity: Vehicle's initial velocity [m/sec]
        :param interm_goals: list of intermediate goals of the format: (time to goal [sec],
        goal lateral deviation [lanes], goal velocity [m/sec])
        :param lane_width: [m] assumes constant lane widths on the road, used for offsetting the center of lanes
        :return: a single trajectory that goes through all intermediate goals specified
        """
        ctm_transform = CtmService.get_ctm()
        map_coordinates = np.array(ctm_transform.transform_geo_location_to_map(geo_coordinates[0], geo_coordinates[1]))

        finit = self.frenet.cpoint_to_fpoint(map_coordinates)
        init_fstate_vec = np.array([finit[FP_SX], init_velocity, 0, finit[FP_DX], 0, 0])
        init_state = FrenetConstraints.from_state(init_fstate_vec)

        # compose a trajectory from intermediate actions
        trajectory = [self.frenet.ftrajectory_to_ctrajectory(np.array([init_fstate_vec]))[0]]
        current_state = init_state
        for action in interm_goals:
            interm_traj, goal_state = self._plan_segment(current_state, action[0], action[1] * lane_width,
                                                         action[2])

            # trim first point to remove duplicates
            trajectory = np.concatenate((trajectory, interm_traj[1:, :]), axis=0)

            current_state = goal_state

        return trajectory

    def _plan_segment(self, init_constraints: FrenetConstraints, T: float, delta_dx: float, goal_sv: float):
        """
        Given initial state in frenet frame, planning-time, and desired change in (dx, sv) - generate trajectory
        (returns trajectory in cartesian)
        :param init_constraints: initial frenet-frame state of the vehicle
        :param T: planning time [sec]
        :param delta_dx: desired change in lateral position [m] (difference between goal and current)
        :param goal_sv: desired longitudinal velocity [m/sec] (of goal)
        :return: trajectory in Cartesian-frame
        """
        goal_state = FrenetConstraints(sx=init_constraints._sx + (goal_sv + init_constraints._sv) / 2 * T, sv=goal_sv,
                                       sa=0, dx=init_constraints._dx + delta_dx, dv=0, da=0)
        ftrajectories, _, _ = WerlingPlanner._solve_optimization(init_constraints, goal_state, T, np.array([T]),
                                                                 WERLING_TIME_RESOLUTION)
        ctrajectories = self.frenet.ftrajectories_to_ctrajectories(ftrajectories)
        return ctrajectories[0], goal_state


def main():
    NORTH_POINT_LAT_LON = [32.218577, 34.835475]
    SOUTH_POINT_LAT_LON = [32.212071, 34.83709]

    # initial velocity at the initial point
    INIT_VEL = 20 / 3.6  # [m/s]

    # list of intermediate goals [time to goal, goal lateral deviation in lanes, goal velocity]
    CRUISE_IN_LANE_TIME = 7.0
    LANE_CHANGE_TIME = 7.0
    CRUISE_VEL = 50 / 3.6  # [m/s]
    interm_goals = [
        [15.0, 0, CRUISE_VEL],
        [LANE_CHANGE_TIME, 1, CRUISE_VEL],
        [CRUISE_IN_LANE_TIME, 0, CRUISE_VEL],
        [LANE_CHANGE_TIME, -1, CRUISE_VEL],
        [CRUISE_IN_LANE_TIME, 0, CRUISE_VEL],
        [LANE_CHANGE_TIME, 1, CRUISE_VEL],
        [CRUISE_IN_LANE_TIME, 0, CRUISE_VEL],
        [LANE_CHANGE_TIME, -1, CRUISE_VEL],
        [CRUISE_IN_LANE_TIME, 0, CRUISE_VEL]
    ]

    LANE_WIDTH = 3.6

    init_geo_coordinate = NORTH_POINT_LAT_LON
    init_geo_name = 'north'

    frenet = None  # Instantiate a FrenetSerret2DFrame here.
    generator = OfflineTrajectoryGenerator(frenet)
    trajectory = generator.plan(geo_coordinates=init_geo_coordinate, init_velocity=INIT_VEL, interm_goals=interm_goals, lane_width=LANE_WIDTH)
    time_points = np.arange(0.0, len(trajectory)*WERLING_TIME_RESOLUTION, WERLING_TIME_RESOLUTION)

    ### VIZ ###

    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()

    pmap = fig.add_subplot(211)
    plt.title('Map')
    pvel = fig.add_subplot(234)
    plt.title(r'v_lon[$\frac{m}{s}$]')
    pacc = fig.add_subplot(235)
    plt.title(r'a_lon[$\frac{m}{s^2}$]')
    pcurv = fig.add_subplot(236)
    plt.title(r'curvature[$\frac{1}{m}$]')

    WerlingVisualizer.plot_route(pmap, generator.frenet.O)
    pmap.plot(init_geo_coordinate[0], init_geo_coordinate[1], '.b')

    WerlingVisualizer.plot_best(pmap, trajectory)

    WerlingVisualizer.plot_route(pvel, np.c_[time_points, trajectory[:, C_V]])
    WerlingVisualizer.plot_route(pacc, np.c_[time_points, trajectory[:, C_A]])
    WerlingVisualizer.plot_route(pcurv, np.c_[time_points, trajectory[:, C_K]])

    plt.show()
    fig.clear()

    # assert no consecutive duplicates in positions within a trajectory
    assert np.all(np.greater(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=-1), 0))

    np.savetxt('lane_changes_from_%s_%skmh_change_%ssec_cruise_%ssec.txt' %
               (init_geo_name, CRUISE_VEL*3.6, LANE_CHANGE_TIME, CRUISE_IN_LANE_TIME),
               trajectory, delimiter=', ', newline='\n', fmt='%1.8f')


if __name__ == '__main__':
    main()
