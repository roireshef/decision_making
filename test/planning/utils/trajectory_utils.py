import numpy as np
import pytest

from decision_making.src.global_constants import EPS, TRAJECTORY_TIME_RESOLUTION
from decision_making.src.planning.trajectory.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D, QuinticPoly1D


@pytest.fixture(scope='module')
def T_s():
    """
    longitudinal planning time of testable trajectories
    """
    return 10


class TrajectoryUtils:

    @staticmethod
    def create_ftrajectory(v0: float, vT: float, lane0: int, laneT: int, lon0: float=0, T_d: float=T_s()):
        """
        Create Frenet trajectory (quartic for longitudinal axis, quintic for lateral axis)
        Create trajectory (for ego or object) for given start/end velocities, start/end lane centers, relative longitude, T_d
        Used by all RSS tests.
        :param v0: initial velocity
        :param vT: end velocity
        :param lane0: initial lane
        :param laneT: end lane
        :param lon0: initial relative longitude (wrt initial ego longitude ego_lon)
        :param T_d: lateral motion time
        :return: resulting Frenet trajectory
        """
        ego_lon = 200
        lane_wid = 3.5
        center_lane_lats = lane_wid / 2 + np.arange(0, 2 * lane_wid + EPS, lane_wid)  # 3 lanes

        fconst_0 = FrenetConstraints(ego_lon + lon0, v0, 0, center_lane_lats[lane0], 0, 0)
        fconst_t = FrenetConstraints(0, vT, 0, center_lane_lats[laneT], 0, 0)
        ftraj, _, _ = WerlingPlanner.solve_optimization(fconst_0, fconst_t, T_s(), np.array([T_d]),
                                                        TRAJECTORY_TIME_RESOLUTION, QuarticPoly1D)
        return ftraj
