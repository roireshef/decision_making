import numpy as np
import pytest

from decision_making.src.global_constants import EPS, TRAJECTORY_TIME_RESOLUTION
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
        time_range = np.arange(0, T_s()+EPS, TRAJECTORY_TIME_RESOLUTION)
        center_lane_lats = lane_wid / 2 + np.arange(0, 2 * lane_wid + EPS, lane_wid)  # 3 lanes

        constraints_s = np.array([[ego_lon + lon0, v0, 0, vT, 0]])
        constraints_d = np.array([[center_lane_lats[lane0], 0, 0, center_lane_lats[laneT], 0, 0]])
        poly_s = WerlingPlanner._solve_1d_poly(constraints_s, T_s(), QuarticPoly1D)
        fstates_s = QuarticPoly1D.polyval_with_derivatives(poly_s, time_range)
        poly_d = WerlingPlanner._solve_1d_poly(constraints_d, T_d, QuinticPoly1D)
        fstates_d = QuinticPoly1D.polyval_with_derivatives(poly_d, time_range)
        if T_d < T_s():  # fill all samples beyond T_d by the sample at T_d
            T_d_sample = int(T_d / TRAJECTORY_TIME_RESOLUTION)
            fstates_d[:, T_d_sample + 1:, :] = fstates_d[:, T_d_sample:T_d_sample + 1, :]
        ftraj = np.dstack((fstates_s, fstates_d))
        return ftraj
