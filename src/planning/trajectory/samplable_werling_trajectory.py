import numpy as np

from decision_making.src.global_constants import EPS
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.trajectory.werling_utils import WerlingUtils
from decision_making.src.planning.types import CartesianExtendedTrajectory, FrenetTrajectory2D, FS_DX, FS_SX
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor


class SamplableWerlingTrajectory(SamplableTrajectory):
    def __init__(self, timestamp_in_sec: float, T_s: float, T_d: float, total_time: float, frenet_frame: FrenetSerret2DFrame,
                 poly_s_coefs: np.ndarray, poly_d_coefs: np.ndarray):
        """To represent a trajectory that is a result of Werling planner, we store the frenet frame used and
        two polynomial coefficients vectors (for dimensions s and d)"""
        super().__init__(timestamp_in_sec, T_s)
        self.T_d = T_d
        self.total_trajectory_time = total_time
        self.frenet_frame = frenet_frame
        self.poly_s_coefs = poly_s_coefs
        self.poly_d_coefs = poly_d_coefs

    @property
    def T_s(self):
        return self.T

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        """See base method for API. In this specific representation of the trajectory, we sample from s-axis polynomial
        (longitudinal) and partially (up to some time-horizon cached in self.lon_plan_horizon) from d-axis polynomial
        (lateral) and extrapolate the rest of the states in d-axis to conform to the trajectory's total duration"""

        # Sample the trajectory in the desired points in time in Frenet coordinates
        fstates = self.sample_frenet(time_points=time_points)

        # project from road coordinates to cartesian coordinate frame
        cstates = self.frenet_frame.ftrajectory_to_ctrajectory(fstates)

        return cstates

    def sample_frenet(self, time_points: np.ndarray) -> FrenetTrajectory2D:
        """
        This function takes an array of time stamps and returns an array of Frenet states along the trajectory.
        We sample from s-axis polynomial (longitudinal) and partially (up to some time-horizon cached in
        self.lon_plan_horizon) from d-axis polynomial (lateral) and extrapolate the rest of the states in d-axis
        to conform to the trajectory's total duration.
        :param time_points: 1D numpy array of time stamps *in seconds* (global self.timestamp)
        :return: Frenet Trajectory
        """

        relative_time_points = time_points - self.timestamp_in_sec

        # Make sure no unplanned extrapolation will occur due to overreaching time points
        # This check is done in relative-to-ego units
        assert max(relative_time_points) <= self.total_trajectory_time + EPS, \
            'self.total_trajectory_time=%f, max(relative_time_points)=%f' % (self.total_trajectory_time, max(relative_time_points))

        # Handle the longitudinal(s) axis

        in_traj_time_points = relative_time_points[relative_time_points <= self.T_s+EPS]
        padded_time_points = relative_time_points[np.logical_and(self.total_trajectory_time >= relative_time_points,relative_time_points > self.T_s + EPS)]

        # assign values from <time_points> in s-axis polynomial
        fstates_s = QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_s_coefs]), in_traj_time_points)[0]

        if len(padded_time_points) > 0:
            road_following_predictor = RoadFollowingPredictor(None)
            last_frenet_state = np.array([np.append(fstates_s[-1], [0, 0, 0])])
            extrapolated_fstates_s = road_following_predictor.predict_frenet_states(last_frenet_state, padded_time_points-self.T_s)[0]
            fstates_s = np.vstack((extrapolated_fstates_s[:, FS_SX: FS_DX], fstates_s))

        # Now handle the lateral(d) axis:

        fstates_d = np.empty(shape=np.append(time_points.shape, 3))

        is_within_horizon_d = relative_time_points <= self.T_d

        fstates_d[is_within_horizon_d] = QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_d_coefs]),
                                                                                relative_time_points[
                                                                                    is_within_horizon_d])

        # Expand lateral solution to the size of the longitudinal solution with its final positions replicated
        # NOTE: we assume that velocity and accelerations = 0 !!
        end_of_horizon_state_d = QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_d_coefs]),
                                                                        np.array([self.T_d]))[0]
        extrapolation_state_d = WerlingUtils.repeat_1d_state(
            fstate=end_of_horizon_state_d,
            repeats=1,
            override_values=np.zeros(3),
            override_mask=np.array([0, 1, 1]))

        fstates_d[np.logical_not(is_within_horizon_d)] = extrapolation_state_d

        # Return trajectory in Frenet coordinates
        fstates = np.hstack((fstates_s, fstates_d))

        return fstates