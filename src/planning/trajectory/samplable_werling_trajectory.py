import numpy as np

from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.trajectory.werling_utils import WerlingUtils
from decision_making.src.planning.types import CartesianExtendedTrajectory, FrenetTrajectory2D, FS_1D_LEN, \
    FrenetTrajectory1D
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor


class SamplableWerlingTrajectory(SamplableTrajectory):
    def __init__(self, timestamp_in_sec: float, T_s: float, T_d: float, T_extended: float, frenet_frame: FrenetSerret2DFrame,
                 poly_s_coefs: np.ndarray, poly_d_coefs: np.ndarray):
        """
        To represent a trajectory that is a result of Werling planner, we store the frenet frame used and
        two polynomial coefficients vectors (for dimensions s and d)
        :param timestamp_in_sec: [sec] global timestamp *in seconds* to use as a reference
                (other timestamps will be given relative to it)
        :param T_s: [sec] longitudinal trajectory duration (relative to self.timestamp).
        :param T_d: [sec] lateral trajectory duration (relative to self.timestamp).
        :param T_extended: [sec] lateral trajectory duration (relative to self.timestamp).
        :param frenet_frame: frenet frame of the curve which was used to create this samplable trajectory, used for
                            transforming between frenet and cartesian coordinates.
        :param poly_s_coefs: coefficients of the longitudinal polynomial which is being sampled for getting the
                longitudinal frenet states
        :param poly_d_coefs: coefficients of the lateral polynomial which is being sampled for getting the
                lateral frenet states
        """
        super().__init__(timestamp_in_sec, T_s)
        self.T_d = T_d
        self.T_extended = T_extended
        self.frenet_frame = frenet_frame
        self.poly_s_coefs = poly_s_coefs
        self.poly_d_coefs = poly_d_coefs

    @property
    def T_s(self):
        return self.T

    @property
    def max_sample_time(self):
        return self.timestamp_in_sec + self.T_extended

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
        assert max(relative_time_points) <= self.T_extended, \
            'self.total_trajectory_time=%f, max(relative_time_points)=%f' % (self.T_extended, max(relative_time_points))

        # Handle the longitudinal(s) axis
        fstates_s = self.sample_lon_frenet(relative_time_points)
        # Now handle the lateral(d) axis:
        fstates_d = self._sample_lat_frenet(relative_time_points)

        # Return trajectory in Frenet coordinates
        return np.hstack((fstates_s, fstates_d))

    def sample_lon_frenet(self, relative_time_points: np.array) -> FrenetTrajectory1D:
        """
        Samples the appropriate longitudinal frenet states from the longitudinal polynomial w.r.t to the given
        relative time points
        :param relative_time_points: time points relative to the trajectory timestamp itself [s]
        :return:
        """

        in_traj_time_points = relative_time_points[relative_time_points <= self.T_s]
        extrapolated_time_points = relative_time_points[np.logical_and(relative_time_points > self.T_s,
                                                                       relative_time_points <= self.T_extended)]

        fstates_s = np.empty((0, FS_1D_LEN))

        if len(in_traj_time_points) > 0:  # time points are part of the real polynomial part of the trajectory
            # assign values from <time_points> in s-axis polynomial
            in_traj_fstates_s = \
            QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_s_coefs]), in_traj_time_points)[0]
            fstates_s = np.vstack((fstates_s, in_traj_fstates_s))

        if len(extrapolated_time_points) > 0:
            # time points will trigger extrapolating the last sampled point from the polynomial using a constant
            # velocity predictor
            road_following_predictor = RoadFollowingPredictor(None)
            fstate_in_T_s = QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_s_coefs]), np.array([self.T_s]))[
                0]
            extrapolated_fstates_s = \
            road_following_predictor.predict_1d_frenet_states(fstate_in_T_s, extrapolated_time_points - self.T_s)[0]
            fstates_s = np.vstack((fstates_s, extrapolated_fstates_s))

        return fstates_s

    def _sample_lat_frenet(self, relative_time_points: np.array) -> FrenetTrajectory1D:
        """
        Samples the appropriate lateral frenet states from the lateral polynomial w.r.t to the given
        relative time points
        :param relative_time_points: time points relative to the trajectory timestamp itself [s]
        :return:
        """

        fstates_d = np.empty(shape=np.append(relative_time_points.shape, 3))

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

        return fstates_d
