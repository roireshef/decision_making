import sys
from typing import List
from typing import Optional

import numpy as np
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.trajectory.werling_utils import WerlingUtils
from decision_making.src.planning.types import CartesianExtendedTrajectory, Limits, LIMIT_MIN, LIMIT_MAX, FS_SX, FS_SV, \
    FrenetState2D, FrenetTrajectories2D, FrenetTrajectories1D
from decision_making.src.planning.types import FrenetTrajectory2D, FrenetTrajectory1D
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.utils.frenet_prediction_utils import FrenetPredictionUtils
from decision_making.src.rl_agent.utils.class_utils import Representable


class SamplableWerlingTrajectory(SamplableTrajectory, Representable):
    def __init__(self, timestamp_in_sec: float, T_s: float, T_d: float, T_extended: float,
                 poly_s_coefs: np.ndarray, poly_d_coefs: np.ndarray, gff_id: Optional[str] = None):
        """
        To represent a trajectory that is a result of Werling planner, we store the frenet frame used and
        two polynomial coefficients vectors (for dimensions s and d)
        :param timestamp_in_sec: [sec] global timestamp *in seconds* to use as a reference
                (other timestamps will be given relative to it)
        :param T_s: [sec] longitudinal trajectory duration (relative to self.timestamp).
        :param T_d: [sec] lateral trajectory duration (relative to self.timestamp).
        :param T_extended: [sec] maximal samplable relative time.
        :param poly_s_coefs: coefficients of the longitudinal polynomial which is being sampled for getting the
                longitudinal frenet states
        :param poly_d_coefs: coefficients of the lateral polynomial which is being sampled for getting the
                lateral frenet states
        :param gff_id: the identifier of the generalized frenet frame this trajectory is represented relative to

        """
        super().__init__(timestamp_in_sec, max(T_s, T_d))
        self.T_d = T_d
        self.T_s = T_s
        self.T_extended = T_extended
        self.poly_s_coefs = poly_s_coefs
        self.poly_d_coefs = poly_d_coefs
        self.gff_id = gff_id

        # Lazy evaluation for extrapolation states
        self._fstate_in_T_s = None
        self._fstate_in_T_d = None

        assert not np.isnan(T_s), "T_s is NaN in %s" % self
        assert not np.isnan(T_d), "T_d is NaN in %s" % self
        assert not np.any(np.isnan(poly_s_coefs)), "poly_s_coefs has NaNs in %s" % self
        assert not np.any(np.isnan(poly_d_coefs)), "poly_d_coefs has NaNs in %s" % self

    @property
    def fstate_in_T_s(self):
        if self._fstate_in_T_s is None:
            self._fstate_in_T_s = QuinticPoly1D.polyval_with_derivatives(self.poly_s_coefs[np.newaxis, :],
                                                                         np.array([self.T_s]))[0]
        return self._fstate_in_T_s

    @property
    def fstate_in_T_d(self):
        if self._fstate_in_T_d is None:
            self._fstate_in_T_d = QuinticPoly1D.polyval_with_derivatives(self.poly_d_coefs[np.newaxis, :],
                                                                         np.array([self.T_d]))[0, 0]
        return self._fstate_in_T_d

    @property
    def max_sample_time(self):
        return self.timestamp_in_sec + self.T_extended

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        raise NotImplementedError("SamplableWerlingTrajectory doesn't implement sample() in its RL version")

    def sample_frenet(self, time_points: np.ndarray) -> FrenetTrajectory2D:
        """
        This function takes an array of time stamps and returns an array of Frenet states along the trajectory.
        We sample from s-axis polynomial (longitudinal) and partially (up to some time-horizon cached in
        self.lon_plan_horizon) from d-axis polynomial (lateral) and extrapolate the rest of the states in d-axis
        to conform to the trajectory's total duration.
        :param time_points: 1D numpy array of time stamps *in seconds* (global self.timestamp)
        :return: Frenet Trajectory
        """
        try:
            relative_time_points = time_points - self.timestamp_in_sec

            # Handle the longitudinal(s) axis
            fstates_s = self.sample_lon_frenet(relative_time_points)
            # Now handle the lateral(d) axis:
            fstates_d = self.sample_lat_frenet(relative_time_points)

            # Return trajectory in Frenet coordinates
            fstates_2d = np.hstack((fstates_s, fstates_d))

        except Exception as e:
            raise type(e)(str(e) + '; called from SamplableWerlingTrajectory(%s) with time_points: %s' %
                          (self.__dict__, time_points)).with_traceback(sys.exc_info()[2])

        return fstates_2d

    def sample_frenet_once(self, time_point: float) -> FrenetState2D:
        """ convenience wrapper around self.sample_frenet """
        return self.sample_frenet(np.array([time_point]))[0]

    def sample_lon_frenet(self, relative_time_points: np.array) -> FrenetTrajectory1D:
        """
        Samples the appropriate longitudinal frenet states from the longitudinal polynomial w.r.t to the given
        relative time points
        :param relative_time_points: time points relative to the trajectory timestamp itself [s]
        :return:
        """
        assert min(relative_time_points) >= 0, "can't extrapolate back. relative_time_points=%s" % relative_time_points
        assert max(relative_time_points) <= self.T_extended, \
            'self.total_trajectory_time=%f, max(relative_time_points)=%f' % (self.T_extended, max(relative_time_points))

        in_traj_time_points = relative_time_points[relative_time_points <= self.T_s]
        extrapolated_time_points = relative_time_points[relative_time_points > self.T_s]

        fstates_list = []

        if len(in_traj_time_points) > 0:  # time points are part of the real polynomial part of the trajectory
            # assign values from <time_points> in s-axis polynomial
            fstates_list.append(QuinticPoly1D.polyval_with_derivatives(
                self.poly_s_coefs[np.newaxis, :], in_traj_time_points)[0])

        if len(extrapolated_time_points) > 0:
            # time points will trigger extrapolating the last sampled point from the polynomial using a constant
            # velocity predictor
            fstates_list.append(FrenetPredictionUtils.predict_1d_frenet_states(
                self.fstate_in_T_s, extrapolated_time_points - self.T_s)[0])

        return np.vstack(fstates_list)

    def sample_lat_frenet(self, relative_time_points: np.array) -> FrenetTrajectory1D:
        """
        Samples the appropriate lateral frenet states from the lateral polynomial w.r.t to the given
        relative time points
        :param relative_time_points: time points relative to the trajectory timestamp itself [s]
        :return:
        """
        assert min(relative_time_points) >= 0, "can't extrapolate back. relative_time_points=%s" % relative_time_points
        assert max(relative_time_points) <= self.T_extended, \
            'self.total_trajectory_time=%f, max(relative_time_points)=%f' % (self.T_extended, max(relative_time_points))

        in_traj_time_points = relative_time_points[relative_time_points <= self.T_d]
        extrapolated_time_points = relative_time_points[relative_time_points > self.T_d]

        fstates_list = []

        if len(in_traj_time_points) > 0:  # time points are part of the real polynomial part of the trajectory
            # assign values from <time_points> in s-axis polynomial
            fstates_list.append(QuinticPoly1D.polyval_with_derivatives(
                self.poly_d_coefs[np.newaxis, :], in_traj_time_points)[0])

        if len(extrapolated_time_points) > 0:
            # time points will trigger extrapolating the last sampled point from the polynomial using a constant
            # velocity predictor
            fstates_list.append(
                WerlingUtils.repeat_1d_state(
                    fstate=self.fstate_in_T_d,
                    repeats=len(extrapolated_time_points),
                    override_values=np.zeros(3),
                    override_mask=np.array([0, 1, 1])
                )
            )

        return np.vstack(fstates_list)

    def sample_lon_frenet_abs(self, abs_time_points: np.ndarray) -> FrenetTrajectory1D:
        return self.sample_lon_frenet(abs_time_points - self.timestamp_in_sec)

    def get_lon_sq_acc_at(self, timerange: Limits) -> float:
        return QuinticPoly1D.sq_acc_between(
            self.poly_s_coefs, timerange[LIMIT_MIN], min(timerange[LIMIT_MAX], self.T_s)).item()

    def get_lon_sq_jerk_at(self, timerange: Limits) -> float:
        return QuinticPoly1D.sq_jerk_between(
            self.poly_s_coefs, timerange[LIMIT_MIN], min(timerange[LIMIT_MAX], self.T_s)).item()

    def get_lat_sq_jerk_at(self, timerange: Limits) -> float:
        return QuinticPoly1D.sq_jerk_between(
            self.poly_d_coefs, timerange[LIMIT_MIN], min(timerange[LIMIT_MAX], self.T_d)).item()

    def get_lon_position_at_Td(self):
        return self.sample_lon_frenet(np.array([self.T_d]))[0, FS_SX]

    def get_lon_time_of_arrival(self, sx: float):
        """
        Get time to arrival to the longitudinal position <sx>. If <sx> is greater than the longitudinal position at
        self.T_s (the terminal pose of that trajectory), then, with accordance to how this class is built, the
        constant velocity of the terminal state (at T_s) is maintained.
        :param sx: absolute longitudinal position (frenet sx value) relative to self.gff_id
        :return: Time to arrival to <sx> in sec relative to self.timestamp_in_sec
        """
        sx_T = np.polyval(self.poly_s_coefs, self.T_s)
        if sx == sx_T:
            time = self.T_s
        elif sx < sx_T:
            optima = Math.find_real_roots_in_limits(self.poly_s_coefs - sx * np.array([0, 0, 0, 0, 0, 1]),
                                                    np.array([0, self.T_s]))
            time = np.fmin.reduce(optima)
        else:
            fstate_in_T_s = QuinticPoly1D.polyval_with_derivatives(
                self.poly_s_coefs[np.newaxis, :], np.array([self.T_s]))[0, 0]
            time = self.T_s + NumpyUtils.div((sx - fstate_in_T_s[FS_SX]), fstate_in_T_s[FS_SV])

        return time


class Utils:
    @staticmethod
    def sample_frenet(trajectories: List[SamplableWerlingTrajectory], time_points: np.ndarray) \
            -> FrenetTrajectories2D:
        """
        Sample longitudinal frenet coordinates from multiple trajectory at one 
        :param trajectories: a list of SamplableWerlingTrajectory to sample from
        :param time_points: 
        :return: a 3d numpy array (dim0 - trajectories, dim1 - timepoints, dim2 - frenet coordinates)
        """
        Utils.validate_sampling_query(trajectories, time_points)

        # Handle the longitudinal(s) axis
        fstates_s = Utils.sample_lon_frenet(trajectories, time_points, validate=False)
        # Now handle the lateral(d) axis:
        fstates_d = Utils.sample_lat_frenet(trajectories, time_points, validate=False)

        # Return trajectory in Frenet coordinates
        fstates_2d = np.dstack((fstates_s, fstates_d))

        return fstates_2d

    @staticmethod
    def sample_frenet_abs(trajectories: List[SamplableWerlingTrajectory], time_points: np.ndarray) \
            -> FrenetTrajectories2D:
        """
        Convenience method to wrap sample_frenet_relative for sampling time points in absolute values
        :param trajectories: a list of SamplableWerlingTrajectory to sample from
        :param time_points: an array of 2d time points (absolute times)
        :return: a 3d numpy array (dim0 - trajectories, dim1 - timepoints, dim2 - frenet coordinates)
        """
        trajectory_time = np.array([trajectory.timestamp_in_sec for trajectory in trajectories])
        relative_time_points = time_points - trajectory_time

        return Utils.sample_frenet(trajectories, relative_time_points)

    @staticmethod
    def sample_lon_frenet(trajectories: List[SamplableWerlingTrajectory], time_points: np.array, validate: bool = True) \
            -> FrenetTrajectories1D:
        """
        Sample longitudinal frenet coordinates from multiple trajectory at one
        :param trajectories: the list of trajectories to sample from
        :param time_points: a 2d numpy array - first dimension's legnth will be 1 to repeat the time points sampling
        over all trajectories, or of the trajectories list's length for zip a time point vector to its corresponding
        trajectory
        :param validate: bool. either to run validation of timestamps and trajectories or not to
        :return: a 3d numpy array (dim0 - trajectories, dim1 - timepoints, dim2 - longitudinal coordinates)
        """
        if validate:
            Utils.validate_sampling_query(trajectories, time_points)

        n, m = len(trajectories), time_points.shape[1]

        T_s = np.array([trajectory.T_s for trajectory in trajectories])
        poly_s_coefs = np.array([trajectory.poly_s_coefs for trajectory in trajectories])

        # sample all polynomials all relative timepoints (some of which will be overridden by extrapolations later)
        if time_points.shape[0] == 1:
            poly_samples = QuinticPoly1D.polyval_with_derivatives(poly_s_coefs, time_points[0])
        else:
            poly_samples = QuinticPoly1D.zip_polyval_with_derivatives(poly_s_coefs, time_points)

        # for each trajectory and timepoint, compute relative time to T_s
        time_delta_from_T_s = time_points - T_s[:, np.newaxis]

        # if no extrapolation needed, just return the samples from the polynomials
        if not np.any(time_delta_from_T_s > 0):
            return poly_samples

        # compute extrapolations
        sx_T_s, sv_T_s, _ = QuinticPoly1D.zip_polyval_with_derivatives(poly_s_coefs, T_s[:, np.newaxis])[:, 0, :].T
        extrapolated_sx = sx_T_s[:, np.newaxis] + sv_T_s[:, np.newaxis] * time_delta_from_T_s
        extrapolated_samples = np.dstack((extrapolated_sx, np.tile(sv_T_s[:, np.newaxis], m), np.zeros((n, m))))

        result = np.empty((n, m, 3))
        result[time_delta_from_T_s < 0] = poly_samples[time_delta_from_T_s < 0]
        result[time_delta_from_T_s >= 0] = extrapolated_samples[time_delta_from_T_s >= 0]

        return result

    @staticmethod
    def sample_lat_frenet(trajectories: List[SamplableWerlingTrajectory], time_points: np.array, validate: bool = True) \
            -> FrenetTrajectories1D:
        """
        Sample lateral frenet coordinates from multiple trajectory at one
        :param trajectories: the list of trajectories to sample from
        :param time_points: a 2d numpy array - first dimension's legnth will be 1 to repeat the time points sampling
        over all trajectories, or of the trajectories list's length for zip a time point vector to its corresponding
        trajectory
        :param validate: bool. either to run validation of timestamps and trajectories or not to
        :return: a 3d numpy array (dim0 - trajectories, dim1 - timepoints, dim2 - lateral coordinates)
        """
        if validate:
            Utils.validate_sampling_query(trajectories, time_points)

        n, m = len(trajectories), time_points.shape[1]

        T_d = np.array([trajectory.T_d for trajectory in trajectories])
        poly_d_coefs = np.array([trajectory.poly_d_coefs for trajectory in trajectories])

        # sample all polynomials all relative timepoints (some of which will be overridden by extrapolations later)
        if time_points.shape[0] == 1:
            poly_samples = QuinticPoly1D.polyval_with_derivatives(poly_d_coefs, time_points[0])
        else:
            poly_samples = QuinticPoly1D.zip_polyval_with_derivatives(poly_d_coefs, time_points)

        # for each trajectory and timepoint, compute relative time to T_d
        time_delta_from_T_d = time_points - T_d[:, np.newaxis]

        # if no extrapolation needed, just return the samples from the polynomials
        if not np.any(time_delta_from_T_d > 0):
            return poly_samples

        extrapolated_sx = QuinticPoly1D.zip_polyval(poly_d_coefs, T_d[:, np.newaxis])
        extrapolated_samples = np.dstack((np.tile(extrapolated_sx, m), np.zeros((n, m)), np.zeros((n, m))))

        result = np.empty((n, m, 3))
        result[time_delta_from_T_d < 0] = poly_samples[time_delta_from_T_d < 0]
        result[time_delta_from_T_d >= 0] = extrapolated_samples[time_delta_from_T_d >= 0]

        return result

    @staticmethod
    def validate_sampling_query(trajectories: List[SamplableWerlingTrajectory], time_points: np.array):
        """
        Validate the fit of time_points for trajectories, that is time is not lesser than zero and not greater than
        the max length of trajectory
        :param trajectories:
        :param time_points: 2d numpy array of timepoints for sampling (first dimension's length is either 1 or the
        length of the list of trajectories - each row will be sampled in corresponding trajectory)
        :return: True if all timestamps fit in the trajectories' sample-able time ranges
        """
        assert time_points.ndim == 2
        assert np.all(time_points >= 0), "can't extrapolate back. relative_time_points=%s" % time_points

        T_extended = np.array([trajectory.T_extended for trajectory in trajectories])
        assert np.all(time_points <= T_extended[:, np.newaxis]), "can't extrapolate beyond T_extended"
