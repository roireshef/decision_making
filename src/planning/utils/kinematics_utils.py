import numpy as np
from decision_making.src.global_constants import EPS, NEGLIGIBLE_VELOCITY, TRAJECTORY_TIME_RESOLUTION, \
    MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, \
    SPEEDING_VIOLATION_TIME_TH, SPEEDING_SPEED_TH, BP_JERK_S_JERK_D_TIME_WEIGHTS, BP_ACTION_T_LIMITS
from decision_making.src.planning.types import C_V, C_A, C_K, Limits, FS_SX, FS_DX, Limits2D, RangedLimits2D, FrenetTrajectories2D, LIMIT_MAX
from decision_making.src.planning.types import CartesianExtendedTrajectories
from decision_making.src.planning.utils.frenet_utils import FrenetUtils
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.utils.map_utils import MapUtils


class KinematicUtils:
    @staticmethod
    def get_lateral_acceleration_limit_by_curvature(curvatures: np.ndarray, ranged_limits: RangedLimits2D):
        """
        takes a 1d array of curvature values and compares them against the acceleration limits table per curvature,
        to get the lateral acceleration limit corresponding to those curvature values. The acceleration limits are given
        by a table <ranged_limits> that represents a piecewise-linear function, so for each curvature value in
        <curvatures> we look for the radius (inverse of curvature) range and within that range we interpolate linearly
        given the acceleration limits on the boundaries of that range
        :param curvatures: numpy array of curvature values (any shape)
        :param ranged_limits: ranged limits (radius ranges -> acceleration limits at those ranges)
        :return: 1d array of lateral acceleration limits
        """
        # extract column-vectors from the range_limits matrix
        min_radius, max_radius, min_accels, max_accels = np.split(ranged_limits, 4, axis=1)

        # flatten and compute inverse of curvatures to get 1D array of radii
        radii_1d = 1 / np.maximum(np.abs(curvatures.ravel()), EPS)

        # compute the lateral acceleration slope for each radius range (that is, for each range, compute a of ax+b)
        slopes = (max_accels - min_accels) / (max_radius - min_radius)

        # for each radius in <radii_1d>, get the index of the corresponding range in <ranged_limits>
        row_idxs = np.argmin(max_radius <= radii_1d, axis=0)

        # simple a*(X-x0) + value(x0) formula relative to the lower side of the relevant range
        delta_radius = radii_1d[:, np.newaxis] - min_radius[row_idxs]
        acceleration_limit = slopes[row_idxs] * delta_radius + min_accels[row_idxs]

        # when taking radius_of_turn==inf, <delta_radius> is nan and therefor <acceleration_limit>. In this spacial case
        # the acceleration limit at the upper side of the range is taken
        is_inf = np.isinf(radii_1d[:, np.newaxis])
        acceleration_limit[is_inf] = max_accels[row_idxs][is_inf]

        # lookup the correct range for each radius in <radii_1d> and compute ax+b to get interpolated lateral acc limit
        return np.reshape(acceleration_limit, curvatures.shape)

    @staticmethod
    def is_maintaining_distance(poly_host: np.array, poly_target: np.array, margin: float, headway: float, time_range: Limits):
        """
        Given two s(t) longitudinal polynomials (one for host, one for target), this function checks if host maintains
        at least a distance of margin + headway (time * host_velocity) in the time range specified by <time_range>.
        :param poly_host: 1d numpy array - coefficients of host's polynomial s(t)
        :param poly_target: 1d numpy array - coefficients of target's polynomial s(t)
        :param margin: the minimal stopping distance to keep in meters (in addition to headway, highly relevant for stopping)
        :param headway: the time to use for the headway formula: time*velocity = distance to keep.
        :param time_range: the relevant range of t for checking the polynomials, i.e. [0, T]
        :return: boolean - True if host maintains proper distance from target, False otherwise
        """
        # coefficients of host vehicle velocity v_h(t) of host
        vel_poly = np.polyder(poly_host, 1)

        # poly_diff is the polynomial of the distance between poly2 and poly1 with subtracting the required distance also
        poly_diff = poly_target - poly_host
        poly_diff[-1] -= margin

        # add headway
        poly_diff[1:] -= vel_poly * headway

        roots = Math.find_real_roots_in_limits(poly_diff, time_range)

        return np.all(np.greater(np.polyval(poly_diff, time_range), 0)) and np.all(np.isnan(roots))

    @staticmethod
    def filter_by_cartesian_limits(ctrajectories: CartesianExtendedTrajectories, velocity_limits: Limits,
                                   lon_acceleration_limits: Limits, lat_acceleration_limits: Limits2D) -> np.ndarray:
        """
        Given a set of trajectories in Cartesian coordinate-frame, it validates them against the following limits:
        longitudinal velocity, longitudinal acceleration, lateral acceleration (via curvature and lon. velocity)
        :param ctrajectories: CartesianExtendedTrajectories object of trajectories to validate
        :param velocity_limits: longitudinal velocity limits to test for in cartesian frame [m/sec]
        :param lon_acceleration_limits: longitudinal acceleration limits to test for in cartesian frame [m/sec^2]
        :param lat_acceleration_limits: lateral acceleration limits to test for in cartesian frame [m/sec^2]
        :return: 1D boolean np array, True where the respective trajectory is valid and false where it is filtered out
        """
        lon_acceleration = ctrajectories[:, :, C_A]
        lat_acceleration = ctrajectories[:, :, C_V] ** 2 * ctrajectories[:, :, C_K]
        lon_velocity = ctrajectories[:, :, C_V]

        # check velocity and acceleration limits
        # note: while we filter any trajectory that exceeds the velocity limit, we allow trajectories to break the
        #       desired velocity limit, as long as they slowdown towards the desired velocity.
        conforms_limits = np.all(NumpyUtils.is_in_limits(lon_velocity, velocity_limits) &
                                 NumpyUtils.is_in_limits(lon_acceleration, lon_acceleration_limits) &
                                 NumpyUtils.zip_is_in_limits(lat_acceleration, lat_acceleration_limits), axis=1)

        return conforms_limits

    @staticmethod
    def filter_by_relative_lateral_acceleration_limits(ftrajectories: FrenetTrajectories2D, ctrajectories: CartesianExtendedTrajectories,
                                                       relative_lat_acceleration_limits: Limits,
                                                       baseline_lat_accel_curve_control: np.array,
                                                       reference_route: GeneralizedFrenetSerretFrame) -> np.ndarray:
        """
        Given a set of trajectories in Cartesian coordinate-frame, this filter checks to see if the lateral accelerations
        at every point is within some range of the lateral accelerations that would have been experienced by following
        the ftrajectories on the reference_route.
        :param ftrajectories: FrenetTrajectories2D object to use while checking relative lat. accel. limit
        :param ctrajectories: CartesianExtendedTrajectories object of trajectories to validate
        :param baseline_lat_accel_curve_control: lateral acceleration limits to test for in cartesian frame [m/sec^2]
        given by road's curvature (if only doing curve speed control - those will be the effective accelerations)
        :param relative_lat_acceleration_limits: lat. accel. limits relative to baseline lat. accel. [m/sec^2]
        :param reference_route: GFF used for calculating baseline lat. accel.
        :return: 1D boolean np array, True where the respective trajectory is valid and false where it is filtered out
        """
        lane_speed_limits = {lane_id: MapUtils.get_lane(lane_id).e_v_nominal_speed for lane_id in reference_route.segment_ids}

        # get baseline lane's speed_limits for each trajectory point
        lane_segment_ids, _ = reference_route.convert_to_segment_states(ftrajectories)
        baseline_speed_limits = np.vectorize(lane_speed_limits.get)(lane_segment_ids)

        # get road-curvatures on the target GFF (baseline)
        baseline_curvatures = reference_route.get_curvature(ftrajectories[:, :, FS_SX])

        # calculate baseline *absolute* lat acceleration (if driving on target GFF) and ONLY keeping speed limits
        baseline_lat_accel_speed_limit = baseline_speed_limits ** 2 * np.abs(baseline_curvatures)

        # calculate baseline lat acceleration (if driving on target GFF) and doing
        # BOTH curve speed control and keeping speed limits
        baseline_lat_accel = np.sign(baseline_curvatures) * np.minimum(baseline_lat_accel_speed_limit,
                                                                       baseline_lat_accel_curve_control)

        # compare target lat accels to baseline lat accels
        effective_lat_accel = ctrajectories[:, :, C_V] ** 2 * ctrajectories[:, :, C_K]

        # return true if effective and baseline accelerations are of the same sign and effective is at most MARGIN
        # bigger (in absolute value) than baseline, or if they have different sign, than if effective acceleration is
        # at most MARGIN
        lat_accel_difference_same_sign = (np.sign(effective_lat_accel) == np.sign(baseline_lat_accel)) & \
                                         (np.abs(effective_lat_accel) - np.abs(baseline_lat_accel) < relative_lat_acceleration_limits[LIMIT_MAX])

        lat_accel_difference_diff_sign = (np.sign(effective_lat_accel) != np.sign(baseline_lat_accel)) & \
                                         (np.abs(effective_lat_accel) < relative_lat_acceleration_limits[LIMIT_MAX])

        conforms_rel_lat_accel_limits = np.all(np.logical_or(lat_accel_difference_same_sign,
                                                             lat_accel_difference_diff_sign), axis=1)

        return conforms_rel_lat_accel_limits

    @staticmethod
    def filter_by_velocity_limit(ctrajectories: CartesianExtendedTrajectories, velocity_limits: np.ndarray,
                                 T: np.array) -> np.array:
        """
        validates the following behavior for each trajectory:
        (1) applies negative jerk to reduce initial positive acceleration, if necessary
            (initial jerk is calculated by subtracting the first two acceleration samples)
        (2) applies negative acceleration to reduce velocity until it reaches the desired velocity, if necessary
        (3) keeps the velocity under the desired velocity limit.
        Note: This method assumes velocities beyond the spec.t are set below the limit (e.g. to 0) by the callee
        :param ctrajectories: CartesianExtendedTrajectories object of trajectories to validate
        :param velocity_limits: 2D matrix [trajectories, timestamps] of nominal velocities to validate against
        :param T: array of target times for ctrajectories
        :return: 1D boolean np array, True where the respective trajectory is valid and false where it is filtered out
        """
        lon_acceleration = ctrajectories[:, :, C_A]
        lon_velocity = ctrajectories[:, :, C_V]
        last_pad_idxs = KinematicUtils.convert_padded_spec_time_to_index(T)
        last_pad_idxs = np.minimum(last_pad_idxs, ctrajectories.shape[1] - 1)
        # for each trajectory use the appropriate last time index (possibly after padding)
        end_velocities = ctrajectories[np.arange(ctrajectories.shape[0]), last_pad_idxs, C_V]
        end_velocity_limits = velocity_limits[np.arange(ctrajectories.shape[0]), last_pad_idxs]

        # TODO: velocity comparison is temporarily done with an EPS margin, due to numerical issues
        conforms_velocity_limits = np.logical_and(
            end_velocities <= end_velocity_limits + NEGLIGIBLE_VELOCITY,  # final speed must comply with limits
            np.all(KinematicUtils._speeding_within_allowed_limits(lon_velocity, lon_acceleration, velocity_limits),
                   axis=1))

        return conforms_velocity_limits

    @staticmethod
    def _speeding_within_allowed_limits(lon_velocity: np.array, lon_acceleration: np.array,
                                        velocity_limits: np.ndarray) -> np.array:
        """
        speeding is within allowed limits if it does not violate the speed limit for more than VIOLATION_TIME_TH.
        Furthermore it does so by no more than VIOLATION_SPEED_TH, unless starting velocity is above this value.
        Note: This method assumes velocities beyond the spec.t are set below the limit (e.g. to 0) by the callee
        :param lon_velocity: trajectories velocities
        :param lon_acceleration: trajectories accelerations
        :return:
        """
        # anywhere speed is below limit
        speeding_is_within_limits = lon_velocity <= velocity_limits + NEGLIGIBLE_VELOCITY
        # or violation is limited to first SPEEDING_VIOLATION_TIME_TH seconds,last_allowed_idx
        last_allowed_idx = int(min(SPEEDING_VIOLATION_TIME_TH, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON) /
                           TRAJECTORY_TIME_RESOLUTION)
        # and vehicle is slowing down when it doesn't,
        is_decelerating = lon_acceleration[:, 0:last_allowed_idx] <= 0
        # or speed limit is exceeded by no more than SPEEDING_SPEED_TH
        is_within_allowed_speed_violation = lon_velocity[:, 0:last_allowed_idx] <= \
            velocity_limits[:, 0:last_allowed_idx] + SPEEDING_SPEED_TH
        # or we were above this value to start with, and jerk is negative
        was_violating_and_jerk_negative = np.logical_and(lon_velocity[:, 0] > velocity_limits[:, 0] + SPEEDING_SPEED_TH,
                                                         lon_acceleration[:, 0] > lon_acceleration[:, 1])[:, np.newaxis]
        speeding_is_within_limits[:, 0:last_allowed_idx] |= \
            is_decelerating | is_within_allowed_speed_violation | was_violating_and_jerk_negative
        return speeding_is_within_limits

    @staticmethod
    def convert_padded_spec_time_to_index(T: np.array):
        return (np.maximum(T, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON) / TRAJECTORY_TIME_RESOLUTION).astype(int)

    @staticmethod
    # TODO: add jerk to filter?
    def filter_by_longitudinal_frenet_limits(poly_coefs_s: np.ndarray, T_s_vals: np.ndarray,
                                             lon_acceleration_limits: Limits,
                                             lon_velocity_limits: Limits,
                                             reference_route_limits: Limits) -> np.ndarray:
        """
        Given a set of trajectories in Frenet coordinate-frame, it validates them against the following limits:
        (longitudinal progress on the frenet frame curve, positive longitudinal velocity)
        :param poly_coefs_s: 2D matrix of solutions (1st dim), each one is a vector of coefficients of a longitudinal
        s(t) polynomial (2nd dim), with t in the range [0, T] (T specified in T_s_vals)
        :param T_s_vals: 1d numpy array - the T for the polynomials in <poly_coefs_s>
        :param lon_acceleration_limits: acceleration limits to test the trajectories keep
        :param lon_velocity_limits: velocity limits to test the trajectories keep
        :param reference_route_limits: the minimal and maximal progress (s value) on the reference route used
        in the frenet frame used for planning
        :return: A boolean numpy array, True where the respective trajectory is valid and false where it is filtered out
        """
        # validate the progress on the reference-route curve doesn't extrapolate, and that velocity is non-negative
        conforms = \
            QuinticPoly1D.are_accelerations_in_limits(poly_coefs_s, T_s_vals, lon_acceleration_limits) & \
            QuinticPoly1D.are_velocities_in_limits(poly_coefs_s, T_s_vals, lon_velocity_limits) & \
            QuinticPoly1D.are_derivatives_in_limits(0, poly_coefs_s, T_s_vals, reference_route_limits)

        return conforms

    @staticmethod
    def filter_by_lateral_frenet_limits(poly_coefs_d: np.ndarray, T_d_vals: np.ndarray,
                                        lat_acceleration_limits: Limits) -> np.ndarray:
        """
        Given a set of trajectories in Frenet coordinate-frame, it validates that the acceleration of the lateral
        polynomial used in planning is in the allowed limits.
        :param poly_coefs_d: 2D numpy array of each row has 6 poly coefficients of lateral polynomial
        :param lat_acceleration_limits: lateral acceleration limits to test for in frenet frame [m/sec^2]
        :param T_d_vals: 1D numpy array with lateral planning-horizons (correspond to each trajectory)
        :return: A boolean numpy array, True where the respective trajectory is valid and false where it is filtered out
        """
        # here we validate feasible lateral acceleration *directly from the lateral polynomial* because our
        # discretization of the trajectory (via sampling with constant self.dt) can overlook cases where there is a high
        # lateral acceleration between two adjacent sampled points (critical in the lateral case because we allow
        # shorter lateral maneuvers
        frenet_lateral_movement_is_feasible = \
            QuinticPoly1D.are_accelerations_in_limits(poly_coefs_d, T_d_vals, lat_acceleration_limits)

        return frenet_lateral_movement_is_feasible

    @staticmethod
    def calc_poly_coefs(T: np.array, initial_fstates, terminal_fstates, padding_mode: np.array) -> [np.array, np.array]:
        """
        Given initial and end constraints for multiple actions and their time horizons, calculate polynomials,
        describing s and optionally d profiles.
        :param T: 1D array. Actions' time horizons
        :param initial_fstates: 2D matrix Nx6 or Nx3. Initial constraints for s and optionally for d
        :param terminal_fstates: e2D matrix Nx6 or Nx3. End constraints for s and optionally for d
        :param padding_mode: 1D boolean array. True if an action is in padding mode (shorter than 0.1)
        :return: 2D array of actions' polynomials for s and for d. If d is not given in the constraints, poly_d is None.
        """
        # allocate polynomials for s and optionally for d if d constraints are given
        poly_coefs_s = np.empty(shape=(len(T), QuinticPoly1D.num_coefs()), dtype=np.float)
        poly_coefs_d = np.zeros(shape=(len(T), QuinticPoly1D.num_coefs()), dtype=np.float) \
            if initial_fstates.shape[1] > FS_DX else None

        not_padding_mode = ~padding_mode
        if not_padding_mode.any():
            # generate a matrix that is used to find jerk-optimal polynomial coefficients
            A_inv = QuinticPoly1D.inverse_time_constraints_tensor(T[not_padding_mode])

            # solve for s
            constraints_s = np.c_[(initial_fstates[not_padding_mode, :FS_DX], terminal_fstates[not_padding_mode, :FS_DX])]
            poly_coefs_s[not_padding_mode] = QuinticPoly1D.zip_solve(A_inv, constraints_s)

            # solve for d if the constraints are given also for d dimension
            if poly_coefs_d is not None:
                constraints_d = np.c_[(initial_fstates[not_padding_mode, FS_DX:], terminal_fstates[not_padding_mode, FS_DX:])]
                poly_coefs_d[not_padding_mode] = QuinticPoly1D.zip_solve(A_inv, constraints_d)

        # create linear polynomials for padding mode
        poly_coefs_s[padding_mode], _ = FrenetUtils.create_linear_profile_polynomial_pairs(terminal_fstates[padding_mode])
        return poly_coefs_s, poly_coefs_d

    @staticmethod
    def specify_lateral_planning_time(a_0: np.array, v_0: np.array, dx: np.array) -> np.array:
        """
        Calculate lateral planning times by time-jerk cost optimization. Here we choose the calmest aggressiveness level.
        :param a_0: initial lateral acceleration in Frenet frame
        :param v_0: initial lateral velocity in Frenet frame
        :param dx: array or scalar lateral distance to the target in Frenet frame
        :return: lateral planning times of the same size like dx or array of size 1 if dx is scalar.
        """
        # choose the calmest lateral aggressiveness level
        weights = np.tile(BP_JERK_S_JERK_D_TIME_WEIGHTS[0], (1 if np.isscalar(dx) else dx.shape[0], 1))

        cost_coeffs_d = QuinticPoly1D.time_cost_function_derivative_coefs(
            w_T=weights[:, 2], w_J=weights[:, 1], a_0=a_0, v_0=v_0, v_T=0, dx=dx, T_m=0)
        roots_d = Math.find_real_roots_in_limits(cost_coeffs_d, BP_ACTION_T_LIMITS)
        return np.fmin.reduce(roots_d, axis=-1)
