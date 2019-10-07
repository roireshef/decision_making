import numpy as np
from decision_making.src.global_constants import FILTER_V_T_GRID, FILTER_V_0_GRID, BP_JERK_S_JERK_D_TIME_WEIGHTS, \
    LON_ACC_LIMITS, EPS, NEGLIGIBLE_VELOCITY, TRAJECTORY_TIME_RESOLUTION, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, ActionSpec
from decision_making.src.planning.types import C_V, C_A, C_K, Limits, FrenetState2D, FS_SV, FS_SX, FrenetStates2D, S2, \
    FS_DX, S3, FS_SA
from decision_making.src.planning.types import CartesianExtendedTrajectories
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D, Poly1D


class KinematicUtils:
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

        first_non_zero = np.argmin(np.equal(poly_diff, 0))
        roots = Math.find_real_roots_in_limits(poly_diff[first_non_zero:], time_range)

        return np.all(np.greater(np.polyval(poly_diff, time_range), 0)) and np.all(np.isnan(roots))

    @staticmethod
    def filter_by_cartesian_limits(ctrajectories: CartesianExtendedTrajectories, velocity_limits: Limits,
                                   lon_acceleration_limits: Limits, lat_acceleration_limits: Limits) -> np.ndarray:
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
                                 NumpyUtils.is_in_limits(lat_acceleration, lat_acceleration_limits), axis=1)

        return conforms_limits

    @staticmethod
    def filter_by_velocity_limit(ctrajectories: CartesianExtendedTrajectories, velocity_limits: np.ndarray,
                                 T: np.array) -> np.array:
        """
        validates the following behavior for each trajectory:
        (1) applies negative jerk to reduce initial positive acceleration, if necessary
            (initial jerk is calculated by subtracting the first two acceleration samples)
        (2) applies negative acceleration to reduce velocity until it reaches the desired velocity, if necessary
        (3) keeps the velocity under the desired velocity limit.
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
            np.logical_or(
                # either speed is below limit, or vehicle is slowing down when it doesn't
                np.all(np.logical_or(lon_acceleration <= 0, lon_velocity <= velocity_limits + EPS), axis=1),
                # negative initial jerk
                lon_acceleration[:, 0] > lon_acceleration[:, 1]))

        return conforms_velocity_limits

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
    def create_linear_profile_polynomial_pair(frenet_state: FrenetState2D) -> (np.ndarray, np.ndarray):
        """
        Given a frenet state, create two (s, d) polynomials that assume constant velocity (we keep the same momentary
        velocity). Those polynomials are degenerate to s(t)=v*t+x form
        :param frenet_state: the current frenet state to pull positions and velocities from
        :return: a tuple of (s(t), d(t)) polynomial coefficient arrays
        """
        poly_s, poly_d = KinematicUtils.create_linear_profile_polynomial_pairs(frenet_state[np.newaxis])
        return poly_s[0], poly_d[0]

    @staticmethod
    def create_linear_profile_polynomial_pairs(frenet_states: FrenetStates2D) -> (np.ndarray, np.ndarray):
        """
        Given N frenet states, create two Nx6 matrices (s, d) of polynomials that assume constant velocity
        (we keep the same momentary velocity). Those polynomials are degenerate to s(t)=v*t+x form
        :param frenet_states: the current frenet states to pull positions and velocities from
        :return: a tuple of Nx6 matrices (s(t), d(t)) polynomial coefficient arrays
        """
        # zero 4 highest coefficients of poly_s: from x^5 until x^2 (including)
        poly_s = np.c_[np.zeros((frenet_states.shape[0], S3+1)), 0.5 * frenet_states[:, FS_SA], frenet_states[:, FS_SV], frenet_states[:, FS_SX]]
        # We zero out the lateral polynomial because we strive for being in the lane center with zero lateral velocity
        poly_d = np.zeros((frenet_states.shape[0], QuinticPoly1D.num_coefs()))
        return poly_s, poly_d

    @staticmethod
    def create_ego_by_goal_state(goal_frenet_state: FrenetState2D, ego_to_goal_time: float) -> FrenetState2D:
        """
        calculate Frenet state in ego time, such that its constant-velocity prediction in goal time is goal_frenet_state
        :param goal_frenet_state: goal Frenet state
        :param ego_to_goal_time: the difference between the goal time and ego time
        :return: ego by goal frenet state
        """
        return np.array([goal_frenet_state[FS_SX] - ego_to_goal_time * goal_frenet_state[FS_SV] -
                                    0.5 * ego_to_goal_time * ego_to_goal_time * goal_frenet_state[FS_SA],
                         goal_frenet_state[FS_SV] - ego_to_goal_time * goal_frenet_state[FS_SA],
                         goal_frenet_state[FS_SA], 0, 0, 0])

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
        poly_coefs_s[padding_mode], _ = KinematicUtils.create_linear_profile_polynomial_pairs(terminal_fstates[padding_mode])
        return poly_coefs_s, poly_coefs_d


class BrakingDistances:
    """
    Calculates braking distances
    """
    @staticmethod
    def create_braking_distances(aggresiveness_level: AggressivenessLevel=AggressivenessLevel.CALM.value) -> np.array:
        """
        Creates distances of all follow_lane with the given aggressiveness_level, braking actions with a0 = 0
        :return: the actions' distances
        """
        # create v0 & vT arrays for all braking actions
        v0, vT = np.meshgrid(FILTER_V_0_GRID.array, FILTER_V_T_GRID.array, indexing='ij')
        v0, vT = np.ravel(v0), np.ravel(vT)
        # calculate distances for braking actions
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggresiveness_level]
        distances = np.zeros_like(v0)
        distances[v0 > vT] = BrakingDistances._calc_actions_distances_for_given_weights(w_T, w_J, v0[v0 > vT],
                                                                                        vT[v0 > vT])
        return distances.reshape(len(FILTER_V_0_GRID), len(FILTER_V_T_GRID))

    @staticmethod
    def _calc_actions_distances_for_given_weights(w_T: np.array, w_J: np.array, v_0: np.array, v_T: np.array,
                                                  poly: Poly1D = QuarticPoly1D) -> np.array:
        """
        Calculate the distances for the given actions' weights and scenario params
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param v_0: array of initial velocities [m/s]
        :param v_T: array of desired final velocities [m/s]
        :param poly: The Poly1D (Quintic or Quartic) to use when checking the acceleration limits.
         Currently supporting only QuarticPoly1D
        :return: actions' distances; actions not meeting acceleration limits have infinite distance
        """
        # calculate actions' planning time
        a_0 = np.zeros_like(v_0)
        T = BrakingDistances.calc_T_s(w_T, w_J, v_0, a_0, v_T)

        # check acceleration limits
        if poly is not QuarticPoly1D:
            raise NotImplementedError('Currently function expects only QuarticPoly1Dsdf')
        # TODO: Once Quintic might be used, pull `s_profile_coefficients` method up
        poly_coefs = poly.position_profile_coefficients(a_0, v_0, v_T, T)
        in_limits = poly.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)

        # Calculate actions' distances, assuming a_0 = a_T = 0, and an average speed between v_0 an v_T.
        # Since the velocity profile is symmetric around the midpoint then the average velocity is (v_0 + v_T)/2 - this holds for Quartic.
        # Distances for accelerations which are not in limits are defined as infinity. This implied that braking on
        # invalid accelerations would take infinite distance, which in turn filters out these (invalid) action specs.
        distances = T * (v_0 + v_T) / 2
        distances[np.logical_not(in_limits)] = np.inf
        return distances

    @staticmethod
    def calc_T_s(w_T: float, w_J: float, v_0: np.array, a_0: np.array, v_T: np.array, poly: Poly1D = QuarticPoly1D):
        """
        given initial & end constraints and time-jerk weights, calculate longitudinal planning time
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param v_0: array of initial velocities [m/s]
        :param a_0: array of initial accelerations [m/s^2]
        :param v_T: array of final velocities [m/s]
        :param poly: The Poly1D (Quintic or Quartic) to use when checking the acceleration limits.
         Currently supporting only QuarticPoly1D
        :return: array of longitudinal trajectories' lengths (in seconds) for all sets of constraints
        """
        if poly is not QuarticPoly1D:
            raise NotImplementedError('Currently function expects only QuarticPoly1D')

        # Agent is in tracking mode, meaning the required velocity change is negligible and action time is actually
        # zero. This degenerate action is valid but can't be solved analytically.
        non_zero_actions = np.logical_not(poly.is_tracking_mode(v_0, v_T, a_0))

        w_T_array = np.full(v_0[non_zero_actions].shape, w_T)
        w_J_array = np.full(v_0[non_zero_actions].shape, w_J)

        # Get polynomial coefficients of time-jerk cost function derivative for our settings
        time_cost_derivative_poly_coefs = poly.time_cost_function_derivative_coefs(
            w_T_array, w_J_array, a_0[non_zero_actions], v_0[non_zero_actions], v_T[non_zero_actions])

        # Find roots of the polynomial in order to get extremum points
        cost_real_roots = Math.find_real_roots_in_limits(time_cost_derivative_poly_coefs, np.array([0, np.inf]))

        # return T as the minimal real root
        T = np.zeros_like(v_0)
        T[non_zero_actions] = np.fmin.reduce(cost_real_roots, axis=-1)
        return T

