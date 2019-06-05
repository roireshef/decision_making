import numpy as np
from decision_making.src.global_constants import FILTER_V_T_GRID, FILTER_V_0_GRID, BP_JERK_S_JERK_D_TIME_WEIGHTS, \
    LON_ACC_LIMITS
from decision_making.src.global_constants import MAX_CURVATURE
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel
from decision_making.src.planning.types import C_V, C_A, C_K, Limits, FrenetState2D, FS_SV, FS_SX
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
                                   lon_acceleration_limits: Limits, lat_acceleration_limits: Limits,
                                   desired_velocity: float) -> np.ndarray:
        """
        Given a set of trajectories in Cartesian coordinate-frame, it validates them against the following limits:
        longitudinal velocity, longitudinal acceleration, lateral acceleration (via curvature and lon. velocity)
        :param ctrajectories: CartesianExtendedTrajectories object of trajectories to validate
        :param velocity_limits: longitudinal velocity limits to test for in cartesian frame [m/sec]
        :param lon_acceleration_limits: longitudinal acceleration limits to test for in cartesian frame [m/sec^2]
        :param lat_acceleration_limits: lateral acceleration limits to test for in cartesian frame [m/sec^2]
        :param desired_velocity: desired longitudinal speed [m/sec]
        :return: A boolean numpy array, True where the respective trajectory is valid and false where it is filtered out
        """
        lon_acceleration = ctrajectories[:, :, C_A]
        lat_acceleration = ctrajectories[:, :, C_V] ** 2 * ctrajectories[:, :, C_K]
        lon_velocity = ctrajectories[:, :, C_V]
        curvature = ctrajectories[:, :, C_K]

        # validates the following behavior for each trajectory:
        # (1) applies negative jerk to reduce initial positive acceleration, if necessary
        #     (initial jerk is calculated by subtracting the first two acceleration samples)
        # (2) applies negative acceleration to reduce velocity until it reaches the desired velocity, if necessary
        # (3) keeps the velocity under the desired velocity limit.
        conforms_desired = np.logical_or(
            np.all(np.logical_or(lon_acceleration < 0, lon_velocity <= desired_velocity), axis=1),
            (lon_acceleration[:, 0] > lon_acceleration[:, 1]))

        # check velocity and acceleration limits
        # note: while we filter any trajectory that exceeds the velocity limit, we allow trajectories to break the
        #       desired velocity limit, as long as they slowdown towards the desired velocity.
        conforms_limits = np.all(NumpyUtils.is_in_limits(lon_velocity, velocity_limits) &
                                 NumpyUtils.is_in_limits(lon_acceleration, lon_acceleration_limits) &
                                 NumpyUtils.is_in_limits(lat_acceleration, lat_acceleration_limits) &
                                 NumpyUtils.is_in_limits(curvature, np.array([-MAX_CURVATURE, MAX_CURVATURE])), axis=1)

        conforms = np.logical_and(conforms_limits, conforms_desired)
        return conforms

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
    def create_linear_profile_polynomials(frenet_state: FrenetState2D) -> (np.ndarray, np.ndarray):
        """
        Given a frenet state, create two (s, d) polynomials that assume constant velocity (we keep the same momentary
        velocity). Those polynomials are degenerate to s(t)=v*t+x form
        :param frenet_state: the current frenet state to pull positions and velocities from
        :return: a tuple of (s(t), d(t)) polynomial coefficient arrays
        """
        poly_s = np.array([0, 0, 0, 0, frenet_state[FS_SV], frenet_state[FS_SX]])
        # We zero out the lateral polynomial because we strive for being in the lane center with zero lateral velocity
        poly_d = np.zeros(QuinticPoly1D.num_coefs())
        return poly_s, poly_d

    @staticmethod
    def create_ego_by_goal_state(goal_frenet_state: FrenetState2D, ego_to_goal_time: float) -> FrenetState2D:
        """
        calculate Frenet state in ego time, such that its constant-velocity prediction in goal time is goal_frenet_state
        :param goal_frenet_state: goal Frenet state
        :param ego_to_goal_time: the difference between the goal time and ego time
        :return: ego by goal frenet state
        """
        return np.array([goal_frenet_state[FS_SX] - ego_to_goal_time * goal_frenet_state[FS_SV],
                         goal_frenet_state[FS_SV], 0, 0, 0, 0])



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
        poly_coefs = poly.s_profile_coefficients(a_0, v_0, v_T, T)
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

