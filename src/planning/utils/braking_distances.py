import numpy as np
from decision_making.src.global_constants import FILTER_V_T_GRID, FILTER_V_0_GRID, BP_JERK_S_JERK_D_TIME_WEIGHTS, \
    LON_ACC_LIMITS
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D


class BrakingDistances:
    """
    Calculates braking distances
    """
    @staticmethod
    def create_braking_distances(aggressiveness_level: AggressivenessLevel) -> np.array:
        """
        Creates distances of all follow_lane with the given aggressiveness_level.
        :return: the actions' distances
        """
        # create v0 & vT arrays for all braking actions
        v0, vT = np.meshgrid(FILTER_V_0_GRID.array, FILTER_V_T_GRID.array, indexing='ij')
        v0, vT = np.ravel(v0), np.ravel(vT)
        # calculate distances for braking actions
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggressiveness_level]
        distances = np.zeros_like(v0)
        distances[v0 > vT], _ = BrakingDistances.calc_quartic_action_distances(w_T, w_J, v0[v0 > vT], vT[v0 > vT])
        return distances.reshape(len(FILTER_V_0_GRID), len(FILTER_V_T_GRID))

    @staticmethod
    def calc_quartic_action_distances(w_T: np.array, w_J: np.array, v_0: np.array, v_T: np.array,
                                      a_0: np.array = None) -> [np.array, np.array]:
        """
        Calculate the distances and times for the given actions' weights and scenario params.
        Actions not meeting the acceleration limits have infinite distance and time.
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param v_0: array of initial velocities [m/s]
        :param v_T: array of desired final velocities [m/s]
        :param a_0: array of initial accelerations [m/s^2]
        :return: two arrays: actions' distances and times; actions not meeting acceleration limits have infinite distance
        """
        # calculate actions' planning time
        if a_0 is None:
            a_0 = np.zeros_like(v_0)
        T = BrakingDistances.calc_T_s_for_quartic(w_T, w_J, v_0, a_0, v_T)
        non_zero = ~np.isclose(T, 0)

        # check acceleration limits
        s_profile_coefs = QuarticPoly1D.position_profile_coefficients(a_0[non_zero], v_0[non_zero], v_T[non_zero], T[non_zero])
        in_limits = QuarticPoly1D.are_accelerations_in_limits(s_profile_coefs, T[non_zero], LON_ACC_LIMITS)

        # Distances for accelerations which are not in limits are defined as infinity. This implied that braking on
        # invalid accelerations would take infinite distance, which in turn filters out these (invalid) action specs.
        distances = np.zeros_like(T)
        distances[non_zero] = Math.zip_polyval2d(s_profile_coefs, T[non_zero, np.newaxis])[:, 0]
        distances[non_zero][~in_limits] = np.inf
        T[non_zero][~in_limits] = np.inf

        return distances, T

    @staticmethod
    def calc_T_s_for_quartic(w_T: float, w_J: float, v_0: np.array, a_0: np.array, v_T: np.array):
        """
        given initial & end constraints and time-jerk weights, calculate longitudinal planning time
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param v_0: array of initial velocities [m/s]
        :param a_0: array of initial accelerations [m/s^2]
        :param v_T: array of final velocities [m/s]
        :return: array of longitudinal trajectories' lengths (in seconds) for all sets of constraints
        """
        # Agent is in tracking mode, meaning the required velocity change is negligible and action time is actually
        # zero. This degenerate action is valid but can't be solved analytically.
        non_zero_actions = np.logical_not(QuarticPoly1D.is_tracking_mode(v_0, v_T, a_0))

        w_T_array = np.full(v_0[non_zero_actions].shape, w_T)
        w_J_array = np.full(v_0[non_zero_actions].shape, w_J)

        # Get polynomial coefficients of time-jerk cost function derivative for our settings
        time_cost_derivative_poly_coefs = QuarticPoly1D.time_cost_function_derivative_coefs(
            w_T_array, w_J_array, a_0[non_zero_actions], v_0[non_zero_actions], v_T[non_zero_actions])

        # Find roots of the polynomial in order to get extremum points
        cost_real_roots = Math.find_real_roots_in_limits(time_cost_derivative_poly_coefs, np.array([0, np.inf]))

        # return T as the minimal real root
        T = np.zeros_like(v_0)
        T[non_zero_actions] = np.fmin.reduce(cost_real_roots, axis=-1)
        return T

