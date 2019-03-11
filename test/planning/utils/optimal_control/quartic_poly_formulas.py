import numpy as np
import sympy as sp
from sympy import symbols
from sympy.matrices import *

from decision_making.paths import Paths
from decision_making.src.global_constants import LON_ACC_LIMITS, VELOCITY_LIMITS, \
    BP_ACTION_T_LIMITS, EPS
from decision_making.src.planning.behavioral.data_objects import ActionType
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import UniformGrid
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D


class QuarticMotionSymbolicsCreator:
    @staticmethod
    def create_symbolic_quartic_motion_equations():
        """
        This function uses symbolic package SymPy and computes the motion equations and cost function used for analysis of
        trajectories (in e.g. desmos) , action specification and filtering.
        """
        T = symbols('T')
        t = symbols('t')

        s0, v0, a0, vT, aT = symbols('s_0 v_0 a_0 v_T a_T')

        A = Matrix([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 2, 0, 0],
            [4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
            [12 * T ** 2, 6 * T, 2, 0, 0]
        ])

        # solve to get solution
        # this assumes a0=aT==0 (constant velocity)
        [c4, c3, c2, c1, c0] = A.inv() * Matrix([s0, v0, a0, vT, aT])

        x_t = (c4 * t ** 4 + c3 * t ** 3 + c2 * t ** 2 + c1 * t + c0).simplify()
        v_t = sp.diff(x_t, t).simplify()
        a_t = sp.diff(v_t, t).simplify()
        j_t = sp.diff(a_t, t).simplify()

        J = sp.integrate(j_t ** 2, (t, 0, T)).simplify()

        wJ, wT = symbols('w_J w_T')

        cost = (wJ * J + wT * T).simplify()

        package_cost = cost.subs(s0, 0).subs(aT, 0).simplify()
        package_cost_diff = sp.diff(package_cost, T).simplify()
        package_x_t = x_t.subs(s0, 0).subs(aT, 0).simplify()
        package_v_t = v_t.subs(s0, 0).subs(aT, 0).simplify()
        package_a_t = sp.diff(package_v_t, t).simplify()

        desmos_cost = package_cost.subs(a0, 0).simplify()
        desmos_cost_diff = package_cost_diff.subs(a0, 0).simplify()
        desmos_x_t = package_x_t.subs(a0, 0).simplify()
        desmos_v_t = package_v_t.subs(a0, 0).simplify()
        desmos_a_t = package_a_t.subs(a0, 0).simplify()

        return package_x_t, package_v_t, package_a_t, desmos_cost, desmos_cost_diff, desmos_x_t, desmos_v_t, desmos_a_t


class QuarticMotionPredicatesCreator:
    """This class creates predicates for filtering trajectories before specification according to initial velocity and
     acceleration and desired velocity"""

    def __init__(self, v0_grid: UniformGrid, a0_grid: UniformGrid, vT_grid: UniformGrid,
                 predicates_resources_target_directory: str):
        """
        :param v0_grid: A grid of initial velocities by which the predicates will be created (typically constant)
        :param a0_grid: A grid of initial accelerations by which the predicates will be created (typically constant)
        :param vT_grid: A grid of final velocities by which the predicates will be created (typically constant)
        :param predicates_resources_target_directory: A target directory inside resources directory where the predicates
                will be created(typically constant)
        """
        self.v0_grid = v0_grid
        self.a0_grid = a0_grid
        self.vT_grid = vT_grid

        self.predicates_resources_target_directory = predicates_resources_target_directory  # 'predicates'

    @staticmethod
    def create_quartic_motion_funcs(a_0, v_0, v_T, T):
        """
        :param a_0: initial acceleration [m/s^2]
        :param v_0: initial velocity [m/s^2]
        :param v_T: desired velocity [m/s^2]
        :param T: action_time [s]
        :return: lambda functions of velocity and acceleration w.r.t time (valid in the range [0,T])
        """
        return QuarticPoly1D.velocity_profile_function(a_0, v_0, v_T, T), \
               QuarticPoly1D.acceleration_profile_function(a_0, v_0, v_T, T)

    @staticmethod
    def generate_predicate_value(w_T, w_J, a_0: np.array, v_0: np.array, v_T: np.array) -> [np.array, np.array]:
        """
        Generates the actual predicate value (true/false) for the given action,weights and scenario params
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param a_0: array of initial accelerations [m/s^2]
        :param v_0: array of initial velocities [m/s]
        :param v_T: array of desired final velocities [m/s]
        :return: 1. True if given parameters will generate a feasible trajectory that meets time, velocity and
                    acceleration constraints and doesn't get into target vehicle safety zone.
                 2. Actions time (T_s)
        """
        T = QuarticMotionPredicatesCreator.calc_T_s(w_T, w_J, v_0, a_0, v_T)
        is_in_limits = (T == 0)  # zero actions are valid

        # get indices of non-nan positive T values; for nan values of T, is_in_limits = False
        valid_non_zero = np.logical_and(T > 0, T <= BP_ACTION_T_LIMITS[1])
        if not valid_non_zero.any():
            return is_in_limits  # only zero actions are valid

        # check actions validity: velocity & acceleration limits
        is_in_limits[valid_non_zero], _ = QuarticMotionPredicatesCreator.check_action_limits(
            T[valid_non_zero], v_0[valid_non_zero], v_T[valid_non_zero], a_0[valid_non_zero])
        return is_in_limits, T

    @staticmethod
    def generate_actions_distances(w_T, w_J, a_0: np.array, v_0: np.array, v_T: np.array) -> np.array:
        """
        Generates the actual predicate value (true/false) for the given action,weights and scenario params
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param a_0: array of initial accelerations [m/s^2]
        :param v_0: array of initial velocities [m/s]
        :param v_T: array of desired final velocities [m/s]
        :return: True if given parameters will generate a feasible trajectory that meets time, velocity and
                acceleration constraints and doesn't get into target vehicle safety zone.
        """
        is_in_limits, T = QuarticMotionPredicatesCreator.generate_predicate_value(w_T, w_J, a_0, v_0, v_T)
        non_zero_in_limits = np.logical_and(is_in_limits, T > 0)

        distances = np.full(T.shape, np.inf)
        distances[T == 0] = 0
        distances[non_zero_in_limits] = np.array([QuarticPoly1D.distance_profile_function(a_0[i], v_0[i], v_T[i], T[i])(T[i])
                                                  for i, Ti in enumerate(T) if non_zero_in_limits[i]])
        return distances

    @staticmethod
    def calc_T_s(w_T: float, w_J: float, v_0: np.array, a_0: np.array, v_T: np.array):
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
        non_zero_actions = np.logical_not(np.logical_and(np.isclose(v_0, v_T, atol=1e-3, rtol=0),
                                                         np.isclose(a_0, 0.0, atol=1e-3, rtol=0)))
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

    @staticmethod
    def check_action_limits(T: np.array, v_0: np.array, v_T: np.array, a_0: np.array) -> [np.array, np.array]:
        """
        Given longitudinal action dynamics, calculate validity wrt velocity & acceleration limits, and safety of each action.
        :param T: array of action times
        :param v_0: array of initial velocities
        :param v_T: array of final velocities
        :param a_0: array of initial acceleration
        :return: (1) boolean array: are velocity & acceleration in limits,
                 (2) boolean array: is the baseline trajectory safe
                 (3) coefficients of s polynomial
        """
        poly_coefs = QuarticPoly1D.s_profile_coefficients(a_0, v_0, v_T, T)
        # check acc & vel limits
        poly_coefs[np.where(poly_coefs[:, 0] == 0), 0] = EPS  # keep the polynomials to be quartic
        acc_in_limits = QuarticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)
        vel_in_limits = QuarticPoly1D.are_velocities_in_limits(poly_coefs, T, VELOCITY_LIMITS)
        is_in_limits = np.logical_and(acc_in_limits, vel_in_limits)
        return is_in_limits, poly_coefs

    def create_predicates(self, jerk_time_weights: np.ndarray) -> None:
        """
        Creates predicates for the jerk-time weights and the follow_lane static action
        :param jerk_time_weights: a 2-dimensional of shape [Kx3] where its rows are different sets of weights and each
                set of weights is built from 3 terms :  longitudinal jerk, latitudinal jerk and action time weights.
        :return:
        """
        action_type = ActionType.FOLLOW_LANE

        v0, a0, vT = np.meshgrid(self.v0_grid.array, self.a0_grid.array, self.vT_grid.array, indexing='ij')
        v0, a0, vT = np.ravel(v0), np.ravel(a0), np.ravel(vT)

        print('Save quartic actions limits...')
        for wi, weight in enumerate(jerk_time_weights):
            w_J, w_T = weight[0], weight[2]  # w_T stays the same (0.1), w_J is now to be one of [12,2,0.01]
            print('weights are: %.4f,%.4f' % (w_J, w_T))

            limits, _ = QuarticMotionPredicatesCreator.generate_predicate_value(w_T, w_J, a0, v0, vT)

            # save 'limits' predicate to file
            output_predicate_file_name = '%s_limits_wT_%.4f_wJ_%.4f.bin' % (action_type.name.lower(), w_T, w_J)
            output_predicate_file_path = Paths.get_resource_absolute_path_filename(
                '%s/%s' % (self.predicates_resources_target_directory,
                           output_predicate_file_name))
            BinaryReadWrite.save(array=limits.reshape(self.v0_grid.length, self.a0_grid.length, self.vT_grid.length),
                                 file_path=output_predicate_file_path)

    def create_actions_distances(self, jerk_time_weights: np.ndarray) -> None:
        """
        Creates predicates for the jerk-time weights and the follow_lane static action
        :param jerk_time_weights: a 2-dimensional of shape [Kx3] where its rows are different sets of weights and each
                set of weights is built from 3 terms :  longitudinal jerk, latitudinal jerk and action time weights.
        :return:
        """
        action_type = ActionType.FOLLOW_LANE

        v0, a0, vT = np.meshgrid(self.v0_grid.array, self.a0_grid.array, self.vT_grid.array, indexing='ij')
        v0, a0, vT = np.ravel(v0), np.ravel(a0), np.ravel(vT)

        print('Save quartic actions distances...')
        for wi, weight in enumerate(jerk_time_weights):
            w_J, w_T = weight[0], weight[2]  # w_T stays the same (0.1), w_J is now to be one of [12,2,0.01]
            print('weights are: %.4f,%.4f' % (w_J, w_T))

            distances = QuarticMotionPredicatesCreator.generate_actions_distances(w_T, w_J, a0, v0, vT)

            # save actions distances to file
            output_distances_file_name = '%s_distances_wT_%.4f_wJ_%.4f.bin' % (action_type.name.lower(), w_T, w_J)
            output_distances_file_path = Paths.get_resource_absolute_path_filename(
                '%s/%s' % (self.predicates_resources_target_directory,
                           output_distances_file_name))
            np.save(file=output_distances_file_path,  # np.save adds extension .npy to the file name
                    arr=distances.reshape(self.v0_grid.length, self.a0_grid.length, self.vT_grid.length))
