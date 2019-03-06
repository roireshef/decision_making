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
    def generate_predicate_value(w_T, w_J, a_0, v_0, v_T) -> [bool, float]:
        """
        Generates the actual predicate value (true/false) for the given action,weights and scenario params
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param a_0: initial acceleration [m/s^2]
        :param v_0: initial velocity [m/s]
        :param v_T: desired final velocity [m/s]
        :return: True if given parameters will generate a feasible trajectory that meets time, velocity and
                acceleration constraints and doesn't get into target vehicle safety zone.
        """
        T = QuarticMotionPredicatesCreator.calc_T_s(w_T, w_J, np.array([v_0]), np.array([a_0]), np.array([v_T]))
        if np.isnan(T):
            return False, np.inf

        in_limits = QuarticMotionPredicatesCreator.check_validity(a_0, v_0, v_T, T)
        distance = QuarticPoly1D.distance_profile_function(a_0, v_0, v_T, T)(T) \
            if T > 0 and in_limits else 0 if T == 0 else np.inf
        return in_limits, distance

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
        w_T_array = np.full(v_T.shape, w_T)
        w_J_array = np.full(v_T.shape, w_J)

        # Get polynomial coefficients of time-jerk cost function derivative for our settings
        time_cost_derivative_poly_coefs = \
            QuarticPoly1D.time_cost_function_derivative_coefs(w_T_array, w_J_array, a_0, v_0, v_T)

        # Find roots of the polynomial in order to get extremum points
        cost_real_roots = Math.find_real_roots_in_limits(time_cost_derivative_poly_coefs, np.array([0, np.inf]))

        # return T as the minimal real root
        return np.fmin.reduce(cost_real_roots, axis=-1)

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

    @staticmethod
    def check_validity(a_0, v_0, v_T, T):

        # Handling the case of an action where we'd like to continue doing exactly what we're doing,
        # so action time might be zero or very small and gets quantized to zero.
        if T == 0:
            return True

        # if the horizon T is too long for an action:
        if T > BP_ACTION_T_LIMITS[1] + EPS:
            return False

        return QuarticMotionPredicatesCreator.check_action_limits(T, v_0, v_T, a_0)

    def create_predicates(self, jerk_time_weights: np.ndarray) -> None:
        """
        Creates predicates for the jerk-time weights and the follow_lane static action
        :param jerk_time_weights: a 2-dimensional of shape [Kx3] where its rows are different sets of weights and each
                set of weights is built from 3 terms :  longitudinal jerk, latitudinal jerk and action time weights.
        :return:
        """
        action_type = ActionType.FOLLOW_LANE
        predicate = np.full(shape=[len(self.v0_grid), len(self.a0_grid), len(self.vT_grid)], fill_value=False)
        distances = np.full(shape=[len(self.v0_grid), len(self.a0_grid), len(self.vT_grid)], dtype=float, fill_value=np.inf)

        for wi, weight in enumerate(jerk_time_weights):
            w_J, w_T = weight[0], weight[2]  # w_T stays the same (0.1), w_J is now to be one of [12,2,0.01]
            print('weights are: %.4f,%.4f' % (w_J, w_T))
            for k, v_0 in enumerate(self.v0_grid):
                print('v_0 is: %.2f' % v_0)
                for m, a_0 in enumerate(self.a0_grid):
                    for j, v_T in enumerate(self.vT_grid):
                        predicate[k, m, j], distances[k, m, j] = \
                            QuarticMotionPredicatesCreator.generate_predicate_value(w_T, w_J, a_0, v_0, v_T)

            # save 'limits' predicate to file
            output_predicate_file_name = '%s_limits_wT_%.4f_wJ_%.4f.bin' % (action_type.name.lower(), w_T, w_J)
            output_predicate_file_path = Paths.get_resource_absolute_path_filename(
                '%s/%s' % (self.predicates_resources_target_directory,
                           output_predicate_file_name))
            BinaryReadWrite.save(array=predicate, file_path=output_predicate_file_path)

            # save actions distances to file
            output_distances_file_name = '%s_distances_wT_%.4f_wJ_%.4f.bin' % (action_type.name.lower(), w_T, w_J)
            output_distances_file_path = Paths.get_resource_absolute_path_filename(
                '%s/%s' % (self.predicates_resources_target_directory,
                           output_distances_file_name))
            np.save(file=output_distances_file_path, arr=distances)  # np.save adds extension .npy to the file name
