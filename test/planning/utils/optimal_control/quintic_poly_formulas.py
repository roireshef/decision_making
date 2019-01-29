from decision_making.src.planning.utils.safety_utils import SafetyUtils
from typing import List

import numpy as np
import sympy as sp
from sympy import symbols
from sympy.matrices import *

from decision_making.paths import Paths
from decision_making.src.global_constants import LON_ACC_LIMITS, VELOCITY_LIMITS, \
    BP_ACTION_T_LIMITS, EPS, SPECIFICATION_MARGIN_TIME_DELAY, TRAJECTORY_TIME_RESOLUTION, SAFETY_MARGIN_TIME_DELAY
from decision_making.src.planning.behavioral.data_objects import ActionType
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import UniformGrid
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, Poly1D


class QuinticMotionSymbolicsCreator:
    @staticmethod
    def create_symbolic_quintic_motion_equations():
        """
        This function uses symbolic package SymPy and computes the motion equations and cost function used for analysis of
        trajectories (in e.g. desmos) , action specification and filtering.
        """
        T = symbols('T')
        t = symbols('t')
        Tm = symbols('T_m')  # safety margin in seconds

        s0, v0, a0, sT, vT, aT = symbols('s_0 v_0 a_0 s_T v_T a_T')

        A = Matrix([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
            [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
            [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0]]
        )

        # solve to get solution
        # this assumes a0=aT==0 (constant velocity)
        [c5, c4, c3, c2, c1, c0] = A.inv() * Matrix([s0, v0, a0, sT + vT * (T - Tm), vT, aT])

        x_t = (c5 * t ** 5 + c4 * t ** 4 + c3 * t ** 3 + c2 * t ** 2 + c1 * t + c0).simplify()
        v_t = sp.diff(x_t, t).simplify()
        a_t = sp.diff(v_t, t).simplify()
        j_t = sp.diff(a_t, t).simplify()

        J = sp.integrate(j_t ** 2, (t, 0, T)).simplify()

        wJ, wT = symbols('w_J w_T')

        cost = (wJ * J + wT * T).simplify()

        package_cost = cost.subs(s0, 0).subs(aT, 0).simplify()
        package_cost_diff = sp.diff(package_cost, T).simplify()
        package_v_t = v_t.subs(s0, 0).subs(aT, 0).simplify()
        package_delta_s_t = (sT + vT * t - x_t.subs(s0, 0).subs(aT, 0)).simplify()
        package_distance_from_target_deriv = sp.diff(package_delta_s_t, t).simplify()
        package_a_t = sp.diff(package_v_t, t).simplify()
        package_j_t = sp.diff(package_a_t, t).simplify()

        desmos_cost = package_cost.subs(a0, 0).simplify()
        desmos_cost_diff = package_cost_diff.subs(a0, 0).simplify()
        desmos_delta_s_t = package_delta_s_t.subs(a0, 0).simplify()
        desmos_v_t = package_v_t.subs(a0, 0).simplify()
        desmos_a_t = package_a_t.subs(a0, 0).simplify()
        desmos_j_t = package_j_t.subs(a0, 0).simplify()

        return package_v_t, package_delta_s_t, package_distance_from_target_deriv, package_a_t, package_j_t, \
               desmos_cost, desmos_cost_diff, desmos_delta_s_t, desmos_v_t, desmos_a_t, desmos_j_t


class QuinticMotionPredicatesCreator:
    """This class creates predicates for filtering trajectories before specification according to initial velocity and
     acceleration, distance from target vehicle, and final velocity"""

    def __init__(self, v0_grid: UniformGrid, a0_grid: UniformGrid, sT_grid: UniformGrid, vT_grid: UniformGrid,
                 T_m: float, predicates_resources_target_directory: str):
        """
        :param v0_grid: A grid of initial velocities by which the predicates will be created (typically constant)
        :param a0_grid: A grid of initial accelerations by which the predicates will be created (typically constant)
        :param sT_grid: A grid of initial distances from target by which the predicates will be created (typically constant)
        :param vT_grid: A grid of final velocities by which the predicates will be created (typically constant)
        :param T_m: Specification margin time interval for following/overtaking actions (typically constant)
        :param predicates_resources_target_directory: A target directory inside resources directory where the predicates
                will be created(typically constant)
        """
        self.v0_grid = v0_grid
        self.a0_grid = a0_grid
        self.sT_grid = sT_grid
        self.vT_grid = vT_grid
        self.T_m = T_m

        self.predicates_resources_target_directory = predicates_resources_target_directory  # 'predicates'
        self.limits = np.full(shape=[len(v0_grid), len(a0_grid), len(sT_grid), len(vT_grid)], fill_value=False)
        self.safety = np.full(shape=[len(v0_grid), len(a0_grid), len(sT_grid), len(vT_grid)], fill_value=False)

    @staticmethod
    def generate_predicate_value(w_T: float, w_J: float, v_0: np.array, a_0: np.array, v_T: np.array, s_T: np.array, T_m: float):
        """
        Generates the actual predicate value (true/false) for the given action,weights and scenario params
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param v_0: array of initial velocities [m/s]
        :param a_0: array of initial accelerations [m/s^2]
        :param v_T: array of final velocities [m/s]
        :param s_T: array of initial distances from target car (+/- constant safety margin) [m]
        :param T_m: specification margin from target vehicle [s]
        :return: True if given parameters will generate a feasible trajectory that meets time, velocity and
                acceleration constraints and doesn't get into target vehicle safety zone.
        """
        # calculate T for all actions
        T = QuinticMotionPredicatesCreator.calc_T_s(w_T, w_J, v_0, a_0, v_T, s_T, T_m)

        # get indices of non-nan positive T values; for nan values of T, is_in_limits = False
        valid_idxs = np.where(np.logical_and(T > 0, T <= BP_ACTION_T_LIMITS[1]))[0]
        if len(valid_idxs) == 0:
            # for T == 0 the action is "do nothing" (ego is at the goal), then such action is valid
            return T == 0, T == 0  # T is a vector of times

        is_in_limits = np.full(T.shape, False)
        is_safe = np.full(T.shape, False)

        # check actions validity: velocity & acceleration limits and longitudinal safety
        is_in_limits[valid_idxs], is_safe[valid_idxs], _ = QuinticMotionPredicatesCreator.check_action_validity(
            T[valid_idxs], v_0[valid_idxs], v_T[valid_idxs], s_T[valid_idxs], a_0[valid_idxs], T_m)

        # for T == 0 the actions are valid
        is_in_limits[T == 0] = True
        is_safe[T == 0] = True

        return is_in_limits, is_safe

    @staticmethod
    def calc_T_s(w_T: float, w_J: float, v_0: np.array, a_0: np.array, v_T: np.array, s_T: np.array,
                 T_m: float=SPECIFICATION_MARGIN_TIME_DELAY):
        """
        given initial & end constraints and time-jerk weights, calculate longitudinal planning time
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param v_0: array of initial velocities [m/s]
        :param a_0: array of initial accelerations [m/s^2]
        :param v_T: array of final velocities [m/s]
        :param s_T: array of initial distances from target car (+/- constant safety margin) [m]
        :param T_m: specification margin from target vehicle [s]
        :return: array of longitudinal trajectories' lengths (in seconds) for all sets of constraints
        """
        w_T_array = np.full(s_T.shape, w_T)
        w_J_array = np.full(s_T.shape, w_J)

        # Get polynomial coefficients of time-jerk cost function derivative for our settings
        time_cost_derivative_poly_coefs = \
            QuinticPoly1D.time_cost_function_derivative_coefs(w_T_array, w_J_array, a_0, v_0, v_T, s_T, T_m)

        # Find roots of the polynomial in order to get extremum points
        cost_real_roots = Math.find_real_roots_in_limits(time_cost_derivative_poly_coefs, np.array([0, np.inf]))

        # calculate T for all actions
        return np.fmax.reduce(cost_real_roots, axis=-1)

    @staticmethod
    def check_action_validity(T: np.array, v_0: np.array, v_T: np.array, s_T: np.array, a_0: np.array,
                              T_m: float = SPECIFICATION_MARGIN_TIME_DELAY) -> [np.array, np.array, np.array]:
        """
        Given longitudinal action dynamics, calculate validity wrt velocity & acceleration limits, and safety of each action.
        :param T: array of action times
        :param v_0: array of initial velocities
        :param v_T: array of final velocities
        :param a_0: array of initial acceleration
        :param s_T: array of initial distances from target
        :param T_m: time delay behind the car
        :return: (1) boolean array: are velocity & acceleration in limits,
                 (2) boolean array: is the baseline trajectory safe
                 (3) coefficients of s polynomial
        """
        poly_coefs = QuinticPoly1D.s_profile_coefficients(a_0, v_0, v_T, s_T, T, T_m)
        # check acc & vel limits
        poly_coefs[np.where(poly_coefs[:, 0] == 0), 0] = EPS  # keep the polynomials to be quintic
        acc_in_limits = QuinticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)
        vel_in_limits = QuinticPoly1D.are_velocities_in_limits(poly_coefs, T, VELOCITY_LIMITS)
        is_in_limits = np.logical_and(acc_in_limits, vel_in_limits)
        # check safety
        is_safe = QuinticMotionPredicatesCreator.get_lon_safety_for_action_specs(poly_coefs, T, v_T, s_T)
        return is_in_limits, is_safe, poly_coefs

    @staticmethod
    def get_lon_safety_for_action_specs(poly_coefs: np.array, T: np.array, v_T: np.array, s_T: np.array) -> np.array:
        """
        Given polynomial coefficients and action specs for each polynomial, calculate longitudinal safety
        w.r.t. the front cars with constant velocities, described by the specs.
        :param poly_coefs: 2D matrix Nx6: N quintic polynomials of ego
        :param T: array of action times
        :param v_T: array of final velocities
        :param s_T: array of initial distances from target
        :return: boolean array of size N: longitudinal safety for each spec
        """
        time_samples = np.arange(0, np.max(T) + EPS, TRAJECTORY_TIME_RESOLUTION)

        # sample polynomials and create ftrajectories_s
        trajectories_s = Poly1D.polyval_with_derivatives(poly_coefs, time_samples)

        # test longitudinal safety only for the second half of each trajectory, because in the first half ego
        # may be safe laterally w.r.t. the followed/overtaken object
        from_idx = [int(0.5 * T[i] / TRAJECTORY_TIME_RESOLUTION) for i, trajectory in enumerate(trajectories_s)]
        ego_trajectories = [trajectory[from_idx[i]:int(T[i] / TRAJECTORY_TIME_RESOLUTION) + 1]
                            for i, trajectory in enumerate(trajectories_s)]

        obj_trajectories = [np.c_[s_T[i] + v_T[i] * from_idx[i] * TRAJECTORY_TIME_RESOLUTION +
                                  np.linspace(0, v_T[i] * (len(ego_trajectory)-1) * TRAJECTORY_TIME_RESOLUTION, len(ego_trajectory)),
                                  np.full(len(ego_trajectory), v_T[i]),
                                  np.zeros(len(ego_trajectory))]
                            for i, ego_trajectory in enumerate(ego_trajectories)]

        # concatenate all trajectories to a single long trajectory
        concat_ego_trajectory = np.concatenate(ego_trajectories)
        concat_obj_trajectory = np.concatenate(obj_trajectories)

        # calc longitudinal RSS for the long trajectory
        concat_safe_times = SafetyUtils._get_lon_safety(concat_ego_trajectory, SAFETY_MARGIN_TIME_DELAY,
                                                        concat_obj_trajectory, SPECIFICATION_MARGIN_TIME_DELAY, margin=0)

        # split the safety results according to the original trajectories
        trajectories_lengths = [len(trajectory) for trajectory in ego_trajectories]
        safe_times_matrix = np.split(concat_safe_times, np.cumsum(trajectories_lengths[:-1]))
        # AND on all time samples for each action
        safe_specs = [action_safe_times.all() for action_safe_times in safe_times_matrix]
        return np.array(safe_specs)

    def create_predicates(self, jerk_time_weights: np.ndarray, action_types: List[ActionType]) -> None:
        """
        Creates predicates for the jerk-time weights and dynamic action types given
        :param jerk_time_weights: a 2-dimensional of shape [Kx3] where its rows are different sets of weights and each
                set of weights is built from 3 terms :  longitudinal jerk, latitudinal jerk and action time weights.
        :param action_types: a list of all action types for which predicates will be created
        :return:
        """
        for action_type in action_types:
            margin_sign = +1 if action_type == ActionType.FOLLOW_VEHICLE else -1
            T_m = self.T_m * margin_sign
            sT_grid = margin_sign * self.sT_grid.array
            a0, sT, vT = np.meshgrid(self.a0_grid.array, sT_grid, self.vT_grid.array, indexing='ij')
            a0, sT, vT = np.ravel(a0), np.ravel(sT), np.ravel(vT)

            for weight in jerk_time_weights:
                w_J, w_T = weight[0], weight[2]
                print('weights are: %.2f,%.2f' % (w_J, w_T))

                for k, v_0 in enumerate(self.v0_grid):
                    print('v_0 is: %.1f' % v_0)
                    limits, safety = QuinticMotionPredicatesCreator.generate_predicate_value(
                        w_T, w_J, np.full(vT.shape, v_0), a0, vT, sT, T_m)
                    self.limits[k] = limits.reshape((len(self.a0_grid), len(self.sT_grid), len(self.vT_grid)))
                    self.safety[k] = safety.reshape((len(self.a0_grid), len(self.sT_grid), len(self.vT_grid)))

                output_limits_file_name = '%s_limits_wT_%.2f_wJ_%.2f.bin' % (action_type.name.lower(), w_T, w_J)
                output_predicate_file_path = Paths.get_resource_absolute_path_filename(
                    '%s/%s' % (self.predicates_resources_target_directory, output_limits_file_name))
                BinaryReadWrite.save(array=self.limits, file_path=output_predicate_file_path)

                output_safety_file_name = '%s_safety_wT_%.2f_wJ_%.2f.bin' % (action_type.name.lower(), w_T, w_J)
                output_safety_file_path = Paths.get_resource_absolute_path_filename(
                    '%s/%s' % (self.predicates_resources_target_directory, output_safety_file_name))
                BinaryReadWrite.save(array=self.safety, file_path=output_safety_file_path)
