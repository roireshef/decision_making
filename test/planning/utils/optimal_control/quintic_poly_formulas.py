import numpy as np
import sympy as sp
from sympy import symbols
from sympy.matrices import *
from typing import List

from decision_making.paths import Paths
from decision_making.src.global_constants import LON_ACC_LIMITS, VELOCITY_LIMITS, \
    BP_ACTION_T_LIMITS, EPS
from decision_making.src.planning.behavioral.data_objects import ActionType
from decision_making.src.planning.types import LIMIT_MAX, LIMIT_MIN
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import UniformGrid
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


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

        J_sq = sp.integrate(j_t ** 2, (t, 0, T)).simplify()
        a_sq = sp.integrate(a_t ** 2, (t, 0, T)).simplify()

        wJ, wT = symbols('w_J w_T')

        cost = (wJ * J_sq + wT * T).simplify()

        cost = cost.subs(s0, 0).subs(aT, 0).simplify()
        cost_diff = sp.diff(cost, T).simplify()

        package_v_t = v_t.subs(s0, 0).subs(aT, 0).simplify()
        package_delta_s_t = (sT + vT * t - x_t.subs(s0, 0).subs(aT, 0)).simplify()
        package_distance_from_target_deriv = sp.diff(package_delta_s_t, t).simplify()
        package_a_t = sp.diff(package_v_t, t).simplify()
        package_j_t = sp.diff(package_a_t, t).simplify()

        cost_desmos = cost.subs(a0, 0).simplify()
        cost_diff_desmos = cost_diff.subs(a0, 0).simplify()
        delta_s_t_desmos = package_delta_s_t.subs(a0, 0).simplify()
        v_t_desmos = package_v_t.subs(a0, 0).simplify()
        a_t_desmos = package_a_t.subs(a0, 0).simplify()
        j_t_desmos = package_j_t.subs(a0, 0).simplify()

        return package_v_t, package_delta_s_t, package_distance_from_target_deriv, package_a_t, package_j_t,\
               cost_desmos, cost_diff_desmos, delta_s_t_desmos, v_t_desmos, a_t_desmos, j_t_desmos


class QuinticMotionPredicatesCreator:
    """This class creates predicates for filtering trajectories before specification according to initial velocity and
     acceleration, distance from target vehicle, and final velocity"""

    def __init__(self, v0_grid: UniformGrid, a0_grid: UniformGrid, sT_grid: UniformGrid, vT_grid: UniformGrid,
                 T_m: float, T_safety: float, predicates_resources_target_directory: str):
        """
        :param v0_grid: A grid of initial velocities by which the predicates will be created (typically constant)
        :param a0_grid: A grid of initial accelerations by which the predicates will be created (typically constant)
        :param sT_grid: A grid of initial distances from target by which the predicates will be created (typically constant)
        :param vT_grid: A grid of final velocities by which the predicates will be created (typically constant)
        :param T_m: Specification margin time interval for following/overtaking actions (typically constant)
        :param T_safety: Safety margin time interval for following/overtaking actions (typically constant)
        :param predicates_resources_target_directory: A target directory inside resources directory where the predicates
                will be created(typically constant)
        """
        self.v0_grid = v0_grid
        self.a0_grid = a0_grid
        self.sT_grid = sT_grid
        self.vT_grid = vT_grid
        self.T_m = T_m
        self.T_safety = T_safety

        self.predicates_resources_target_directory = predicates_resources_target_directory  # 'predicates'
        self.predicate = np.full(shape=[len(v0_grid), len(a0_grid), len(sT_grid), len(vT_grid)],
                                 fill_value=False)

    @staticmethod
    def create_quintic_motion_funcs(a_0, v_0, v_T, s_T, T, T_m):
        """
        :param a_0: initial acceleration [m/s^2]
        :param v_0: initial velocity [m/s^2]
        :param v_T: desired velocity [m/s^2]
        :param s_T: initial distance from target [m]
        :param T: action_time [s]
        :param T_m: Specification margin time interval behind or ahead of target vehicle [s]
        :return: lambda functions of distance_from_target, the coefficients of the derivative polynomial of distance
         from target, velocity and acceleration w.r.t time (valid in the range [0,T])
        """
        return QuinticPoly1D.distance_from_target(a_0, v_0, v_T, s_T, T, T_m), \
               QuinticPoly1D.distance_from_target_derivative_coefs(a_0, v_0, v_T, s_T, T, T_m), \
               QuinticPoly1D.velocity_profile_function(a_0, v_0, v_T, s_T, T, T_m), \
               QuinticPoly1D.acceleration_profile_function(a_0, v_0, v_T, s_T, T, T_m)

    @staticmethod
    def generate_predicate_value(action_type, w_T, w_J, a_0, v_0, v_T, s_T, T_m, T_safety):
        """
        Generates the actual predicate value (true/false) for the given action,weights and scenario params
        :param action_type:
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param a_0: initial acceleration [m/s^2]
        :param v_0: initial velocity [m/s]
        :param v_T: desired final velocity [m/s]
        :param s_T: initial distance from target car (+/- constant safety margin) [m]
        :param T_m: specification margin from target vehicle [s]
        :param T_safety: safety margin from target vehicle [s]
        :return: True if given parameters will generate a feasible trajectory that meets time, velocity and
                acceleration constraints and doesn't get into target vehicle safety zone.
        """

        # Agent is in tracking mode, meaning the required velocity change is negligible and action time is actually
        # zero. This degenerate action is valid but can't be solved analytically.
        # Here we can't find a local minima as the equation is close to a linear line, intersecting in T=0.
        if QuinticPoly1D.is_tracking_mode(v_0, np.array([v_T]), a_0, np.array([s_T]), T_m)[0]:
            return True

        time_cost_poly_coefs = \
            QuinticPoly1D.time_cost_function_derivative_coefs(np.array([w_T]), np.array([w_J]),
                                                              np.array([a_0]), np.array([v_0]),
                                                              np.array([v_T]), np.array([s_T]),
                                                              np.array([T_m]))[0]
        cost_roots_reals = Math.find_real_roots_in_limits(time_cost_poly_coefs, BP_ACTION_T_LIMITS)
        extremum_T = cost_roots_reals[np.isfinite(cost_roots_reals)]

        if len(extremum_T) == 0:
            return False

        T = extremum_T.min()  # First extrema is our local (and sometimes global) minimum

        # Handling the case of an action where we'd like to continue doing what we're doing, so action time is zero
        # or very small and gets quantized to zero.
        if T == 0:
            return True

        delta_s_t_func, coefs_s_der, v_t_func, a_t_func = QuinticMotionPredicatesCreator.create_quintic_motion_funcs(
            a_0, v_0,
            v_T, s_T,
            T,
            T_m=T_m)

        dt = 0.001  # small margin

        velocity_profile_derivative_coefs = QuinticPoly1D.velocity_profile_derivative_coefs(a_0, v_0, v_T, s_T, T, T_m)
        velocity_roots_reals = Math.find_real_roots_in_limits(velocity_profile_derivative_coefs, np.array(
            [dt, T-dt]))
        extremum_vel = v_t_func(velocity_roots_reals[np.isfinite(velocity_roots_reals)])

        acceleration_profile_derivative_coefs = QuinticPoly1D.acceleration_profile_derivative_coefs(a_0, v_0, v_T, s_T, T, T_m)
        acceleration_roots_reals = Math.find_real_roots_in_limits(acceleration_profile_derivative_coefs, np.array(
            [dt, T - dt]))
        extremum_acc = a_t_func(acceleration_roots_reals[np.isfinite(acceleration_roots_reals)])

        s_roots_reals = Math.find_real_roots_in_limits(coefs_s_der, np.array(
            [dt, T - dt]))
        extremum_delta_s_val = delta_s_t_func(s_roots_reals[np.isfinite(s_roots_reals)])

        min_v = min(v_0, v_T, min(extremum_vel) if len(extremum_vel) > 0 else VELOCITY_LIMITS[LIMIT_MAX])
        max_v = max(v_0, v_T, max(extremum_vel) if len(extremum_vel) > 0 else VELOCITY_LIMITS[LIMIT_MIN])
        min_a = min(a_0, 0, min(extremum_acc) if len(extremum_acc) > 0 else LON_ACC_LIMITS[LIMIT_MAX])
        max_a = max(a_0, 0, max(extremum_acc) if len(extremum_acc) > 0 else LON_ACC_LIMITS[LIMIT_MIN])

        is_T_in_range = (T <= BP_ACTION_T_LIMITS[LIMIT_MAX] + EPS)
        is_vel_in_range = (min_v >= VELOCITY_LIMITS[LIMIT_MIN]) and (max_v <= VELOCITY_LIMITS[LIMIT_MAX])
        is_acc_in_range = (min_a >= LON_ACC_LIMITS[LIMIT_MIN]) and (max_a <= LON_ACC_LIMITS[LIMIT_MAX])

        if action_type == ActionType.FOLLOW_VEHICLE:
            is_dist_safe = np.all(extremum_delta_s_val >= T_safety * v_T)
        elif action_type == ActionType.OVERTAKE_VEHICLE:
            is_dist_safe = np.all(extremum_delta_s_val <= T_safety * v_T)
        else:
            is_dist_safe = True

        return is_T_in_range and is_vel_in_range and is_acc_in_range and is_dist_safe

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
            T_safety = self.T_safety * margin_sign
            sT_grid = margin_sign * self.sT_grid.array
            for weight in jerk_time_weights:
                w_J, w_T = weight[0], weight[2]
                print('weights are: %.2f,%.2f' % (w_J, w_T))
                for k, v_0 in enumerate(self.v0_grid):
                    print('v_0 is: %.1f' % v_0)
                    for m, a_0 in enumerate(self.a0_grid):
                        for i, s_T in enumerate(sT_grid):
                            for j, v_T in enumerate(self.vT_grid):
                                self.predicate[k, m, i, j] = \
                                    QuinticMotionPredicatesCreator.generate_predicate_value(
                                        action_type, w_T, w_J, a_0, v_0, v_T, s_T, T_m, T_safety)

                output_predicate_file_name = '%s_predicate_wT_%.2f_wJ_%.2f.bin' % (action_type.name.lower(), w_T, w_J)
                output_predicate_file_path = Paths.get_resource_absolute_path_filename(
                    '%s/%s' % (self.predicates_resources_target_directory,
                               output_predicate_file_name))
                BinaryReadWrite.save(array=self.predicate, file_path=output_predicate_file_path)
