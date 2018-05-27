import numpy as np
import sympy as sp
from sympy import symbols
from sympy.matrices import *

from decision_making.paths import Paths
from decision_making.src.global_constants import LON_ACC_LIMITS, BP_JERK_S_JERK_D_TIME_WEIGHTS, VELOCITY_LIMITS, \
    BP_ACTION_T_LIMITS, EPS, FILTER_V_T_GRID, FILTER_A_0_GRID, FILTER_V_0_GRID
from decision_making.src.planning.behavioral.data_objects import ActionType
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.numpy_utils import UniformGrid
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D


def create_quartic_motion_funcs(a_0, v_0, v_T, T):
    return QuarticPoly1D.velocity_profile_function(a_0, v_0, v_T, T), \
           QuarticPoly1D.acceleration_profile_function(a_0, v_0, v_T, T)


def create_symbolic_quintic_motion_equations():
    """
    This function uses symbolic package SymPy and computes the motion equations and cost function used for analysis of
    trajectories (in e.g. desmos) , action specification and filtering.
    """
    T = symbols('T')
    t = symbols('t')
    Tm = symbols('T_m')  # safety margin in seconds

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

    cost = cost.subs(s0, 0).subs(aT, 0).simplify()
    cost_diff = sp.diff(cost, T).simplify()

    package_x_t = x_t.subs(s0, 0).subs(aT, 0).simplify()
    package_v_t = v_t.subs(s0, 0).subs(aT, 0).simplify()
    package_a_t = sp.diff(package_v_t, t).simplify()

    cost_desmos = cost.subs(a0, 0).simplify()
    cost_diff_desmos = cost_diff.subs(a0, 0).simplify()
    x_t_desmos = package_x_t.subs(a0, 0).simplify()
    v_t_desmos = package_v_t.subs(a0, 0).simplify()
    a_t_desmos = package_a_t.subs(a0, 0).simplify()

    return package_x_t, package_v_t, package_a_t, cost_desmos, cost_diff_desmos, x_t_desmos, v_t_desmos, a_t_desmos


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
        self.predicate = np.full(shape=[len(v0_grid), len(a0_grid), len(vT_grid)],
                                 fill_value=False)

    @staticmethod
    def generate_predicate_value(w_T, w_J, a_0, v_0, v_T):
        time_cost_poly_coefs = \
            QuarticPoly1D.time_cost_function_derivative_coefs(np.array([w_T]), np.array([w_J]),
                                                              np.array([a_0]), np.array([v_0]),
                                                              np.array([v_T]))[0]
        cost_roots_reals = Math.find_real_roots_in_limits(time_cost_poly_coefs,
                                                          np.array([EPS, BP_ACTION_T_LIMITS[1]]))
        extremum_T = cost_roots_reals[np.isfinite(cost_roots_reals)]

        if len(extremum_T) == 0:
            return False

        T = extremum_T.min()  # First extrema is our local (and sometimes global) minimum

        v_t_func, a_t_func = create_quartic_motion_funcs(a_0, v_0, v_T, T)

        time_res_for_extremum_query = 0.01
        t = np.arange(0, T, time_res_for_extremum_query)
        min_v, max_v = min(v_t_func(t)), max(v_t_func(t))
        min_a, max_a = min(a_t_func(t)), max(a_t_func(t))

        is_T_in_range = (T >= EPS) and (T <= BP_ACTION_T_LIMITS[1] + EPS)
        is_vel_in_range = (min_v >= VELOCITY_LIMITS[0] - EPS) and (max_v <= VELOCITY_LIMITS[1] + EPS)
        is_acc_in_range = (min_a >= LON_ACC_LIMITS[0] - EPS) and (max_a <= LON_ACC_LIMITS[1] + EPS)

        return is_T_in_range and is_vel_in_range and is_acc_in_range

    def create_predicates(self, jerk_time_weights: np.ndarray) -> None:
        """
        Creates predicates for the jerk-time weights and the follow_lane static action
        :param jerk_time_weights: a 2-dimensional of shape [Kx3] where its rows are different sets of weights and each
                set of weights is built from 3 terms :  longitudinal jerk, latitudinal jerk and action time weights.
        :return:
        """
        action_type = ActionType.FOLLOW_LANE
        for weight in jerk_time_weights:
            w_J, w_T = weight[0], weight[2]  # w_T stays the same (0.1), w_J is now to be one of [12,2,0.01]
            print('weights are: %.2f,%.2f' % (w_J, w_T))
            for k, v_0 in enumerate(self.v0_grid):
                print('v_0 is: %.2f' % v_0)
                for m, a_0 in enumerate(self.a0_grid):
                    for j, v_T in enumerate(self.vT_grid):
                        self.predicate[k, m, j] = QuarticMotionPredicatesCreator.generate_predicate_value(w_T, w_J, a_0, v_0, v_T)

            output_predicate_file_name = '%s_predicate_wT_%.2f_wJ_%.2f.bin' % (action_type.name.lower(), w_T, w_J)
            output_predicate_file_path = Paths.get_resource_absolute_path_filename(
                '%s/%s' % (self.predicates_resources_target_directory,
                           output_predicate_file_name))
            BinaryReadWrite.save(array=self.predicate, file_path=output_predicate_file_path)
