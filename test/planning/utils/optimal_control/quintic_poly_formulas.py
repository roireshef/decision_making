import sympy as sp
import numpy as np
from sympy import symbols
from sympy.matrices import *
import time

from decision_making.paths import Paths
from decision_making.src.global_constants import LON_ACC_LIMITS, BP_JERK_S_JERK_D_TIME_WEIGHTS, VELOCITY_LIMITS, \
    BP_ACTION_T_LIMITS, EPS
from decision_making.src.planning.behavioral.data_objects import ActionType
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


def create_quintic_motion_funcs(a_0, v_0, v_T, s_T, T, T_m):

    return QuinticPoly1D.distance_profile_function(a_0, v_0, v_T, s_T, T, T_m), \
           QuinticPoly1D.distance_from_target_derivative_coefs(a_0, v_0, v_T, s_T, T, T_m), \
           QuinticPoly1D.velocity_profile_function(a_0, v_0, v_T, s_T, T, T_m),\
           QuinticPoly1D.acceleration_profile_function(a_0, v_0, v_T, s_T, T, T_m)


action_type = ActionType.FOLLOW_VEHICLE
T_m = 2
st_limits = [0, 110]

# v_0_grid = np.arange(VELOCITY_LIMITS[0], VELOCITY_LIMITS[1] + EPS, 1)
# a_0_grid = np.arange(-2, 2 + EPS, 0.5)
# s_T_grid = np.arange(st_limits[0], st_limits[1] + EPS, 0.5)
# v_T_grid = np.arange(VELOCITY_LIMITS[0], VELOCITY_LIMITS[1] + EPS, 0.5)

a_0_grid = np.arange(LON_ACC_LIMITS[0], LON_ACC_LIMITS[1]+EPS, 0.5)
v_0_grid = np.arange(VELOCITY_LIMITS[0], VELOCITY_LIMITS[1]+EPS, 0.5)
s_T_grid = np.arange(st_limits[0], st_limits[1]+EPS, 1)
v_T_grid = np.arange(VELOCITY_LIMITS[0], VELOCITY_LIMITS[1]+EPS, 0.5)
predicate = np.full(shape=[v_0_grid.shape[0], a_0_grid.shape[0], s_T_grid.shape[0], v_T_grid.shape[0]], fill_value=False)

if __name__ == "__main__":

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

    cost = cost.subs(s0, 0).subs(aT, 0).simplify()
    cost_diff = sp.diff(cost, T).simplify()

    temp_v_t = v_t.subs(s0, 0).subs(aT, 0).simplify()
    temp_delta_s_t = (sT + vT * t - x_t.subs(s0, 0).subs(aT, 0)).simplify()
    temp_distance_from_target_deriv = sp.diff(temp_delta_s_t, t).simplify()
    temp_a_t = sp.diff(temp_v_t, t).simplify()

    cost_desmos = cost.subs(a0, 0).simplify()
    cost_diff_desmos = cost_diff.subs(a0, 0).simplify()
    delta_s_t_desmos = temp_delta_s_t.subs(a0, 0).simplify()
    v_t_desmos = temp_v_t.subs(a0, 0).simplify()
    a_t_desmos = temp_a_t.subs(a0, 0).simplify()

    start_time = time.time()
    for action_type in [ActionType.FOLLOW_VEHICLE, ActionType.OVER_TAKE_VEHICLE]:
        if action_type == ActionType.OVER_TAKE_VEHICLE:
            T_m *= -1
            s_T_grid = -1*s_T_grid
        for weight in BP_JERK_S_JERK_D_TIME_WEIGHTS:
            w_J, w_T = weight[0], weight[2]  # w_T stays the same (0.1), w_J is now to be one of [12,2,0.01]
            print('weights are: %.2f,%.2f' % (w_J, w_T))
            for k, v_0 in enumerate(v_0_grid):
                print('v_0 is: %.1f' % v_0)
                for m, a_0 in enumerate(a_0_grid):
                    for i, s_T in enumerate(s_T_grid):
                        for j, v_T in enumerate(v_T_grid):
                            time_cost_poly_coefs = QuinticPoly1D.time_cost_function_derivative_coefs(np.array([w_T]), np.array([w_J]),
                                        np.array([a_0]), np.array([v_0]), np.array([v_T]), np.array([s_T]), np.array([T_m]))[0]
                            cost_roots_reals = Math.find_real_roots_in_limits(time_cost_poly_coefs, np.array([BP_ACTION_T_LIMITS[0], BP_ACTION_T_LIMITS[1]]))
                            extremum_T = cost_roots_reals[np.isfinite(cost_roots_reals)]
                            if len(extremum_T) == 0:
                                predicate[k, m, i, j] = False
                                continue
                            T = extremum_T.min()  # First extrema is our local (and sometimes global) minimum

                            delta_s_t_func, coefs_s_der, v_t_func, a_t_func = create_quintic_motion_funcs(a_0, v_0, v_T, s_T, T, T_m=T_m)

                            roots_s_reals = Math.find_real_roots_in_limits(coefs_s_der, np.array([BP_ACTION_T_LIMITS[0], BP_ACTION_T_LIMITS[1]]))

                            t = np.arange(0, T, 0.01)
                            extremum_delta_s_val = delta_s_t_func(roots_s_reals[np.isfinite(roots_s_reals)])
                            min_v, max_v = min(v_t_func(t)), max(v_t_func(t))
                            min_a, max_a = min(a_t_func(t)), max(a_t_func(t))

                            is_T_in_range = (T >= BP_ACTION_T_LIMITS[0]) and (T <= BP_ACTION_T_LIMITS[1]+EPS)
                            is_vel_in_range = (min_v >= VELOCITY_LIMITS[0]) and (max_v <= VELOCITY_LIMITS[1]+EPS)
                            is_acc_in_range = (min_a >= LON_ACC_LIMITS[0]) and (max_a <= LON_ACC_LIMITS[1]+EPS)
                            if action_type == ActionType.FOLLOW_VEHICLE:
                                is_dist_safe = np.all(extremum_delta_s_val >= T_m*v_T)
                            elif action_type == ActionType.OVER_TAKE_VEHICLE:
                                is_dist_safe = np.all(extremum_delta_s_val <= T_m*v_T)
                            else:
                                is_dist_safe = True

                            predicate[k, m, i, j] = (is_T_in_range and is_vel_in_range and is_acc_in_range and is_dist_safe)

            output_predicate_file_name = '%s_predicate_wT_%.2f_wJ_%.2f.bin' % (action_type.name.lower(), w_T, w_J)
            output_predicate_file_path = Paths.get_resource_absolute_path_filename('predicates/%s' % output_predicate_file_name)
            BinaryReadWrite.save(array=predicate, file_path=output_predicate_file_path)

    print('Entire process took: %f' % (time.time() - start_time))
