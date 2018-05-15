import sympy as sp
import numpy as np
from sympy import symbols
from sympy.matrices import *
import time

from decision_making.paths import Paths
from decision_making.src.global_constants import LON_ACC_LIMITS, BP_JERK_S_JERK_D_TIME_WEIGHTS, VELOCITY_LIMITS, \
    BP_ACTION_T_LIMITS, EPS
from decision_making.src.planning.utils.file_utils import BinaryReadWrite


def create_action_time_cost_func_deriv(w_T, w_J, a_0, v_0, v_T, s_T, T_m):
    def action_time_cost_func_deriv(T):
        return (T**6*w_T + 3*w_J*(3*T**4*a_0**2 + 24*T**3*a_0*v_0 - 24*T**3*a_0*v_T + 40*T**2*T_m*a_0*v_T -
                40*T**2*a_0*s_T + 64*T**2*v_0**2 - 128*T**2*v_0*v_T + 64*T**2*v_T**2 + 240*T*T_m*v_0*v_T -
                240*T*T_m*v_T**2 - 240*T*s_T*v_0 + 240*T*s_T*v_T + 240*T_m**2*v_T**2 - 480*T_m*s_T*v_T + 240*s_T**2))/T**5

    return action_time_cost_func_deriv


def create_motion_funcs(a_0, v_0, v_T, s_T, T):

    def delta_s_t_func(t):
        return (-T**5*t*(a_0*t + 2*v_0) + 2*T**5*(s_T + t*v_T) + T**2*t**3*(3*T**2*a_0 + 4*T*(3*v_0 + 2*v_T) -
                20*s_T - 20*v_T*(T - 2)) - T*t**4*(3*T**2*a_0 + 2*T*(8*v_0 + 7*v_T) - 30*s_T - 30*v_T*(T - 2)) +
                t**5*(T**2*a_0 + 6*T*(v_0 + v_T) - 12*s_T - 12*v_T*(T - 2)))/(2*T**5)

    def v_t_func(t):
        return (2*T**5*(a_0*t + v_0) + 3*T**2*t**2*(-3*T**2*a_0 - 4*T*(3*v_0 + 2*v_T) + 20*s_T + 20*v_T*(T - 2)) +
                4*T*t**3*(3*T**2*a_0 + 2*T*(8*v_0 + 7*v_T) - 30*s_T - 30*v_T*(T - 2)) +
                5*t**4*(-T**2*a_0 - 6*T*(v_0 + v_T) + 12*s_T + 12*v_T*(T - 2)))/(2*T**5)

    def a_t_func(t):
        return (T**5*a_0 - 3*T**2*t*(3*T**2*a_0 + 4*T*(3*v_0 + 2*v_T) - 20*s_T - 20*v_T*(T - 2)) +
                6*T*t**2*(3*T**2*a_0 + 2*T*(8*v_0 + 7*v_T) - 30*s_T - 30*v_T*(T - 2)) +
                10*t**3*(-T**2*a_0 - 6*T*(v_0 + v_T) + 12*s_T + 12*v_T*(T - 2)))/T**5

    return delta_s_t_func, v_t_func, a_t_func


st_limits = [0, 100]

v_0_grid = np.arange(VELOCITY_LIMITS[0], VELOCITY_LIMITS[1] + EPS, 1)
a_0_grid = np.arange(-2, 2 + EPS, 0.5)
s_T_grid = np.arange(st_limits[0], st_limits[1] + EPS, 0.5)
v_T_grid = np.arange(VELOCITY_LIMITS[0], VELOCITY_LIMITS[1] + EPS, 0.5)

# a_0_grid = np.arange(LON_ACC_LIMITS[0], LON_ACC_LIMITS[1]+EPS, 0.5)
# v_0_grid = np.arange(VELOCITY_LIMITS[0], VELOCITY_LIMITS[1]+EPS, 0.5)
# s_T_grid = np.arange(st_limits[0], st_limits[1]+EPS, 1)
# v_T_grid = np.arange(VELOCITY_LIMITS[0], VELOCITY_LIMITS[1]+EPS, 0.5)
predicate = np.full(shape=[v_0_grid.shape[0], a_0_grid.shape[0], s_T_grid.shape[0], v_T_grid.shape[0]], fill_value=False)

if __name__ == "__main__":

    T = symbols('T')
    t = symbols('t')
    Tm = symbols('T_m')  # safety margin in seconds

    s0, v0, a0, ds_0, vT, aT = symbols('s_0 v_0 a_0 ds_0 v_T a_T')
    margin = symbols('margin')

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
    [c5, c4, c3, c2, c1, c0] = A.inv() * Matrix([s0, v0, a0, ds_0 + vT * (T - Tm) - margin, vT, aT])

    x_t = (c5 * t ** 5 + c4 * t ** 4 + c3 * t ** 3 + c2 * t ** 2 + c1 * t + c0).simplify()
    v_t = sp.diff(x_t, t).simplify()
    a_t = sp.diff(v_t, t).simplify()
    j_t = sp.diff(a_t, t).simplify()

    J = sp.integrate(j_t ** 2, (t, 0, T)).simplify()

    wJ, wT = symbols('w_J w_T')

    cost = (wJ * J + wT * T).simplify()

    cost = cost.subs(s0, 0).subs(aT, 0).simplify()
    cost_diff = sp.diff(cost, T).simplify()

    temp_v_t = v_t.subs(s0, 0).subs(aT, 0).subs(Tm, 2).simplify()
    temp_delta_s_t = (ds_0 + vT * t - x_t.subs(s0, 0).subs(aT, 0)).simplify()
    temp_a_t = sp.diff(temp_v_t, t).simplify()

    cost_desmos = cost.subs(a0, 0).simplify()
    cost_diff_desmos = cost_diff.subs(a0, 0).simplify()
    delta_s_t_desmos = temp_delta_s_t.subs(a0, 0).simplify()
    v_t_desmos = temp_v_t.subs(a0, 0).simplify()
    a_t_desmos = temp_a_t.subs(a0, 0).simplify()

    optimum_horizon_search_margin = 0.2

    for weight in BP_JERK_S_JERK_D_TIME_WEIGHTS:
        w_J, w_T = weight[0], weight[2]  # w_T stays the same (0.1), w_J is now to be one of [12,2,0.01]
        print('weights are: %.2f,%.2f' % (w_J, w_T))
        for k, v_0 in enumerate(v_0_grid):
            print('v_0 is: %.2f' % v_0)
            for m, a_0 in enumerate(a_0_grid):
                for i, s_T in enumerate(s_T_grid):
                    for j, v_T in enumerate(v_T_grid):
                        # start_time = time.time()
                        lambda_func = create_action_time_cost_func_deriv(w_T, w_J, a_0, v_0, v_T, s_T, T_m=2)
                        t = np.arange(BP_ACTION_T_LIMITS[0]-optimum_horizon_search_margin,
                                      BP_ACTION_T_LIMITS[1]+optimum_horizon_search_margin, 0.001)
                        der = lambda_func(t)
                        ind = np.argwhere(np.abs(der) < 0.01)
                        if len(ind) == 0:
                            predicate[k, m, i, j] = False
                            continue
                        T = t[ind[0]]  # First extrema is our local (and sometimes global) minimum

                        delta_s_t_func, v_t_func, a_t_func = create_motion_funcs(a_0, v_0, v_T, s_T, T)
                        t = np.arange(0, T, 0.01)
                        min_delta_s = min(delta_s_t_func(t))
                        min_v, max_v = min(v_t_func(t)), max(v_t_func(t))
                        min_a, max_a = min(a_t_func(t)), max(a_t_func(t))

                        is_T_in_range = (T > BP_ACTION_T_LIMITS[0]) and (T < BP_ACTION_T_LIMITS[1])
                        is_vel_in_range = (min_v >= VELOCITY_LIMITS[0]) and (max_v <= VELOCITY_LIMITS[1])
                        is_acc_in_range = (min_a >= LON_ACC_LIMITS[0]) and (max_a <= LON_ACC_LIMITS[1])
                        is_dist_safe = min_delta_s >= 2*v_T

                        predicate[k, m, i, j] = (is_T_in_range and is_vel_in_range and is_acc_in_range and is_dist_safe)

        output_predicate_file_name = 'fine_predicate_wT_%.2f_wJ_%.2f.bin' % (w_T, w_J)
        output_predicate_file_path = Paths.get_resource_absolute_path_filename('predicates/%s' % output_predicate_file_name)
        BinaryReadWrite.save(array=predicate, file_path=output_predicate_file_path)
