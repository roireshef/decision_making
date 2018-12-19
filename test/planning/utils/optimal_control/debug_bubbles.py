import matplotlib.pyplot as plt
import numpy as np

from decision_making.paths import Paths
from decision_making.src.global_constants import LON_ACC_LIMITS, VELOCITY_LIMITS, BP_ACTION_T_LIMITS, FILTER_V_0_GRID, \
    FILTER_A_0_GRID, FILTER_V_T_GRID, FILTER_S_T_GRID, SPECIFICATION_MARGIN_TIME_DELAY
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import BinaryReadWrite, EPS, \
    create_quintic_motion_funcs

plt.switch_backend('Qt5Agg')


def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]


is_plot = False

predicate_path = Paths.get_resource_absolute_path_filename('predicates/follow_vehicle_predicate_wT_0.10_wJ_12.00.bin')
wT_str, wJ_str = predicate_path.split('.bin')[0].split('_')[4], predicate_path.split('.bin')[0].split('_')[6]

predicate_shape = (int(len(FILTER_V_0_GRID)), int(len(FILTER_A_0_GRID)), int(len(FILTER_S_T_GRID)), int(len(FILTER_V_T_GRID)))
predicate = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)
predicate_now = np.full(shape=[int(len(FILTER_S_T_GRID)), int(len(FILTER_V_T_GRID))], fill_value=False)

for k, v_0 in enumerate(FILTER_V_0_GRID):
    for m, a_0 in enumerate(FILTER_A_0_GRID):
        if a_0 == 0 and v_0 == 10:
            plt.figure(1)
            plt.imshow(predicate[k][m], aspect='auto', interpolation='none', extent=extents(FILTER_V_T_GRID.array) + extents(FILTER_S_T_GRID.array),
                       origin='lower')
            plt.xlabel(r'$v_T$')
            plt.ylabel(r'$S_T$')
            for i, s_T in enumerate(FILTER_S_T_GRID):
                for j, v_T in enumerate(FILTER_V_T_GRID):
                    if s_T > 77 and v_T < 13:
                        # print('v_0=%.2f,a_0=%.2f,s_T=%.2f,v_T=%.2f' % (v_0, a_0, s_T, v_T))
                        lambda_func = QuinticPoly1D.time_cost_function_derivative(0.1, 2, 0, 20, v_T, s_T, T_m=2)
                        t = np.arange(BP_ACTION_T_LIMITS[0], BP_ACTION_T_LIMITS[1], 0.001)
                        der = lambda_func(t)
                        ind = np.argwhere(np.abs(der) < 0.01)
                        if len(ind) == 0:
                            predicate_now[i, j] = False
                            continue
                        T = t[ind[0]]  # First extrema is our local (and sometimes global) minimum

                        delta_s_t_func, v_t_func, a_t_func = create_quintic_motion_funcs(a_0, v_0, v_T, s_T, T, SPECIFICATION_MARGIN_TIME_DELAY)
                        t = np.arange(0, T, 0.01)
                        min_delta_s = min(delta_s_t_func(t))
                        min_v, max_v = min(v_t_func(t)), max(v_t_func(t))
                        min_a, max_a = min(a_t_func(t)), max(a_t_func(t))

                        is_T_in_range = (T >= BP_ACTION_T_LIMITS[0]) and (T <= BP_ACTION_T_LIMITS[1]+EPS)
                        is_vel_in_range = (min_v >= VELOCITY_LIMITS[0]) and (max_v <= 20+EPS)
                        is_acc_in_range = (min_a >= LON_ACC_LIMITS[0]) and (max_a <= LON_ACC_LIMITS[1]+EPS)
                        is_dist_safe = min_delta_s >= 2*v_T

                        predicate_now[i, j] = (is_T_in_range and is_vel_in_range and is_acc_in_range and is_dist_safe)
                        print('')

plt.figure(2)
plt.imshow(predicate_now, aspect='auto', interpolation='none', extent=extents(FILTER_V_T_GRID.array) + extents(FILTER_S_T_GRID.array),
           origin='lower')
plt.xlabel(r'$v_T$')
plt.ylabel(r'$S_T$')


plt.show()
