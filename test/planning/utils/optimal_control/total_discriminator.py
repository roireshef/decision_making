import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.utils.extmath import cartesian

from decision_making.paths import Paths
from decision_making.src.global_constants import VELOCITY_LIMITS, EPS, BP_ACTION_T_LIMITS, LON_ACC_LIMITS
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import create_motion_funcs

plt.switch_backend('Qt5Agg')


def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]


st_limits = [0, 100]

a_0_grid = np.arange(LON_ACC_LIMITS[0], LON_ACC_LIMITS[1]+EPS, 0.5)
v_0_grid = np.arange(VELOCITY_LIMITS[0], 20+EPS, 0.5)
s_T_grid = np.arange(st_limits[0], st_limits[1]+EPS, 1)
v_T_grid = np.arange(VELOCITY_LIMITS[0], 20+EPS, 0.5)

predicate_path = Paths.get_resource_absolute_path_filename('predicates/predicate_wT_0.10_wJ_2.00.bin')
wT_str, wJ_str = predicate_path.split('.bin')[0].split('_')[-3], predicate_path.split('.bin')[0].split('_')[-1]

predicate_shape = (int(v_0_grid.shape[0]), int(a_0_grid.shape[0]), int(s_T_grid.shape[0]), int(v_T_grid.shape[0]))
predicate = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)
X = cartesian([v_0_grid, a_0_grid, s_T_grid, v_T_grid])
Y = np.matrix.flatten(predicate).astype(int)

print('Started training')
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, Y)
print('Finished training')


num_trials = 10000
estim_lut = np.zeros(shape=num_trials)
estim_dt = np.zeros(shape=num_trials)
# estim_svm = np.zeros(shape=num_trials)
# estim_svm_proba_higher = np.zeros(shape=num_trials)
# estim_svm_proba_higher_higher = np.zeros(shape=num_trials)
result = np.zeros(shape=num_trials)


for i in range(1, num_trials):
    print('Num trial: %d out of %d' % (i, num_trials))
    v_0 = (np.random.uniform(v_0_grid[0], v_0_grid[-1]))
    a_0 = (np.random.uniform(a_0_grid[0], a_0_grid[-1]))
    s_T = (np.random.uniform(s_T_grid[0], s_T_grid[-1]))
    v_T = (np.random.uniform(v_T_grid[0], v_T_grid[-1]))
    lambda_func = QuinticPoly1D.time_cost_function_derivative(float(wT_str), float(wJ_str), a_0, v_0, v_T, s_T, T_m=2)
    # lambda_func = create_action_time_cost_func_deriv(float(wT_str), float(wJ_str), a_0, v_0, v_T, s_T, T_m=2)
    t = np.arange(BP_ACTION_T_LIMITS[0], BP_ACTION_T_LIMITS[1], 0.001)
    der = lambda_func(t)
    ind = np.argwhere(np.abs(der) < 0.01)
    if len(ind) == 0:
        result[i] = 0
    else:
        T = t[ind[0]]  # First extrema is our local (and sometimes global) minimum
        delta_s_t_func, v_t_func, a_t_func = create_motion_funcs(a_0, v_0, v_T, s_T, T)
        t = np.arange(0, T, 0.01)
        min_delta_s = min(delta_s_t_func(t))
        min_v, max_v = min(v_t_func(t)), max(v_t_func(t))
        min_a, max_a = min(a_t_func(t)), max(a_t_func(t))

        is_T_in_range = (T > BP_ACTION_T_LIMITS[0]) and (T < BP_ACTION_T_LIMITS[1])
        is_vel_in_range = (min_v >= VELOCITY_LIMITS[0]) and (max_v <= VELOCITY_LIMITS[1])
        is_acc_in_range = (min_a >= LON_ACC_LIMITS[0]) and (max_a <= LON_ACC_LIMITS[1])
        is_dist_safe = min_delta_s >= 2 * v_T

        result[i] = (is_T_in_range and is_vel_in_range and is_acc_in_range and is_dist_safe)

    estim_lut[i] = predicate[Math.ind_on_uniform_axis(v_0, v_0_grid),
                         Math.ind_on_uniform_axis(a_0, a_0_grid),
                         Math.ind_on_uniform_axis(s_T, s_T_grid),
                         Math.ind_on_uniform_axis(v_T, v_T_grid)]
    estim_dt[i] = clf.predict(np.array([v_0, a_0, s_T, v_T]).reshape(1, -1))[0]
    # estim_svm[i] = clf_svm.predict(np.array([v_0, a_0, s_T, v_T]).reshape(1, -1))[0]
    # estim_svm_proba_higher[i] = clf_svm.predict_proba(np.array([v_0, a_0, s_T, v_T]).reshape(1, -1))[0][1] > 0.6
    # estim_svm_proba_higher_higher[i] = clf_svm.predict_proba(np.array([v_0, a_0, s_T, v_T]).reshape(1, -1))[0][1] > 0.7

    print('')

print('percent of error for LUT: %.4f and FP rate is: %.4f' % (np.mean(np.abs((result - estim_lut)))*100, np.mean((estim_lut-result) == 1)*100))
print('percent of error for Decision Tree: %.4f and FP rate is: %.4f' % (np.mean(np.abs((result - estim_dt)))*100, np.mean((estim_dt-result) == 1)*100))
# print('percent of error for Random Forest: %.4f and FP rate is: %.4f' % (np.mean(np.abs((result - estim_svm)))*100, np.mean((estim_svm-result) == 1)*100))
# print('percent of error for Random Forest with 0.6: %.4f and FP rate is: %.4f' % (np.mean(np.abs((result - estim_svm_proba_higher)))*100, np.mean((estim_svm_proba_higher-result) == 1)*100))
# print('percent of error for Random Forest with 0.7: %.4f and FP rate is: %.4f' % (np.mean(np.abs((result - estim_svm_proba_higher_higher)))*100, np.mean((estim_svm_proba_higher_higher-result) == 1)*100))

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph
#
# # After being fitted, the model can then be used to predict the class of samples:
# clf.predict([[2., 2.]])
#
# # Alternatively, the probability of each class can be predicted, which is the fraction of training samples of the same class in a leaf:
# clf.predict_proba([[2., 2.]])
