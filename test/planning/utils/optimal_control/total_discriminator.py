from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import cartesian
from sklearn import tree
from decision_making.paths import Paths
from decision_making.src.global_constants import VELOCITY_LIMITS, EPS, BP_JERK_S_JERK_D_TIME_WEIGHTS, \
    BP_ACTION_T_LIMITS, LON_ACC_LIMITS
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.src.planning.utils.math import Math
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import \
    create_action_time_cost_func_deriv, create_motion_funcs
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('Qt5Agg')



st_limits = [0, 100]

v_0_grid = np.arange(VELOCITY_LIMITS[0], VELOCITY_LIMITS[1] + EPS, 1)
a_0_grid = np.arange(-2, 2 + EPS, 0.5)
s_T_grid = np.arange(st_limits[0], st_limits[1] + EPS, 0.5)
v_T_grid = np.arange(VELOCITY_LIMITS[0], VELOCITY_LIMITS[1] + EPS, 0.5)

predicate_path = Paths.get_resource_absolute_path_filename('predicates/fine_predicate_wT_0.10_wJ_2.00.bin')
wT_str, wJ_str = predicate_path.split('.bin')[0].split('_')[-3], predicate_path.split('.bin')[0].split('_')[-1]

predicate_shape = (int(v_0_grid.shape[0]), int(a_0_grid.shape[0]), int(s_T_grid.shape[0]), int(v_T_grid.shape[0]))
predicate = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)
X = cartesian([v_0_grid, a_0_grid, s_T_grid, v_T_grid])
Y = np.matrix.flatten(predicate).astype(int)

# print('Started training')
# # clf = tree.DecisionTreeClassifier()
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
#
# clf = clf.fit(X, Y)
# print('Finished training')


num_trials = 1000
estim = np.zeros(shape=num_trials)
result = np.zeros(shape=num_trials)

for i in range(1, num_trials):
    print('Num trial: %d out of %d' % (i, num_trials))
    v_0 = int(np.random.uniform(v_0_grid[0], v_0_grid[-1]))
    a_0 = int(np.random.uniform(a_0_grid[0], a_0_grid[-1]))
    s_T = int(np.random.uniform(s_T_grid[0], s_T_grid[-1]))
    v_T = int(np.random.uniform(v_T_grid[0], v_T_grid[-1]))
    lambda_func = create_action_time_cost_func_deriv(float(wT_str), float(wJ_str), a_0, v_0, v_T, s_T, T_m=2)
    t = np.arange(BP_ACTION_T_LIMITS[0] - 0.2,
                  BP_ACTION_T_LIMITS[1] + 0.2, 0.001)
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

    estim[i] = predicate[Math.ind_on_uniform_axis(v_0, v_0_grid),
                         Math.ind_on_uniform_axis(a_0, a_0_grid),
                         Math.ind_on_uniform_axis(s_T, s_T_grid),
                         Math.ind_on_uniform_axis(v_T, v_T_grid)]
    print('')

print('Num of errors: %d' % np.sum(np.abs((result - estim))))
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph
#
# # After being fitted, the model can then be used to predict the class of samples:
# clf.predict([[2., 2.]])
#
# # Alternatively, the probability of each class can be predicted, which is the fraction of training samples of the same class in a leaf:
# clf.predict_proba([[2., 2.]])
