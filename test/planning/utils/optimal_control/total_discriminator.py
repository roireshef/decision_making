import numpy as np
from sklearn import tree
from sklearn.utils.extmath import cartesian

from decision_making.paths import Paths
from decision_making.src.global_constants import FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_S_T_GRID, FILTER_V_T_GRID, \
    SPECIFICATION_MARGIN_TIME_DELAY
from decision_making.src.planning.behavioral.data_objects import ActionType
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import QuinticMotionPredicatesCreator


predicate_path = Paths.get_resource_absolute_path_filename('predicates/follow_vehicle_predicate_wT_0.10_wJ_2.00.bin')
filename = 'follow_vehicle_predicate_wT_0.10_wJ_2.00.bin'
wT, wJ = [float(filename.split('.bin')[0].split('_')[4]),
          float(filename.split('.bin')[0].split('_')[6])]

predicate_shape = (int(len(FILTER_V_0_GRID)), int(len(FILTER_A_0_GRID)), int(len(FILTER_S_T_GRID)), int(len(FILTER_V_T_GRID)))
predicate = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)
predicate_now = np.full(shape=[int(len(FILTER_S_T_GRID)), int(len(FILTER_V_T_GRID))], fill_value=False)
X = cartesian([FILTER_V_0_GRID.array, FILTER_A_0_GRID.array, FILTER_S_T_GRID.array, FILTER_V_T_GRID.array])
Y = np.matrix.flatten(predicate).astype(int)

# print('Started training')
# clf = tree.DecisionTreeClassifier()
#
# clf = clf.fit(X, Y)
# print('Finished training')


num_trials = 3000
estim_lut = np.zeros(shape=num_trials)
# estim_dt = np.zeros(shape=num_trials)
# estim_svm = np.zeros(shape=num_trials)
# estim_svm_proba_higher = np.zeros(shape=num_trials)
# estim_svm_proba_higher_higher = np.zeros(shape=num_trials)
result = np.zeros(shape=num_trials)
T_m = SPECIFICATION_MARGIN_TIME_DELAY
predicates_creator = QuinticMotionPredicatesCreator(FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_S_T_GRID, FILTER_V_T_GRID, T_m, 'predicates')


for i in range(num_trials):
    print('Num trial: %d out of %d' % (i, num_trials))
    v_0 = np.random.uniform(FILTER_V_0_GRID.start, FILTER_V_0_GRID.end)
    a_0 = np.random.uniform(FILTER_A_0_GRID.start, FILTER_A_0_GRID.end)
    s_T = np.random.uniform(FILTER_S_T_GRID.start, FILTER_S_T_GRID.end)
    v_T = np.random.uniform(FILTER_V_T_GRID.start, FILTER_V_T_GRID.end)

    result[i] = predicates_creator.generate_predicate_value(ActionType.FOLLOW_VEHICLE, wT, wJ, a_0, v_0, v_T, s_T, T_m)

    estim_lut[i] = predicate[FILTER_V_0_GRID.get_index(v_0),
                             FILTER_A_0_GRID.get_index(a_0),
                             FILTER_S_T_GRID.get_index(s_T),
                             FILTER_V_T_GRID.get_index(v_T)]
    # estim_dt[i] = clf.predict(np.array([v_0, a_0, s_T, v_T]).reshape(1, -1))[0]
    # estim_svm[i] = clf_svm.predict(np.array([v_0, a_0, s_T, v_T]).reshape(1, -1))[0]
    # estim_svm_proba_higher[i] = clf_svm.predict_proba(np.array([v_0, a_0, s_T, v_T]).reshape(1, -1))[0][1] > 0.6
    # estim_svm_proba_higher_higher[i] = clf_svm.predict_proba(np.array([v_0, a_0, s_T, v_T]).reshape(1, -1))[0][1] > 0.7

    print('')

print('percent of error for LUT: %.4f and FP rate is: %.4f' % (np.mean(np.abs((result - estim_lut)))*100, np.mean((estim_lut-result) == 1)*100))
# print('percent of error for Decision Tree: %.4f and FP rate is: %.4f' % (np.mean(np.abs((result - estim_dt)))*100, np.mean((estim_dt-result) == 1)*100))
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
