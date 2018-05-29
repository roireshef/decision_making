import numpy as np
import os

from decision_making.paths import Paths
from decision_making.src.global_constants import FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_S_T_GRID, FILTER_V_T_GRID, \
    SAFE_DIST_TIME_DELAY
from decision_making.src.planning.behavioral.data_objects import ActionType
from decision_making.src.planning.utils.file_utils import BinaryReadWrite
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import QuinticMotionPredicatesCreator


def test_createPredicates_predicateFileMatchesCurrentPredicateGeneration():
    # This test generates some random motion parameters (v0,a0,sT,vT) and generates a predicate for these values
    # using QuinticMotionPredicatesCreator create_predicates method. It then checks that all predicates generated now
    # are aligned with the values stored in the predicate file.
    directory = Paths.get_resource_absolute_path_filename('predicates')
    num_trials = 1000
    T_m = SAFE_DIST_TIME_DELAY
    for filename in os.listdir(directory):
        if filename.endswith(".bin"):
            result_now = np.zeros(shape=num_trials)
            result_lut = np.zeros(shape=num_trials)
            predicate_path = Paths.get_resource_absolute_path_filename('%s/%s' % ('predicates', filename))
            action_name = filename.split('.bin')[0].split('_predicate')[0]

            if action_name == 'follow_lane':
                continue

            wT, wJ = [float(filename.split('.bin')[0].split('_')[4]),
                      float(filename.split('.bin')[0].split('_')[6])]
            predicate_shape = (
                len(FILTER_V_0_GRID), len(FILTER_A_0_GRID), len(FILTER_S_T_GRID), len(FILTER_V_T_GRID))
            predicates_creator = QuinticMotionPredicatesCreator(FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_S_T_GRID,
                                                                FILTER_V_T_GRID, T_m, 'predicates')
            predicate = BinaryReadWrite.load(file_path=predicate_path, shape=predicate_shape)

            for i in range(num_trials):
                v_0 = int(np.random.uniform(FILTER_V_0_GRID.start, FILTER_V_0_GRID.end))
                a_0 = int(np.random.uniform(FILTER_A_0_GRID.start, FILTER_A_0_GRID.end))
                s_T = int(np.random.uniform(FILTER_S_T_GRID.start, FILTER_S_T_GRID.end))
                v_T = int(np.random.uniform(FILTER_V_T_GRID.start, FILTER_V_T_GRID.end))

                action_type = ActionType.FOLLOW_VEHICLE if action_name == 'follow_vehicle' else ActionType.OVERTAKE_VEHICLE
                margin_sign = +1 if action_type == ActionType.FOLLOW_VEHICLE else -1
                result_now[i] = predicates_creator.generate_predicate_value(action_type, wT, wJ, a_0,
                                                                            v_0, v_T, margin_sign*s_T, margin_sign*T_m)
                result_lut[i] = predicate[FILTER_V_0_GRID.get_index(v_0),
                                          FILTER_A_0_GRID.get_index(a_0),
                                          FILTER_S_T_GRID.get_index(s_T),
                                          FILTER_V_T_GRID.get_index(v_T)]

            np.testing.assert_array_almost_equal(result_now, result_lut)





