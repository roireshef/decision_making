import re

from decision_making.paths import Paths
from decision_making.src.global_constants import FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_S_T_GRID, FILTER_V_T_GRID, \
    SAFE_DIST_TIME_DELAY, BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.data_objects import ActionType
from decision_making.src.planning.utils.file_utils import TextReadWrite
from decision_making.test.planning.utils.optimal_control.quartic_poly_formulas import QuarticMotionPredicatesCreator
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import QuinticMotionPredicatesCreator


def main():
    weights = BP_JERK_S_JERK_D_TIME_WEIGHTS
    dynamic_actions = [ActionType.FOLLOW_VEHICLE, ActionType.OVERTAKE_VEHICLE]
    resources_directory = 'predicates'
    # Quintic predicates creation
    quintic_predicates_creator = QuinticMotionPredicatesCreator(FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_S_T_GRID,
                                                                FILTER_V_T_GRID, T_m=SAFE_DIST_TIME_DELAY,
                                                                predicates_resources_target_directory=resources_directory)
    quintic_predicates_creator.create_predicates(weights, dynamic_actions)
    # Quartic predicates creation
    quartic_predicates_creator = QuarticMotionPredicatesCreator(FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_V_T_GRID, resources_directory)
    quartic_predicates_creator.create_predicates(weights)
    # Create MetaData log file
    document_created_predicates(resources_directory)


def document_created_predicates(target_directory):
    output_predicate_file_name = 'PredicatesMetaData.txt'
    output_predicate_file_path = Paths.get_resource_absolute_path_filename(
        '%s/%s' % (target_directory,
                   output_predicate_file_name))
    lines_in = ['Current predicates were created with constants:',
                'FILTER_V_0_GRID : ' + str(FILTER_V_0_GRID),
                'FILTER_A_0_GRID : ' + str(FILTER_A_0_GRID),
                'FILTER_S_T_GRID : ' + str(FILTER_S_T_GRID),
                'FILTER_V_T_GRID : ' + str(FILTER_V_T_GRID),
                'SAFE_DIST_TIME_DELAY : ' + str(SAFE_DIST_TIME_DELAY)]
    TextReadWrite.write(lines_in, output_predicate_file_path)


if __name__ == '__main__':
    main()
