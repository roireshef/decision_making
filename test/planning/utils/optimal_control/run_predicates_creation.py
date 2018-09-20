from decision_making.paths import Paths
from decision_making.src.global_constants import FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_S_T_GRID, FILTER_V_T_GRID, \
    SPECIFICATION_MARGIN_TIME_DELAY, BP_JERK_S_JERK_D_TIME_WEIGHTS, SAFETY_MARGIN_TIME_DELAY, \
    LON_SAFETY_ACCEL_DURING_DELAY
from decision_making.src.planning.behavioral.data_objects import ActionType
from decision_making.src.planning.utils.file_utils import TextReadWrite
from decision_making.test.planning.utils.optimal_control.quartic_poly_formulas import QuarticMotionPredicatesCreator
from decision_making.test.planning.utils.optimal_control.quintic_poly_formulas import QuinticMotionPredicatesCreator

"""
This script initializes predicate creators objects and then use them to create predicates (boolean LUTs) for filtering 
recipes which are supposed to be non-valid and thus don't need to be specified. The predicates are created based on
constants defined in global_constants and has to stay the same in the environment of the one running it. 
The constants we use are the sets of jerk-time weights, the grids that define the resolution of the predicates and
the safety time behind or ahead of the followed or overtaken vehicle, respectively. These constants are documented by
being written to a .txt file that will be read in the initializing of the predicates and their values will be verified
against the values used in the target machine.  
"""


def main():
    weights = BP_JERK_S_JERK_D_TIME_WEIGHTS
    dynamic_actions = [ActionType.FOLLOW_VEHICLE, ActionType.OVERTAKE_VEHICLE]
    resources_directory = 'predicates'
    # Quintic predicates creation
    quintic_predicates_creator = QuinticMotionPredicatesCreator(FILTER_V_0_GRID, FILTER_A_0_GRID, FILTER_S_T_GRID,
                                                                FILTER_V_T_GRID, T_m=SPECIFICATION_MARGIN_TIME_DELAY,
                                                                T_safety=SAFETY_MARGIN_TIME_DELAY,
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
                'SPECIFICATION_MARGIN_TIME_DELAY : ' + str(SPECIFICATION_MARGIN_TIME_DELAY),
                'SAFETY_MARGIN_TIME_DELAY : ' + str(SAFETY_MARGIN_TIME_DELAY),
                'LON_SAFETY_ACCEL_DURING_DELAY : ' + str(LON_SAFETY_ACCEL_DURING_DELAY)]
    TextReadWrite.write(lines_in, output_predicate_file_path)


if __name__ == '__main__':
    main()
