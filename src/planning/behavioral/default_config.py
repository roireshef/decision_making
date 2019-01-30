from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterIfNone, \
    FilterActionsTowardsNonOccupiedCells, FilterActionsTowardBackAndParallelCells, FilterOvertakeActions, \
    FilterLimitsViolatingTrajectory, FilterIfNoLane, FilterIfAggressive, FilterLaneChanging, \
    FilterUnsafeExpectedTrajectory
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from rte.python.logger.AV_logger import AV_Logger

# TODO: remove FilterIfAggressive() once we introduce lateral and longitudinal acceleration checks in Cartesian frame
DEFAULT_STATIC_RECIPE_FILTERING = RecipeFiltering(filters=[FilterIfNone(), FilterIfNoLane(), FilterIfAggressive(),
                                                           FilterLaneChanging(),
                                                           FilterLimitsViolatingTrajectory('predicates')],
                                                  logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING))
DEFAULT_DYNAMIC_RECIPE_FILTERING = RecipeFiltering(filters=[FilterIfNone(), FilterActionsTowardsNonOccupiedCells(),
                                                            FilterActionsTowardBackAndParallelCells(),
                                                            FilterOvertakeActions(), FilterLaneChanging(),
                                                            FilterLimitsViolatingTrajectory('predicates'),
                                                            FilterUnsafeExpectedTrajectory('predicates')],
                                                   logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING))


