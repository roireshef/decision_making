from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone as ASpecFilterIfNone, \
    FilterForKinematics, FilterForSafetyTowardsTargetVehicle, BeyondSpecCurvatureFilter, FilterForLaneSpeedLimits, \
    BeyondSpecPartialGffFilter, BeyondSpecSpeedLimitFilter, FilterStopActionIfTooSoonByTime, FilterForSLimit, \
    StaticTrafficFlowControlFilter, BeyondSpecStaticTrafficFlowControlFilter
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterIfNone as RecipeFilterIfNone, \
    FilterActionsTowardsNonOccupiedCells, FilterActionsTowardBackAndParallelCells, FilterOvertakeActions, \
    FilterIfNoLane, FilterLaneChangingIfNotAugmented, FilterSpeedingOverDesiredVelocityDynamic, \
    FilterSpeedingOverDesiredVelocityStatic, FilterActionsTowardsCellsWithoutStopSignsOrStopBars, FilterRoadSignActions
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from rte.python.logger.AV_logger import AV_Logger

# TODO: remove FilterIfAggressive() once we introduce lateral and longitudinal acceleration checks in Cartesian frame
DEFAULT_STATIC_RECIPE_FILTERING = RecipeFiltering(filters=[RecipeFilterIfNone(),
                                                           FilterIfNoLane(),
                                                           FilterLaneChangingIfNotAugmented(),
                                                           FilterSpeedingOverDesiredVelocityStatic()
                                                           ], logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING))
DEFAULT_DYNAMIC_RECIPE_FILTERING = RecipeFiltering(filters=[RecipeFilterIfNone(),
                                                            FilterActionsTowardsNonOccupiedCells(),
                                                            FilterActionsTowardBackAndParallelCells(),
                                                            FilterOvertakeActions(),
                                                            FilterLaneChangingIfNotAugmented(),
                                                            FilterSpeedingOverDesiredVelocityDynamic()
                                                            ],
                                                   logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING))
DEFAULT_ROAD_SIGN_RECIPE_FILTERING = RecipeFiltering(filters=[RecipeFilterIfNone(),
                                                              FilterActionsTowardsCellsWithoutStopSignsOrStopBars()
                                                              # FilterRoadSignActions(),
                                                              ],
                                                     logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING))
DEFAULT_ACTION_SPEC_FILTERING = ActionSpecFiltering(filters=[ASpecFilterIfNone(),
                                                             FilterForSLimit(),
                                                             FilterForKinematics(),
                                                             FilterForLaneSpeedLimits(),
                                                             FilterForSafetyTowardsTargetVehicle(),
                                                             StaticTrafficFlowControlFilter(),
                                                             BeyondSpecStaticTrafficFlowControlFilter(),
                                                             BeyondSpecSpeedLimitFilter(),
                                                             BeyondSpecCurvatureFilter(),
                                                             BeyondSpecPartialGffFilter(),
                                                             FilterStopActionIfTooSoonByTime()
                                                             ],
                                                    logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING))
