from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone as ASpecFilterIfNone, \
    FilterForKinematics, FilterForSafetyTowardsTargetVehicle, StaticTrafficFlowControlFilter, \
    BeyondSpecStaticTrafficFlowControlFilter, BeyondSpecCurvatureFilter, FilterForLaneSpeedLimits, \
    BeyondSpecSpeedLimitFilter, FilterStopActionIfTooSoonByTime
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterIfNone as RecipeFilterIfNone, \
    FilterActionsTowardsNonOccupiedCells, FilterActionsTowardBackAndParallelCells, FilterOvertakeActions, \
    FilterIfNoLane, FilterLaneChanging, FilterSpeedingOverDesiredVelocityDynamic, \
    FilterSpeedingOverDesiredVelocityStatic, FilterActionsTowardsCellsWithoutRoadSigns
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
from rte.python.logger.AV_logger import AV_Logger

# TODO: remove FilterIfAggressive() once we introduce lateral and longitudinal acceleration checks in Cartesian frame
DEFAULT_STATIC_RECIPE_FILTERING = RecipeFiltering(filters=[RecipeFilterIfNone(),
                                                           FilterIfNoLane(),
                                                           FilterLaneChanging(),
                                                           FilterSpeedingOverDesiredVelocityStatic()
                                                           ], logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING))
DEFAULT_DYNAMIC_RECIPE_FILTERING = RecipeFiltering(filters=[RecipeFilterIfNone(),
                                                            FilterActionsTowardsNonOccupiedCells(),
                                                            FilterActionsTowardBackAndParallelCells(),
                                                            FilterOvertakeActions(),
                                                            FilterLaneChanging(),
                                                            FilterSpeedingOverDesiredVelocityDynamic()
                                                            ],
                                                   logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING))
DEFAULT_ROAD_SIGN_RECIPE_FILTERING = RecipeFiltering(filters=[RecipeFilterIfNone(),
                                                              FilterActionsTowardsCellsWithoutRoadSigns(),
                                                              FilterLaneChanging()
                                                              ],
                                                     logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING))
DEFAULT_ACTION_SPEC_FILTERING = ActionSpecFiltering(filters=[ASpecFilterIfNone(),
                                                             FilterForKinematics(),
                                                             FilterForLaneSpeedLimits(),
                                                             FilterForSafetyTowardsTargetVehicle(),
                                                             StaticTrafficFlowControlFilter(),
                                                             BeyondSpecStaticTrafficFlowControlFilter(),
                                                             BeyondSpecSpeedLimitFilter(),
                                                             BeyondSpecCurvatureFilter(),
                                                             FilterStopActionIfTooSoonByTime()
                                                             ],
                                                    logger=AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING))
