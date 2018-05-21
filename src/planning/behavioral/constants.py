from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterIfNone, \
    FilterActionsTowardsNonOccupiedCells, FilterActionsTowardBackAndParallelCells, FilterNonCalmActions, \
    FilterOvertakeActions, FilterBadExpectedTrajectory, FilterIfNoLane
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering

# Velocity limits [m/s] for Static Recipe enumeration
MIN_VELOCITY = 20 / 3.6
MAX_VELOCITY = 100 / 3.6
VELOCITY_STEP = 10 / 3.6

DEFAULT_STATIC_RECIPE_FILTERING = RecipeFiltering(filters=[FilterIfNone(), FilterIfNoLane()])
DEFAULT_DYNAMIC_RECIPE_FILTERING = RecipeFiltering(filters=[FilterIfNone(), FilterActionsTowardsNonOccupiedCells(),
                                                            FilterActionsTowardBackAndParallelCells(),
                                                            FilterOvertakeActions(), FilterNonCalmActions(),
                                                            FilterBadExpectedTrajectory('predicates')])
