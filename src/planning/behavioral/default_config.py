from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterIfNone, \
    FilterActionsTowardsNonOccupiedCells, FilterActionsTowardBackAndParallelCells, FilterNonCalmActions, \
    FilterOvertakeActions, FilterBadExpectedTrajectory, FilterIfNoLane
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering

DEFAULT_STATIC_RECIPE_FILTERING = RecipeFiltering(filters=[FilterIfNone(), FilterIfNoLane(),
                                                           FilterBadExpectedTrajectory('predicates')])
DEFAULT_DYNAMIC_RECIPE_FILTERING = RecipeFiltering(filters=[FilterIfNone(), FilterActionsTowardsNonOccupiedCells(),
                                                            FilterActionsTowardBackAndParallelCells(),
                                                            FilterOvertakeActions(), FilterNonCalmActions(),
                                                            FilterBadExpectedTrajectory('predicates')])


