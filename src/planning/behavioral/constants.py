from decision_making.src.global_constants import LON_ACC_LIMITS, VELOCITY_LIMITS, EPS
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterIfNone, \
    FilterActionsTowardsNonOccupiedCells, FilterActionsTowardBackAndParallelCells, FilterNonCalmActions, \
    FilterOvertakeActions, FilterBadExpectedTrajectory, FilterIfNoLane
from decision_making.src.planning.behavioral.filtering.recipe_filtering import RecipeFiltering
import numpy as np

# Velocity limits [m/s] for Static Recipe enumeration
from decision_making.src.planning.types import LIMIT_MIN, LIMIT_MAX

MIN_VELOCITY = 20 / 3.6
MAX_VELOCITY = 100 / 3.6
VELOCITY_STEP = 10 / 3.6

DEFAULT_STATIC_RECIPE_FILTERING = RecipeFiltering(filters=[FilterIfNone(), FilterIfNoLane(),
                                                           FilterBadExpectedTrajectory('predicates')])
DEFAULT_DYNAMIC_RECIPE_FILTERING = RecipeFiltering(filters=[FilterIfNone(), FilterActionsTowardsNonOccupiedCells(),
                                                            FilterActionsTowardBackAndParallelCells(),
                                                            FilterOvertakeActions(), FilterNonCalmActions(),
                                                            FilterBadExpectedTrajectory('predicates')])

# Distance, velocity and acceleration grids for brute-force filtering purposes
st_limits = [0, 110]
a_0_grid = np.arange(LON_ACC_LIMITS[LIMIT_MIN], LON_ACC_LIMITS[LIMIT_MAX]+EPS, 0.5)
v_0_grid = np.arange(VELOCITY_LIMITS[LIMIT_MIN], VELOCITY_LIMITS[LIMIT_MAX]+EPS, 0.5)
v_T_grid = np.arange(VELOCITY_LIMITS[LIMIT_MIN], VELOCITY_LIMITS[LIMIT_MAX]+EPS, 0.5)
s_T_grid = np.arange(st_limits[LIMIT_MIN], st_limits[LIMIT_MAX]+EPS, 1)
