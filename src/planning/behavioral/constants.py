# The necessary lateral margin in [m] that needs to be taken in order to assume that it is not in car's way
LATERAL_SAFETY_MARGIN_FROM_OBJECT = 0.2

# Define the grid of additive lateral offsets in [lanes] to current car's latitude on road
POLICY_ACTION_SPACE_ADDITIVE_LATERAL_OFFSETS_IN_LANES = [-2.0, -1.0, 0.0, 1.0, 2.0]

# A lower and upper thresholds on the longitudinal offset between object and ego.
# Any object out of this scope won't be accounted in the behavioral planning process
MIN_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING = -20.0
MAX_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING = 80.0

# Planning horizon in [sec]: prediction horizon in the behavioral planner
BEHAVIORAL_PLANNING_HORIZON = 5.0
BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON = 2.0

# Planning resolution in [sec].
# Used for prediction resolution, i.e. the resolution of states along the path to be used for acda computations:
# This may have an effect on efficiency if the acda computation is costly.
BEHAVIORAL_PLANNING_TIME_RESOLUTION = 0.1
