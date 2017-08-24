
# The necessary lateral margin in [m] that needs to be taken in order to assume that it is not in car's way
LATERAL_MARGIN_FROM_OBJECT_TO_ASSUME_OUT_OF_WAY = 0.2

# Define the grid of additive lateral offsets in [lanes] to current car's latitude on road
POLICY_ACTION_SPACE_ADDITIVE_LATERAL_OFFSETS_IN_LANES = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]

# A lower and upper thresholds on the longitudinal offset between object and ego.
# Any object out of this scope won't be accounted in the behavioral planning process
MIN_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING = -20.0
MAX_DISTANCE_OF_OBJECT_FROM_EGO_FOR_FILTERING = 50.0
