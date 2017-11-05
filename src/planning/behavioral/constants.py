# The necessary lateral margin in [m] that needs to be taken in order to assume that it is not in car's way
LATERAL_SAFETY_MARGIN_FROM_OBJECT = 0.2

# Define the grid of additive lateral offsets in [lanes] to current car's latitude on road
POLICY_ACTION_SPACE_ADDITIVE_LATERAL_OFFSETS_IN_LANES = [-2.0, -1.0, 0.0, 1.0, 2.0]

# A lower and upper thresholds on the longitudinal offset between object and ego.
# Any object out of this scope won't be accounted in the behavioral planning process
MAX_PLANNING_DISTANCE_BACKWARD = -20.0
MAX_PLANNING_DISTANCE_FORWARD = 80.0

# Planning horizon in [sec]: prediction horizon in the behavioral planner
BEHAVIORAL_PLANNING_HORIZON = 5.0
BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON = 2.0

# Planning resolution in [sec].
# Used for prediction resolution, i.e. the resolution of states along the path to be used for acda computations:
# This may have an effect on efficiency if the acda computation is costly.
BEHAVIORAL_PLANNING_TIME_RESOLUTION = 0.1

# Planning horizon for the TP query sent by BP [sec]
# Used for grid search in the [T_MIN, T_MAX] range with resolution of T_RES
BP_SPECIFICATION_T_MIN = 0.1
BP_SPECIFICATION_T_MAX = 20.0
BP_SPECIFICATION_T_RES = 0.2

# Longitudinal Acceleration Limits [m/sec^2]
A_LON_MIN = -5
A_LON_MAX = 5

# Latitudinal Acceleration Limits [m/sec^2]
A_LAT_MIN = -2
A_LAT_MAX = 2

# Assumed response delay on road [sec]
# Used to compute safe distance from other agents on road
SAFE_DIST_TIME_DELAY = 0.5

# Semantic Grid indices
SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, SEMANTIC_CELL_LON_REAR = 1, 0, -1
SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LAT_RIGHT = 1, 0, -1

# [m/sec] Minimal difference of velocities to justify an overtake
MIN_OVERTAKE_VEL = 2

# [m] The margin that we take from the front/read of the vehicle to define the front/rear partitions
SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO = 1