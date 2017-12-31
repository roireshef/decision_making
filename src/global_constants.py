import os
import numpy as np

# Behavioral Planner

# [m] high-level behavioral planner lookahead distance
BEHAVIORAL_PLANNING_LOOKAHEAD_DIST = 60.0
# TODO - get this value from the map
BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT = 14.0

# The necessary lateral margin in [m] that needs to be taken in order to assume that it is not in car's way
LATERAL_SAFETY_MARGIN_FROM_OBJECT = 0.1

# A lower and upper thresholds on the longitudinal offset between object and ego.
# Any object out of this scope won't be accounted in the behavioral planning process
# currently unused
MAX_PLANNING_DISTANCE_BACKWARD = -20.0
MAX_PLANNING_DISTANCE_FORWARD = 80.0

# Planning horizon in [sec]: prediction horizon in the behavioral planner
BEHAVIORAL_PLANNING_HORIZON = 7.0
BEHAVIORAL_PLANNING_TRAJECTORY_HORIZON = 2.0

# Planning resolution in [sec].
# Used for prediction resolution, i.e. the resolution of states along the path to be used for acda computations:
# This may have an effect on efficiency if the acda computation is costly.
BEHAVIORAL_PLANNING_TIME_RESOLUTION = 0.1

# Trajectory cost parameters
INFINITE_SIGMOID_COST = 2.0 * 1e2           # cost around obstacles (sigmoid)
DEVIATION_FROM_ROAD_COST = 1.0 * 1e2        # cost of deviation from road (sigmoid)
DEVIATION_TO_SHOULDER_COST = 1.0 * 1e2      # cost of deviation to shoulders (sigmoid)
OUT_OF_LANE_COST = 0.0                      # cost of deviation from lane (sigmoid)
ROAD_SIGMOID_K_PARAM = 1000.0               # sigmoid k (slope) param of going out-of-road
OBJECTS_SIGMOID_K_PARAM = 20.0              # sigmoid k (slope) param of objects on road
DEVIATION_FROM_GOAL_LON_COST = 1.0 * 1e2    # cost of squared longitudinal deviation from the goal
DEVIATION_FROM_GOAL_LAT_COST = 1.5 * 1e2    # cost of squared lateral deviation from the goal
DEVIATION_FROM_REF_ROUTE_COST = 0.0         # cost of squared deviation from the route path

# [m/s] min & max velocity limits are additional parameters for TP
VELOCITY_LIMITS = np.array([0.0, 60.0])

# Planning horizon for the TP query sent by BP [sec]
# Used for grid search in the [T_MIN, T_MAX] range with resolution of T_RES
BP_SPECIFICATION_T_MIN = 2.0
BP_SPECIFICATION_T_MAX = 20.0
BP_SPECIFICATION_T_RES = 0.2

# Longitudinal Acceleration Limits [m/sec^2]
A_LON_MIN = -4.0
A_LON_MAX = 4.0
A_LON_EPS = 3.0

# Latitudinal Acceleration Limits [m/sec^2]
A_LAT_MIN = -2.0
A_LAT_MAX = 2.0

# Assumed response delay on road [sec]
# Used to compute safe distance from other agents on road
SAFE_DIST_TIME_DELAY = 1.0

# Semantic Grid indices
SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, SEMANTIC_CELL_LON_REAR = 1, 0, -1
SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LAT_RIGHT = 1, 0, -1

# [m/sec] Minimal difference of velocities to justify an overtake
MIN_OVERTAKE_VEL = 3

# [m] The margin that we take from the front/read of the vehicle to define the front/rear partitions
LON_MARGIN_FROM_EGO = 1


# Trajectory Planner #

# [m] length of reference trajectory provided by behavioral planner
REFERENCE_TRAJECTORY_LENGTH = 30.0
REFERENCE_TRAJECTORY_LENGTH_EXTENDED = 31.0

# [m] Resolution for the interpolation of the reference route
TRAJECTORY_ARCLEN_RESOLUTION = 0.1

# [seconds] Resolution for the visualization of predicted dynamic objects
VISUALIZATION_PREDICTION_RESOLUTION = 1.0

# Time to remember object after it disappears from perception (in nanoseconds)
OBJECT_HISTORY_TIMEOUT = 1000*1000*1000*2

# Curve interpolation type (order)
TRAJECTORY_CURVE_INTERP_TYPE = 'cubic'

# [m] Do not consider obstacles that are distant than this threshold
TRAJECTORY_OBSTACLE_LOOKAHEAD = 200.0

# [m] Cost function clips higher distances before exponentiation
EXP_CLIP_TH = 50.0

# Number of (best) trajectories to publish to visualization
NUM_ALTERNATIVE_TRAJECTORIES = 10

# Werling Planner #

# [sec] Time-Resolution for planning
WERLING_TIME_RESOLUTION = 0.1

# [m] Range for grid search in werling planner (long. position)
SX_OFFSET_MIN, SX_OFFSET_MAX = -3, 0.1

# [m] Range for grid search in werling planner (long. velocity)
SV_OFFSET_MIN, SV_OFFSET_MAX = 0, 0

# [m] Range for grid search in werling planner (lat. position)
# TODO: set lateral offsets
DX_OFFSET_MIN, DX_OFFSET_MAX = -1, 1

# Linspace number of steps in the constraints parameters grid-search
SX_STEPS, SV_STEPS, DX_STEPS = 15, 1, 5

# State #

# TODO: set real values
# [m] Bounding box size around ego vehicle
EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT = 5.0, 2.0, 2.0

# [m] The distance from ego frame origin to ego rear
EGO_ORIGIN_LON_FROM_REAR = 4

#The id of the ego object
EGO_ID = 0

# [m] Default height for objects - State Module
DEFAULT_OBJECT_Z_VALUE = 0.

### DM Manager configuration ###
BEHAVIORAL_PLANNING_MODULE_PERIOD = 1.0
TRAJECTORY_PLANNING_MODULE_PERIOD = 0.2

#### NAMES OF MODULES FOR LOGGING ####
MAP_NAME_FOR_LOGGING = "Map API"
DM_MANAGER_NAME_FOR_LOGGING = "DM Manager"
NAVIGATION_PLANNING_NAME_FOR_LOGGING = "Navigation Planning"
BEHAVIORAL_PLANNING_NAME_FOR_LOGGING = "Behavioral Planning"
BEHAVIORAL_STATE_NAME_FOR_LOGGING = "Behavioral State"
BEHAVIORAL_POLICY_NAME_FOR_LOGGING = "Behavioral Policy"
ACDA_NAME_FOR_LOGGING = "ACDA Module"
TRAJECTORY_PLANNING_NAME_FOR_LOGGING = "Trajectory Planning"
STATE_MODULE_NAME_FOR_LOGGING = "State Module"


# Predictor #

# This margin in [m] will be added to the lookahead path of dynamic objects to avoid
# extrapolation when resampling the curve
PREDICTION_LOOKAHEAD_LINEARIZATION_MARGIN = 3.0


# Navigation #

ONE_TWO_NAVIGATION_PLAN = np.array([1, 2])
