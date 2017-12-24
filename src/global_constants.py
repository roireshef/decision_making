import os

# Behavioral Planner

# [m] high-level behavioral planner lookahead distance
BEHAVIORAL_PLANNING_LOOKAHEAD_DIST = 60.0
# TODO - get this value from the map
BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT = 14.0

# When retrieving the lookahead path of a given dynamic object, we will multiply the path length
# by the following ratio in order to avoid extrapolation when resampling the path (due to path sampling
# and linearization errors)
PREDICTION_LOOKAHEAD_COMPENSATION_RATIO = 1.1

# Trajectory Planner #

# [m] length of reference trajectory provided by behavioral planner
REFERENCE_TRAJECTORY_LENGTH = 30.0

# [m] extra margin (backwards) for the reference-route generation for TP
#REFERENCE_TRAJECTORY_MARGIN = PREDICTION_LOOKAHEAD_COMPENSATION_RATIO / 2

# [m] Resolution for the interpolation of the reference route
TRAJECTORY_ARCLEN_RESOLUTION = 0.1

# [seconds] Resolution for the visualization of predicted dynamic objects
VISUALIZATION_PREDICTION_RESOLUTION = 1.0

# Time to remember object after it disappears from perception (in nanoseconds)
OBJECT_HISTORY_TIMEOUT = 1000*1000*1000*2

# Curve interpolation type (order)
TRAJECTORY_CURVE_INTERP_TYPE = 'quadratic'

# [m] Do not consider obstacles that are distant than this threshold
TRAJECTORY_OBSTACLE_LOOKAHEAD = 200.0

# [m] Cost function clips higher distances before exponentiation
EXP_CLIP_TH = 50.0

# Number of (best) trajectories to publish to visualization
NUM_ALTERNATIVE_TRAJECTORIES = 10

# [m] "Negligible distance" threshold between the desired location and the actual location between two TP planning
# iterations. If the distance is lower than this threshold, the TP plans the trajectory as is the ego vehicle is
# currently in the desired location and not in its actual location.
# TODO: fix real values for thresholds. those high values are just for testing
NEGLIGIBLE_DISPOSITION_LON = 3  # longitudinal (ego's heading direction) difference threshold
NEGLIGIBLE_DISPOSITION_LAT = 3  # lateral (ego's side direction) difference threshold

# [sec] Time-Resolution for the trajectory's discrete points that are sent to the controller
TRAJECTORY_TIME_RESOLUTION = 0.1

# Number of trajectory points to send out (to controller) from the TP
TRAJECTORY_NUM_POINTS = 10

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

# [m/sec^2] when acceleration is not specified - TP uses this as goal acceleration
DEFAULT_ACCELERATION = 0.0

# [-+1/m] when curvature is not specified - TP uses this as goal curvature
DEFAULT_CURVATURE = 0.0

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

