import numpy as np

from decision_making.src.messages.str_serializable import StrSerializable

# General constants

UNKNOWN_DEFAULT_VAL = 0.0


# Communication Layer

# PubSub message class implementation for all DM messages
PUBSUB_MSG_IMPL = StrSerializable


# Behavioral Planner

# [m] high-level behavioral planner lookahead distance
BEHAVIORAL_PLANNING_LOOKAHEAD_DIST = 80.0

# When retrieving the lookahead path of a given dynamic object, we will multiply the path length
# by the following ratio in order to avoid extrapolation when resampling the path (due to path sampling
# and linearization errors)
PREDICTION_LOOKAHEAD_COMPENSATION_RATIO = 1.2

# The necessary lateral margin in [m] that needs to be taken in order to assume that it is not in car's way
LATERAL_SAFETY_MARGIN_FROM_OBJECT = 0.0

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
OBSTACLE_SIGMOID_COST = 1.0 * 1e4           # cost around obstacles (sigmoid)
OBSTACLE_SIGMOID_K_PARAM = 5.5              # sigmoid k (slope) param of objects on road

DEVIATION_FROM_LANE_COST = 20               # cost of deviation from lane (sigmoid)
LANE_SIGMOID_K_PARAM = 4                    # sigmoid k (slope) param of going out-of-lane-center

DEVIATION_TO_SHOULDER_COST = 3.0 * 1e2      # cost of deviation to shoulders (sigmoid)
SHOULDER_SIGMOID_K_PARAM = 6                # sigmoid k (slope) param of going out-of-shoulder
SHOULDER_SIGMOID_OFFSET = 0.2               # offset param m of going out-of-shoulder: cost = w/(1+e^(k*(m+x)))

DEVIATION_FROM_ROAD_COST = 1.0 * 1e4        # cost of deviation from road (sigmoid)
ROAD_SIGMOID_K_PARAM = 20                   # sigmoid k (slope) param of going out-of-road

DEVIATION_FROM_GOAL_LAT_FACTOR = 4          # ratio between lateral and longitudinal deviation costs from the goal
DEVIATION_FROM_GOAL_COST = 1.0 * 1e2        # cost of longitudinal deviation from the goal
GOAL_SIGMOID_K_PARAM = 0.5                  # sigmoid k (slope) param of going out-of-goal
GOAL_SIGMOID_OFFSET = 7                     # offset param m of going out-of-goal: cost = w/(1+e^(k*(m-d)))

# [m/sec] speed to plan towards by default in BP
BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED = 14.0  # TODO - get this value from the map

# [m/s] min & max velocity limits are additional parameters for TP
VELOCITY_LIMITS = np.array([0.0, 20.0])

# Planning horizon for the TP query sent by BP [sec]
# Used for grid search in the [T_MIN, T_MAX] range with resolution of T_RES
BP_SPECIFICATION_T_MIN = 2.0
BP_SPECIFICATION_T_MAX = 20.0
BP_SPECIFICATION_T_RES = 0.2

# Longitudinal Acceleration Limits [m/sec^2]
LON_ACC_LIMITS = np.array([-8.0, 3.0])  # taken from SuperCruise presentation
A_LON_EPS = 2.0

# Latitudinal Acceleration Limits [m/sec^2]
LAT_ACC_LIMITS = np.array([-3.0, 3.0])

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

# [m] Resolution for the interpolation of the reference route
TRAJECTORY_ARCLEN_RESOLUTION = 0.5

# [seconds] Resolution for the visualization of predicted dynamic objects
VISUALIZATION_PREDICTION_RESOLUTION = 1.0

# Time to remember object after it disappears from perception (in nanoseconds)
OBJECT_HISTORY_TIMEOUT = 1000*1000*1000*2

# Curve interpolation type (order)
TRAJECTORY_CURVE_SPLINE_FIT_ORDER = 4

# [m] Do not consider obstacles that are distant than this threshold
TRAJECTORY_OBSTACLE_LOOKAHEAD = 200.0

# Desired resolution of map [m], used for accuracy test on Frenet-Serret transformations
ROAD_MAP_REQUIRED_RES = 5

# [m] Cost function clips higher distances before exponentiation
EXP_CLIP_TH = 50.0

# Number of (best) trajectories to publish to visualization
NUM_ALTERNATIVE_TRAJECTORIES = 75

# Number of points in trajectories for sending out to visualization (currently VizTool freezes when there are too much)
MAX_NUM_POINTS_FOR_VIZ = 60

# in meters, to be used as an argument in the resample_curve method
DOWNSAMPLE_STEP_FOR_REF_ROUTE_VISUALIZATION = 0.5

# [m] "Negligible distance" threshold between the desired location and the actual location between two TP planning
# iterations. If the distance is lower than this threshold, the TP plans the trajectory as is the ego vehicle is
# currently in the desired location and not in its actual location.
# TODO: fix real values for thresholds (after tuning controller)
NEGLIGIBLE_DISPOSITION_LON = 3  # longitudinal (ego's heading direction) difference threshold
NEGLIGIBLE_DISPOSITION_LAT = 3  # lateral (ego's side direction) difference threshold

# [sec] Time-Resolution for the trajectory's discrete points that are sent to the controller
TRAJECTORY_TIME_RESOLUTION = 0.1

# Number of trajectory points to send out (to controller) from the TP - including the current state of ego
TRAJECTORY_NUM_POINTS = 10

# Werling Planner #

# [sec] Time-Resolution for planning
WERLING_TIME_RESOLUTION = 0.1

# [m] Range for grid search in werling planner (long. position)
SX_OFFSET_MIN, SX_OFFSET_MAX = -15, 0.1

# [m] Range for grid search in werling planner (long. velocity)
SV_OFFSET_MIN, SV_OFFSET_MAX = 0, 0

# [m] Range for grid search in werling planner (lat. position)
DX_OFFSET_MIN, DX_OFFSET_MAX = -3, 3

# Linspace number of steps in the constraints parameters grid-search
SX_STEPS, SV_STEPS, DX_STEPS = 10, 1, 7

# Linspace number of steps in latitudinal horizon planning time (from Td_low_bound to Ts)
TD_STEPS = 5

# Minimal T_d (time-horizon for the lateral movement) - in units of WerlingPlanner.dt
TD_MIN_DT = 3

# Frenet-Serret Conversions #

# [1/m] Curvature threshold for the GD step (if value is smaller than this value, there is no step executed)
TINY_CURVATURE = 10e-5

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

# Whether we filter out dynamic objects that are not on the road
# Request by perception for viewing recordings in non-mapped areas.
# SHOULD ALWAYS BE TRUE FOR NORMAL DM FLOW
FILTER_OFF_ROAD_OBJECTS = True

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
RVIZ_MODULE_NAME_FOR_LOGGING = "Rviz Module"

