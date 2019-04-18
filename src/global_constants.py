import numpy as np

from decision_making.src.messages.str_serializable import StrSerializable
from decision_making.src.planning.utils.numpy_utils import UniformGrid

# General constants
EPS = np.finfo(np.float32).eps

# Communication Layer

# PubSub message class implementation for all DM messages
PUBSUB_MSG_IMPL = StrSerializable


# Behavioral Planner

# [m] high-level behavioral planner lookahead distance
PLANNING_LOOKAHEAD_DIST = 100.0

# [m] Maximal horizon distance for building Generalized Frenet Frames
MAX_HORIZON_DISTANCE = 400

# The necessary lateral margin in [m] that needs to be taken in order to assume that it is not in car's way
LATERAL_SAFETY_MARGIN_FROM_OBJECT = 0.0
LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT = 2.0

# After a change of TP costs run the following test:
# test_werlingPlanner.test_werlingPlanner_testCostsShaping_saveImagesForVariousScenarios

# Trajectory cost parameters
OBSTACLE_SIGMOID_COST = 1.0 * 1e5           # cost around obstacles (sigmoid)
OBSTACLE_SIGMOID_K_PARAM = 9.0              # sigmoid k (slope) param of objects on road

DEVIATION_FROM_LANE_COST = 0.07             # cost of deviation from lane (sigmoid)
LANE_SIGMOID_K_PARAM = 4                    # sigmoid k (slope) param of going out-of-lane-center

DEVIATION_TO_SHOULDER_COST = 1.0 * 1e2      # cost of deviation to shoulders (sigmoid)
SHOULDER_SIGMOID_K_PARAM = 8.0              # sigmoid k (slope) param of going out-of-shoulder
SHOULDER_SIGMOID_OFFSET = 0.2               # offset param m of going out-of-shoulder: cost = w/(1+e^(k*(m+x)))

DEVIATION_FROM_ROAD_COST = 1.0 * 1e3        # cost of deviation from road (sigmoid)
ROAD_SIGMOID_K_PARAM = 20                   # sigmoid k (slope) param of going out-of-road

DEVIATION_FROM_GOAL_LAT_LON_RATIO = 3       # ratio between lateral and longitudinal deviation costs from the goal
DEVIATION_FROM_GOAL_COST = 2.5 * 1e2        # cost of longitudinal deviation from the goal
GOAL_SIGMOID_K_PARAM = 0.5                  # sigmoid k (slope) param of going out-of-goal
GOAL_SIGMOID_OFFSET = 7                     # offset param m of going out-of-goal: cost = w/(1+e^(k*(m-d)))

LON_JERK_COST_WEIGHT = 1.0                  # cost of longitudinal jerk
LAT_JERK_COST_WEIGHT = 1.0                  # cost of lateral jerk

# [m/sec] speed to plan towards by default in BP
BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED = 14.0  # TODO - get this value from the map

# [m/s] min & max velocity limits are additional parameters for TP and for Static Recipe enumeration
VELOCITY_LIMITS = np.array([0.0, 20])
VELOCITY_STEP = 10/3.6

# Planning horizon for the TP query sent by BP [sec]
# Used for grid search in the [T_MIN, T_MAX] range with resolution of T_RES
BP_ACTION_T_LIMITS = np.array([2.0, 20.0])

# Behavioral planner action-specification weights for longitudinal jerk vs lateral jerk vs time of action
BP_JERK_S_JERK_D_TIME_WEIGHTS = np.array([
    [12, 0.15, 0.1],
    [2, 0.15, 0.1],
    [0.01, 0.15, 0.1]
])

# Longitudinal Acceleration Limits [m/sec^2]
LON_ACC_LIMITS = np.array([-4.0, 3.0])  # taken from SuperCruise presentation

# Latitudinal Acceleration Limits [m/sec^2]
LAT_ACC_LIMITS = np.array([-4.0, 4.0])

# Assumed response delay on road [sec]
# Used to compute safe distance from other agents on road
SPECIFICATION_MARGIN_TIME_DELAY = 2
SAFETY_MARGIN_TIME_DELAY = 1


# [m/sec] Minimal difference of velocities to justify an overtake
MIN_OVERTAKE_VEL = 3.5

# [m] The margin that we take from the front/read of the vehicle to define the front/rear partitions
LON_MARGIN_FROM_EGO = 1

# Uniform grids for BP Filters
FILTER_A_0_GRID = UniformGrid(LON_ACC_LIMITS, 0.5)
FILTER_V_0_GRID = UniformGrid(np.array([0.0, 34]), 0.5)  # [m/sec] # TODO: use VELOCITY_LIMITS?
FILTER_V_T_GRID = UniformGrid(np.array([0.0, 34]), 0.5)  # [m/sec] # TODO: use VELOCITY_LIMITS?
FILTER_S_T_GRID = UniformGrid(np.array([-10, 110]), 1)  # TODO: use BEHAVIORAL_PLANNING_LOOKAHEAD_DIST?

# Trajectory Planner #

# [m] Resolution for the interpolation of the reference route
TRAJECTORY_ARCLEN_RESOLUTION = 0.5

# [seconds] Resolution for the visualization of predicted dynamic objects
VISUALIZATION_PREDICTION_RESOLUTION = 1.0

# Curve interpolation type (order)
TRAJECTORY_CURVE_SPLINE_FIT_ORDER = 4

# Desired resolution of map [m], used for accuracy test on Frenet-Serret transformations
ROAD_MAP_REQUIRED_RES = 5

# [m] Cost function clips higher distances before exponentiation
EXP_CLIP_TH = 50.0

# Number of (best) trajectories to publish to visualization
MAX_VIS_TRAJECTORIES_NUMBER = 64

# Number of points in trajectories for sending out to visualization (currently VizTool freezes when there are too much)
MAX_NUM_POINTS_FOR_VIZ = 60

# [m] "Negligible distance" threshold between the desired location and the actual location between two TP planning
# iterations. If the distance is lower than this threshold, the TP plans the trajectory as is the ego vehicle is
# currently in the desired location and not in its actual location.
NEGLIGIBLE_DISPOSITION_LON = 1.5  # longitudinal (ego's heading direction) difference threshold
NEGLIGIBLE_DISPOSITION_LAT = 0.5  # lateral (ego's side direction) difference threshold

# [sec] Time-Resolution for the trajectory's discrete points that are sent to the controller
TRAJECTORY_TIME_RESOLUTION = 0.1

# Number of trajectory points to send out (to controller) from the TP - including the current state of ego
TRAJECTORY_NUM_POINTS = 10

# Waypoints requirements from IDL
TRAJECTORY_WAYPOINT_SIZE = 11
MAX_TRAJECTORY_WAYPOINTS = 100

# Werling Planner #

# [sec] Time-Resolution for planning
WERLING_TIME_RESOLUTION = 0.1

# [m] Range for grid search in werling planner (long. position)
SX_OFFSET_MIN, SX_OFFSET_MAX = -8, 0

# [m] Range for grid search in werling planner (long. velocity)
SV_OFFSET_MIN, SV_OFFSET_MAX = 0, 0

# [m] Range for grid search in werling planner (lat. position)
DX_OFFSET_MIN, DX_OFFSET_MAX = -1.6, 1.6

# Linspace number of steps in the constraints parameters grid-search
SX_STEPS, SV_STEPS, DX_STEPS = 5, 1, 9

# Linspace number of steps in latitudinal horizon planning time (from Td_low_bound to Ts)
TD_STEPS = 6

# Minimal T_d (time-horizon for the lateral movement) - in units of WerlingPlanner.dt
TD_MIN_DT = 3

# Frenet-Serret Conversions #

# [1/m] Curvature threshold for the GD step (if value is smaller than this value, there is no step executed)
TINY_CURVATURE = 10e-5

# [m/sec^2] when acceleration is not specified - TP uses this as goal acceleration
DEFAULT_ACCELERATION = 0.0

# [-+1/m] when curvature is not specified - TP uses this as goal curvature
DEFAULT_CURVATURE = 0.0

# FixedTrajectoryPlanner.plan performs sleep with time = mean + max(0, N(0, std))
FIXED_TRAJECTORY_PLANNER_SLEEP_MEAN = 0.15
FIXED_TRAJECTORY_PLANNER_SLEEP_STD = 0.2


# State #

# TODO: set real values
# [m] Bounding box size around ego vehicle
EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT = 5.0, 2.0, 2.0

# The id of the ego object
EGO_ID = 0

# [m] Default height for objects - State Module
DEFAULT_OBJECT_Z_VALUE = 0.

# Cutoff thershold to avoid negative velocities. Hack.
VELOCITY_MINIMAL_THRESHOLD = 0.001

# Whether we filter out dynamic objects that are not on the road
# Request by perception for viewing recordings in non-mapped areas.
# Since SP is assumed to handle filtering out off-road objects, this is currently in use only in prediction.
FILTER_OFF_ROAD_OBJECTS = False

### DM Manager configuration ###
BEHAVIORAL_PLANNING_MODULE_PERIOD = 0.5
TRAJECTORY_PLANNING_MODULE_PERIOD = 0.2

#### NAMES OF MODULES FOR LOGGING ####
DM_MANAGER_NAME_FOR_LOGGING = "DM Manager"
NAVIGATION_PLANNING_NAME_FOR_LOGGING = "Navigation Planning"
BEHAVIORAL_PLANNING_NAME_FOR_LOGGING = "Behavioral Planning"
BEHAVIORAL_PLANNING_NAME_FOR_METRICS = "BP"
TRAJECTORY_PLANNING_NAME_FOR_LOGGING = "Trajectory Planning"
TRAJECTORY_PLANNING_NAME_FOR_METRICS = "TP"
STATE_MODULE_NAME_FOR_LOGGING = "State Module"

#### MetricLogger
METRIC_LOGGER_DELIMITER = '_'

##### Log messages
# TODO: update decision_making_sim messages
LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS = "Received mission params"
LOG_MSG_SCENE_STATIC_RECEIVED = "Received SceneStatic message with Timestamp: "
LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG = "Publishing Trajectory"
LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT = "BehavioralPlanningFacade output is"
LOG_MSG_BEHAVIORAL_PLANNER_SEMANTIC_ACTION = "Chosen behavioral semantic action is"
LOG_MSG_BEHAVIORAL_PLANNER_ACTION_SPEC = "Chosen action specification is"
LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES = "TP has found %d valid trajectories to choose from"
LOG_MSG_RECEIVED_STATE = "Received state"
LOG_MSG_STATE_MODULE_PUBLISH_STATE = "Publishing State"
LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME = "TrajectoryPlanningFacade._periodic_action_impl time"
LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME = "BehavioralFacade._periodic_action_impl time"
LOG_INVALID_TRAJECTORY_SAMPLING_TIME = "LocalizationUtils.is_actual_state_close_to_expected_state timestamp to sample is " \
                                       "%f while trajectory time range is [%f, %f]"
LOG_MSG_TRAJECTORY_PLAN_FROM_DESIRED = "TrajectoryPlanningFacade planning from desired location (desired frenet: %s, actual frenet: %s)"
LOG_MSG_TRAJECTORY_PLAN_FROM_ACTUAL = "TrajectoryPlanningFacade planning from actual location (actual frenet: %s)"

# Resolution of car timestamps in sec
TIMESTAMP_RESOLUTION_IN_SEC = 1e-9
