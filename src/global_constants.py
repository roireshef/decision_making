import numpy as np

from decision_making.src.planning.utils.numpy_utils import UniformGrid

# General constants
EPS = np.finfo(np.float32).eps
MPH_TO_MPS = 2.23694

# Behavioral Planner

# [m] high-level behavioral planner lookahead distance
PLANNING_LOOKAHEAD_DIST = 150.0

# [m] Maximum forward horizon for building Generalized Frenet Frames
MAX_FORWARD_HORIZON = 600.0

# [m] Maximum backward horizon for building Generalized Frenet Frames
MAX_BACKWARD_HORIZON = 100.0

# [m] distance to the end of a partial GFF at which the vehicle must not be in
PARTIAL_GFF_END_PADDING = 5.0

# The necessary lateral margin in [m] that needs to be taken in order to assume that it is not in car's way
LATERAL_SAFETY_MARGIN_FROM_OBJECT = 0.0

# Prefer left or right split when the costs are the same
PREFER_LEFT_SPLIT_OVER_RIGHT_SPLIT = False

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

LARGE_DISTANCE_FROM_SHOULDER = 1e8          # a large value indicating being very far from road shoulders (so we don't
                                            # penalize on that).

LON_JERK_COST_WEIGHT = 1.0                  # cost of longitudinal jerk
LAT_JERK_COST_WEIGHT = 1.0                  # cost of lateral jerk

# [m/sec] speed to plan towards by default in BP
# original velocities in [mph] are converted into [m/s]
BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED = 85/MPH_TO_MPS # TODO - get this value from the map

# [m/sec] the addition to BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED for TP
# we allow higher desired velocity in TP than in BP because TP & BP are not synchronized
TP_DESIRED_VELOCITY_DEVIATION = 1

# [m/s] min & max velocity limits are additional parameters for TP and for Static Recipe enumeration
# original velocities in [mph] are converted into [m/s]
VELOCITY_LIMITS = np.array([0.0, 90/MPH_TO_MPS])
VELOCITY_STEP = 5/MPH_TO_MPS

# Planning horizon for the TP query sent by BP [sec]
# Used for grid search in the [T_MIN, T_MAX] range with resolution of T_RES
BP_ACTION_T_LIMITS = np.array([0.0, 20.0])

# Behavioral planner action-specification weights for longitudinal jerk vs lateral jerk vs time of action
# For lane change to last 6 seconds, lateral w_J = 1.296 / lw^2, where lw is lane width. For lw=3.6, w_J=0.1.
BP_JERK_S_JERK_D_TIME_WEIGHTS = np.array([
    [1, 0.1, 0.1],
    [0.2, 0.1, 0.1],
    [0.015, 0.1, 0.1]
])

# Longitudinal Acceleration Limits [m/sec^2]
LON_ACC_LIMITS = np.array([-5.5, 3.0])  # taken from SuperCruise presentation

# Lateral Acceleration Limits [m/sec^2]
# TODO: deprecated - should be replaced or removed as it is replaced by LAT_ACC_LIMITS_BY_K
LAT_ACC_LIMITS = np.array([-2, 2])

# Latitudinal Acceleration Limits [m/sec^2] by radius. This table represents a piecewise linear function that maps
# radius of turn to the corresponding lateral acceleration limit to test against. Each tuple in the table consists of
# the following ("from" radius, "to" radius, ...) to represent the radius range,
# and (..., "from" acceleration limit, "to" acceleration limit). Within every range we extrapolate linearly between the
# "from"/"to" acceleration limits.
# NOTE: it's important to have a slope of 0 for the last tuple if it includes np.inf !!
LAT_ACC_LIMITS_BY_K = np.array([(0, 6, 3, 3),
                                (6, 25, 3, 2.8),
                                (25, 50, 2.8, 2.55),
                                (50, 75, 2.55, 2.4),
                                (75, 100, 2.4, 2.32),
                                (100, 150, 2.32, 2.2),
                                (150, 200, 2.2, 2.1),
                                (200, 275, 2.1, 2),
                                (275, np.inf, 2, 2)])

# [m/sec^2] lateral acceleration limits during lane change
LAT_ACC_LIMITS_LANE_CHANGE = np.array([-3, 3])

# Relative Latitudinal Acceleration Limits [m/sec^2] for lane change
REL_LAT_ACC_LIMITS = np.array([-0.6, 0.6])

# BP has more strict lateral acceleration limits than TP. BP_LAT_ACC_STRICT_COEF is the ratio between BP and TP limits
BP_LAT_ACC_STRICT_COEF = 1.0
TP_LAT_ACC_STRICT_COEF = 1.1

# Headway [sec] from a leading vehicle, used for specification target and safety checks accordingly
SPECIFICATION_HEADWAY = 1.5
SAFETY_HEADWAY = 0.7  # Should correspond to assumed delay in response (end-to-end)

# Additional margin to keep from leading vehicle, in addition to the headway, used for specification target and
# safety checks accordingly
LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT = 5.0
LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT = 3.0
LONGITUDINAL_SPECIFY_MARGIN_FROM_STOP_BAR = 1.0

# Additional distance after stop bar, where vehicle still considers bar as active, until DIM is active
DIM_MARGIN_TO_STOP_BAR = 10.0

# Deceleration thresholds as defined by the system requirements
SPEED_THRESHOLDS = np.array([3, 6, 9, 12, 14, 100])  # in [m/s]
TIME_THRESHOLDS = np.array([7, 8, 10, 13, 15, 19.8])  # in [s]

# [m/sec] Minimal difference of velocities to justify an overtake
MIN_OVERTAKE_VEL = 3.5

# [m/sec] zero speed
ZERO_SPEED = 0.0

# [m] road sign length
ROAD_SIGN_LENGTH = 0

# define acceptable distance [m] between stop_bar and stop_sign to be considered as related
CLOSE_ENOUGH = 3.0

# [m] The margin that we take from the front/read of the vehicle to define the front/rear partitions
LON_MARGIN_FROM_EGO = 1

# Uniform grids for BP Filters
FILTER_A_0_GRID = UniformGrid(LON_ACC_LIMITS, 0.5)
FILTER_V_0_GRID = UniformGrid(VELOCITY_LIMITS, 0.5)  # [m/sec]
FILTER_V_T_GRID = UniformGrid(VELOCITY_LIMITS, 0.5)  # [m/sec]
FILTER_S_T_GRID = UniformGrid(np.array([-10, 110]), 1)  # TODO: use BEHAVIORAL_PLANNING_LOOKAHEAD_DIST?

# Step size for indexes in beyond spec filters
BEYOND_SPEC_INDEX_STEP = 4

# Min distance threshold ahead to raise takeover flag
MIN_DISTANCE_TO_SET_TAKEOVER_FLAG = 80
# Time threshold to raise takeover flag
TIME_THRESHOLD_TO_SET_TAKEOVER_FLAG = 12

# Used by TargetActionSpace.modify_target_speed_if_ego_is_faster_than_target() to calculate the speed reduction of the target for the action spec
SLOW_DOWN_FACTOR = 0.5
# Used by TargetActionSpace.modify_target_speed_if_ego_is_faster_than_target() to calculate the lower bound on the speed
# reduction of the target for the action spec. Set to a lower value to account for the fact that deceleration is not immediate
MAX_IMMEDIATE_DECEL = - LON_ACC_LIMITS[0] - 1
# Headway to select calm/aggressive dynamic action. Must be larger than SAFETY_HEADWAY
REQUIRED_HEADWAY_FOR_CALM_DYNAMIC_ACTION = 1.4
REQUIRED_HEADWAY_FOR_STANDARD_DYNAMIC_ACTION = 1.2

SPEEDING_VIOLATION_TIME_TH = 3.0  # in [seconds]. Speeding violation allowed time from START of action.
SPEEDING_SPEED_TH = 2.0 / 3.6  # in [m/s]. Allowed magnitude of speeding violation.

# [sec], Time that has to pass after the turn signal is turned on before considering a lane change
LANE_CHANGE_DELAY = 1.0

# [sec], Time completion target for lane changes
LANE_CHANGE_TIME_COMPLETION_TARGET = 6.0

# [sec], Minimum time allowed for lane change actions. This time will override the lane change specification time towards the end of the
# lane change.
MIN_LANE_CHANGE_ACTION_TIME = 0.2

# [m], Maximum distance from lane center to consider a lane change complete
MAX_OFFSET_FOR_LANE_CHANGE_COMPLETE = 0.35

# [rad], Maximum relative heading to consider a lane change complete
MAX_REL_HEADING_FOR_LANE_CHANGE_COMPLETE = 0.25

# [%], Threshold at which a lane change will be aborted if the maneuver completion percentage is under this value
LANE_CHANGE_ABORT_THRESHOLD = 20.0

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
# iterations. If the distance is lower than this threshold, the TP plans the trajectory as if the ego vehicle is
# currently in the desired location and not in its actual location.
NEGLIGIBLE_DISPOSITION_LON = 1.5  # longitudinal (ego's heading direction) difference threshold
NEGLIGIBLE_DISPOSITION_LAT = 0.5  # lateral (ego's side direction) difference threshold

# limits for allowing tracking mode. During tracking we maintain a fixed speed trajectory with the speed the target.
# May want to consider replacing with ego speed, so that speed is constant
TRACKING_DISTANCE_DISPOSITION_LIMIT = 0.1       # in [m]
TRACKING_VELOCITY_DISPOSITION_LIMIT = 0.1       # in [m/s]
TRACKING_ACCELERATION_DISPOSITION_LIMIT = 0.05  # in [m/s^2]

# [sec] Time-Resolution for the trajectory's discrete points that are sent to the controller
TRAJECTORY_TIME_RESOLUTION = 0.1

# Number of trajectory points to send out (to controller) from the TP - including the current state of ego
# TODO: check safety in BP along the whole padded trajectory (see MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON)
TRAJECTORY_NUM_POINTS = 50

# Waypoints requirements from IDL
TRAJECTORY_WAYPOINT_SIZE = 11
MAX_TRAJECTORY_WAYPOINTS = 200

# [sec] Minimum required time horizon for trajectory (including padding)
# TODO: make it consistent with TRAJECTORY_NUM_POINTS
MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON = 3.

# TODO: set real values from map / perception
# Road shoulders width in [m]
ROAD_SHOULDERS_WIDTH = 1.5

# Amount of error in fitting points to map curve, smaller means using more spline polynomials for the fit (and smaller
# error). This factor is the maximum mean square error (per point) allowed. For example, 0.0001 mean that the
# max. standard deviation is 1 [cm] so the max. squared standard deviation is 10e-4.
SPLINE_POINT_DEVIATION = 0.0001

# [m] occupancy grid resolution for encoding lane merge state
LANE_MERGE_STATE_OCCUPANCY_GRID_RESOLUTION = 4.5
# [m] the horizon from ego in each side of the main road in occupancy grid of lane merge state
LANE_MERGE_STATE_OCCUPANCY_GRID_ONESIDED_LENGTH = 150
# [m] maximum forward horizon from a lane merge on the ego road for engaging the lane-merge policy
LANE_MERGE_STATE_FAR_AWAY_DISTANCE = 300
# [m/sec] maximal velocity of actors and in action space
LANE_MERGE_ACTION_SPACE_MAX_VELOCITY = 25
# [m/sec] velocity of empty cells on actors grid
LANE_MERGE_ACTION_SPACE_EMPTY_CELL_VELOCITY = 25
# [m/sec] velocity resolution in action space
LANE_MERGE_ACTION_SPACE_VELOCITY_RESOLUTION = 5

# [m/sec] maximal velocity from which DIM may be performed
DRIVER_INITIATED_MOTION_VELOCITY_LIMIT = 0.1
# [sec] maximal time to reach the next stop bar (keeping current ego velocity)
DRIVER_INITIATED_MOTION_MAX_TIME_TO_STOP_BAR = 5
# [m] how far to look for the next stop bar to perform DIM
DRIVER_INITIATED_MOTION_STOP_BAR_HORIZON = 5
# acceleration pedal strength in [0..1]
DRIVER_INITIATED_MOTION_PEDAL_THRESH = 0.05
# [sec] time period of sufficient throttle pedal
DRIVER_INITIATED_MOTION_PEDAL_TIME = 0.5
# [sec] time period DIM is active after driver released the pedal
DRIVER_INITIATED_MOTION_TIMEOUT = 10
# indices of tuple
STOP_BAR_IND, STOP_BAR_DISTANCE_IND = 0, 1

# Werling Planner #

# [sec] Time-Resolution for planning
WERLING_TIME_RESOLUTION = 0.1

# [m] Range for grid search in werling planner (long. position)
SX_OFFSET_MIN, SX_OFFSET_MAX = 0, 0

# [m] Range for grid search in werling planner (long. velocity)
SV_OFFSET_MIN, SV_OFFSET_MAX = 0, 0

# [m] Range for grid search in werling planner (lat. position)
DX_OFFSET_MIN, DX_OFFSET_MAX = 0, 0

# Linspace number of steps in the constraints parameters grid-search
SX_STEPS, SV_STEPS, DX_STEPS = 1, 1, 1

# Linspace number of steps in latitudinal horizon planning time (from Td_low_bound to Ts)
TD_STEPS = 1

# Minimal T_d (time-horizon for the lateral movement) - in units of WerlingPlanner.dt
TD_MIN_DT = 3

# negative close to zero trajectory velocity, which may be replaced by zero velocity
CLOSE_TO_ZERO_NEGATIVE_VELOCITY = -0.1

# close to zero velocity, which may be considered as zero velocity (used by frenet->cartesian conversion)
# Don't decrease the value 0.01, since otherwise frenet->cartesian conversion creates state with too high curvature.
NEGLIGIBLE_VELOCITY = 0.01

# Frenet-Serret Conversions #

# [1/m] Curvature threshold for the GD step (if value is smaller than this value, there is no step executed)
TINY_CURVATURE = 1e-4

# [1/m] maximal trajectory curvature, based on the minimal turning radius, which is defined in a basic car's spec
# A typical turning radius = 5 m, then MAX_CURVATURE = 0.2.
MAX_CURVATURE = 0.2

# [m/sec^2] when acceleration is not specified - TP uses this as goal acceleration
DEFAULT_ACCELERATION = 0.0

# [-+1/m] when curvature is not specified - TP uses this as goal curvature
DEFAULT_CURVATURE = 0.0

# FixedTrajectoryPlanner.plan performs sleep with time = mean + max(0, N(0, std))
FIXED_TRAJECTORY_PLANNER_SLEEP_MEAN = 0.15
FIXED_TRAJECTORY_PLANNER_SLEEP_STD = 0.2

# Route Planner #
LANE_ATTRIBUTE_CONFIDENCE_THRESHOLD = 0.7

# Indices for the route plan cost tuple
LANE_OCCUPANCY_COST_IND = 0
LANE_END_COST_IND = 1

# Maximum value for RP's lane end and occupancy costs
# Lane end cost = MAX_COST --> Do not leave the road segment in this lane
# Lane occupancy cost = MAX_COST --> Do not drive in this lane
MAX_COST = 1.0

# Minimum value for RP's lane end and occupancy costs
# Lane end cost = MIN_COST --> There is no penalty for being in this lane as we transition to the next road segment
# Lane occupancy cost = MIN_COST --> We are allowed to drive in this lane
MIN_COST = 0.0

# Tunable parameter that defines the cost at which a lane is no longer considered to be valid. For example, a lane may
# have an end or occupancy cost equal to 0.99, and it may be desirable to not consider it as a valid lane. This is
# different from the actual maximum cost (= MAX_COST).
SATURATED_COST = 1.0

# Discount factor used to limit the effect of backpropagating downstream lane end costs
BACKPROP_DISCOUNT_FACTOR = 0.75

# Threshold at which a backpropagated lane end cost will just be set equal to MIN_COST
BACKPROP_COST_THRESHOLD = 0.001

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
BEHAVIORAL_PLANNING_MODULE_PERIOD = 0.3
TRAJECTORY_PLANNING_MODULE_PERIOD = 0.1
ROUTE_PLANNING_MODULE_PERIOD = 1

#### NAMES OF MODULES FOR LOGGING ####
DM_MANAGER_NAME_FOR_LOGGING = "DM Manager"
BEHAVIORAL_PLANNING_NAME_FOR_LOGGING = "Behavioral Planning"
BEHAVIORAL_PLANNING_NAME_FOR_METRICS = "BP"
TRAJECTORY_PLANNING_NAME_FOR_LOGGING = "Trajectory Planning"
TRAJECTORY_PLANNING_NAME_FOR_METRICS = "TP"
ROUTE_PLANNING_NAME_FOR_LOGGING = "Route Planning"
ROUTE_PLANNING_NAME_FOR_METRICS = "RP"

#### MetricLogger
METRIC_LOGGER_DELIMITER = '_'

##### Log messages
# TODO: update decision_making_sim messages
LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS = "Received mission params"
LOG_MSG_SCENE_STATIC_RECEIVED = "Received SceneStatic message with Timestamp: "
LOG_MSG_SCENE_DYNAMIC_RECEIVED = "Received SceneDynamic message with Timestamp: "
LOG_MSG_CONTROL_STATUS = "Received ControlStatus message with Timestamp: "
LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG = "Publishing Trajectory"
LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT = "BehavioralPlanningFacade output is"
LOG_MSG_ROUTE_PLANNER_OUTPUT = "RoutePlanningFacade output is"
LOG_MSG_BEHAVIORAL_PLANNER_SEMANTIC_ACTION = "Chosen behavioral semantic action is"
LOG_MSG_BEHAVIORAL_PLANNER_ACTION_SPEC = "Chosen action specification is"
LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES = "TP has found %d valid trajectories to choose from"
LOG_MSG_RECEIVED_STATE = "Received state"
LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME = "TrajectoryPlanningFacade._periodic_action_impl time"
LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME = "BehavioralFacade._periodic_action_impl time"
LOG_MSG_ROUTE_PLANNER_IMPL_TIME = "ROUTE Facade._periodic_action_impl time"
LOG_INVALID_TRAJECTORY_SAMPLING_TIME = "LocalizationUtils.is_actual_state_close_to_expected_state timestamp to sample is " \
                                       "%f while trajectory time range is [%f, %f]"
LOG_MSG_TRAJECTORY_PLAN_FROM_DESIRED = "TrajectoryPlanningFacade planning from desired location (desired frenet: %s, actual frenet: %s)"
LOG_MSG_TRAJECTORY_PLAN_FROM_ACTUAL = "TrajectoryPlanningFacade planning from actual location (actual frenet: %s)"
LOG_MSG_BEHAVIORAL_GRID = "BehavioralGrid"
LOG_MSG_PROFILER_PREFIX = "DMProfiler Stats: "

# Resolution of car timestamps in sec
TIMESTAMP_RESOLUTION_IN_SEC = 1e-9

PG_SPLIT_PICKLE_FILE_NAME = 'PG_split.pkl'
PG_PICKLE_FILE_NAME = 'PG.pkl'
ACCEL_TOWARDS_VEHICLE_SCENE_STATIC_PICKLE_FILE_NAME = 'accel_scene_static.pkl'
ACCEL_TOWARDS_VEHICLE_SCENE_DYNAMIC_PICKLE_FILE_NAME = 'accel_scene_dynamic.pkl'
OVAL_WITH_SPLITS_FILE_NAME = 'oval_with_splits.pkl'
OBJ_INTRUDING_IN_LANE_SCENE_STATIC_PICKLE_FILE_NAME = 'obj_corner_intruding_scene_static.pkl'
OBJ_INTRUDING_IN_LANE_SCENE_DYNAMIC_PICKLE_FILE_NAME = 'obj_corner_intruding_scene_dynamic.pkl'
