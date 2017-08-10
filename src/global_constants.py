DIVISION_FLOATING_ACCURACY = 10 ** -10

# Trajectory Planner #

# [m] Resolution for the interpolation of the reference route
TRAJECTORY_ARCLEN_RESOLUTION = 0.1

# Curve interpolation type (order)
TRAJECTORY_CURVE_INTERP_TYPE = 'cubic'

# [m] Do not consider obastacles that are distant than this thershold
MAXIMAL_OBSTACLE_PROXIMITY = 25

# [m] Cost function clips higher distances before exponentiation
EXP_CLIP_TH = 50

# Number of (best) trajectories to publish to visualization
NUM_ALTERNATIVE_TRAJECTORIES = 100

# Werling Planner #

# [sec] Time-Resolution for planning
WERLING_TIME_RESOLUTION = 0.1

# [m] Range for grid search in werling planner (long. position)
SX_OFFSET_MIN, SX_OFFSET_MAX = -3, 0

# [m] Range for grid search in werling planner (long. velocity)
SV_OFFSET_MIN, SV_OFFSET_MAX = -5, 5

# [m] Range for grid search in werling planner (lat. position)
DX_OFFSET_MIN, DX_OFFSET_MAX = -1, 1

# Linspace number of steps in the constraints parameters grid-search
SX_STEPS, SV_STEPS, DX_STEPS = 5, 10, 5

#### Constants for safety metric and ACDA
G = 9.8  # acceleration of gravity in meter/second^2
# sin of the angle of inclination of the road's slope (unitless, between 0.0 and 1.0).
# For a level road this value is zero. Road is assumed flat.
SIN_ROAD_INCLINE = 0.0
CAR_DILATION_LENGTH = 2  # TODO used by acda as the length of the car. True for simulation needs to change for real car.
# sensors assumed to be in the middle of the car. TODO update for real car
SENSOR_OFFSET_FROM_FRONT = CAR_DILATION_LENGTH / 2.0
CAR_DILATION_WIDTH = 1.5  # # TODO used by acda as the width of the car. update for real car.
FORWARD_LOS_MAX_RANGE = 100.0  # TODO - this is our sensor range (forward). update for real car/sensors
HORIZONTAL_LOS_MAX_RANGE = 100.0  # TODO - this is our sensor range (horizontal). update for real car/sensors
LARGEST_CURVE_RADIUS = 1000.0
MODERATE_DECELERATION = 2.0
BEHAVIROAL_PLANNING_LOOKAHEAD_DISTANCE = 40.0   # this affects objects used for horizLOS. TODO update for current setup
TRAJECTORY_PLANNING_LOOKAHEAD_DISTANCE = 27.0   # this affects objects used for horizLOS. TODO update for current setup
SAFETY_MIN_LOOKAHEAD_DIST = 15.0  # we consider at least all objects with range < this constant
HIDDEN_PEDESTRIAN_VEL = 1.5  # in meter/sec. Used for horizontal LOS calculation

#### NAMES OF MODULES FOR LOGGING ####
BEHAVIORAL_PLANNING_NAME_FOR_LOGGING = "Behavioral Planning"
