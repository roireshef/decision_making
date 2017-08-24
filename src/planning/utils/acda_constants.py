#### Constants for safety metric and ACDA

### ROAD, GRAVITY, FRICTION
G = 9.8  # acceleration of gravity in meter/second^2
# sin of the angle of inclination of the road's slope (unitless, between 0.0 and 1.0).
# For a level road this value is zero. Road is assumed flat.
SIN_ROAD_INCLINE = 0.0
MU = 0.7  # the friction coefficient (unitless), a function of the tire type and road conditions

### behavior defining constants
TIME_GAP = 2.0  # this is the X (usually, 2) second rule. Unit is seconds.
TPRT = 0.5  # for our agent, it is the perception-reaction time in seconds. For human it is 0.72s.
LATERAL_MARGIN_FROM_OBJECTS = 0.5 # distance in [m] from objects after which they are considered not in our way

### car dimentions
CAR_DILATION_LENGTH = 2  # TODO used by acda as the length of the car. True for simulation needs to change for real car.
# sensors assumed to be in the middle of the car. TODO update for real car
SENSOR_OFFSET_FROM_FRONT = CAR_DILATION_LENGTH / 2.0
CAR_DILATION_WIDTH = 1.5  # # TODO used by acda as the width of the car. update for real car.

### sensors and lookahead ranges for ACDA
FORWARD_LOS_MAX_RANGE = 100.0  # TODO - this is our sensor range (forward). update for real car/sensors
HORIZONTAL_LOS_MAX_RANGE = 100.0  # TODO - this is our sensor range (horizontal). update for real car/sensors
LARGEST_CURVE_RADIUS = 1000.0
MODERATE_DECELERATION = 2.0
BEHAVIORAL_PLANNING_LOOKAHEAD_DISTANCE = 50.0   # this affects objects used for horizLOS. TODO update for current setup
TRAJECTORY_PLANNING_LOOKAHEAD_DISTANCE = 27.0   # this affects objects used for horizLOS. TODO update for current setup
SAFETY_MIN_LOOKAHEAD_DIST = 15.0  # we consider at least all objects with range < this constant
HIDDEN_PEDESTRIAN_VEL = 1.5  # in meter/sec. Used for horizontal LOS calculation