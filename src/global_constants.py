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


#### NAMES OF MODULES FOR LOGGING ####
BEHAVIORAL_PLANNING_NAME_FOR_LOGGING = "Behavioral Planning"
ACDA_NAME_FOR_LOGGING = "ACDA Module"
