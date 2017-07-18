DIVISION_FLOATING_ACCURACY = 10 ** -10

#### Trajectory Planner ####

# [m] Trajectory Planner - Resolution for the interpolation of the reference route
TRAJECTORY_ARCLEN_RESOLUTION = 0.1

# Trajectory Planner - Curve interpolation type (order)
TRAJECTORY_CURVE_INTERP_TYPE = 'cubic'

# [m] Trajectory Planner - Do not consider obastacles that are distant than this thershold
MAXIMAL_OBSTACLE_PROXIMITY = 25

# [m] Trajectory Planner - Cost function clips higher distances before exponentiation
EXP_CLIP_TH = 50

# Trajectory Planner - Number of (best) trajectories to publish to visualization
NUM_ALTERNATIVE_TRAJECTORIES = 10

#### Werling Planner ####

# [sec] Werling Planner - Time-Resolution for planning
WERLING_TIME_RESOLUTION = 0.1

# [m] Werling Planner - Range for grid search in werling planner (long. position)
SX_OFFSET_MIN, SX_OFFSET_MAX = -3, 0
# [m] Werling Planner - Range for grid search in werling planner (long. velocity)
SV_OFFSET_MIN, SV_OFFSET_MAX = -10, 10
# [m] Werling Planner - Range for grid search in werling planner (lat. position)
DX_OFFSET_MIN, DX_OFFSET_MAX = -2, 2
# [m] Werling Planner - Range for grid search in werling planner (lat. position)
SX_RES, SV_RES, DX_RES = 10, 5, 10

