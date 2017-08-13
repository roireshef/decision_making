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

### DM Manager configuration ###
BEHAVIORAL_PLANNING_MODULE_PERIOD = 1
TRAJECTORY_PLANNING_MODULE_PERIOD = 0.1

### DDS Constants ###
STATE_MODULE_DDS_PARTICIPANT = "DecisionMakingParticipantLibrary::StateModule"
BEHAVIORAL_PLANNER_DDS_PARTICIPANT = "DecisionMakingParticipantLibrary::BehavioralPlanner"
TRAJECTORY_PLANNER_DDS_PARTICIPANT = "DecisionMakingParticipantLibrary::TrajectoryPlanner"

DECISION_MAKING_DDS_FILE = "decisionMakingMain.xml"

DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC = "StateSubscriber::DynamicObjectsReader"
SELF_LOCALIZATION_SUBSCRIBE_TOPIC = "StateSubscriber::SelfLocalizationReader"
OCCUPANCY_STATE_SUBSCRIBE_TOPIC = "StateSubscriber::OccupancyStateReader"

#### NAMES OF MODULES FOR LOGGING ####
DM_MANAGER_NAME_FOR_LOGGING = "DM Manager"
BEHAVIORAL_PLANNING_NAME_FOR_LOGGING = "Behavioral Planning"
TRAJECTORY_PLANNING_NAME_FOR_LOGGING = "Trajectory Planning"
STATE_MODULE_NAME_FOR_LOGGING = "State Module"

