from typing import Tuple, Dict, List

from decision_making.src.planning.behavioral.policies.november_demo_semantic_policy import NovDemoPolicy, \
    NovDemoBehavioralState
from decision_making.src.planning.behavioral.policy import PolicyConfig
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticAction, SEMANTIC_CELL_LANE, \
    SEMANTIC_CELL_LON, SemanticActionType, SemanticActionSpec
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, EgoState, DynamicObject, ObjectSize, OccupancyState
from decision_making.src.state.state_module import StateModule
from mapping.test.model.testable_map_fixtures import *
from logging import Logger


def test_novDemoEvalSemanticActions(testable_map_api):
    max_velocity = 12  # m/s

    ego_v = 10
    pos1 = np.array([7, -3])
    pos2 = np.array([11, 0])
    pos3 = np.array([8, 3])
    vel1 = 12
    vel2 = 6
    vel3 = 9

    road_localization1 = StateModule._compute_road_localization(pos1, 0, testable_map_api)
    road_localization2 = StateModule._compute_road_localization(pos2, 0, testable_map_api)
    road_localization3 = StateModule._compute_road_localization(pos3, 0, testable_map_api)
    size = ObjectSize(1.5, 0.5, 0)
    obs = list([
        DynamicObject(obj_id=0, timestamp=0, x=pos1[0], y=pos1[1], z=0, yaw=0, size=size,
                      road_localization=road_localization1, confidence=1.0, v_x=vel1, v_y=0, acceleration_lon=0.0,
                      omega_yaw=0.0),
        DynamicObject(obj_id=1, timestamp=0, x=pos2[0], y=pos2[1], z=0, yaw=0, size=size,
                      road_localization=road_localization2, confidence=1.0, v_x=vel2, v_y=0, acceleration_lon=0.0,
                      omega_yaw=0.0),
        DynamicObject(obj_id=2, timestamp=0, x=pos3[0], y=pos3[1], z=0, yaw=0, size=size,
                      road_localization=road_localization3, confidence=1.0, v_x=vel3, v_y=0, acceleration_lon=0.0,
                      omega_yaw=0.0)
    ])

    ego = EgoState(obj_id=-1, timestamp=0, x=0, y=0, z=0, yaw=0, size=size,
                   road_localization=StateModule._compute_road_localization(np.array([0, 0]), 0.0, testable_map_api),
                   confidence=1.0, v_x=ego_v, v_y=0, steering_angle=0.0, acceleration_lon=0.0, omega_yaw=0.0)

    # state = State(occupancy_state=OccupancyState(0, np.array([]), np.array([])), dynamic_objects=obs, ego_state=ego)
    # predictor = Predictor(testable_map_api)
    # policy = NovDemoPolicy(Logger("NovDemoTest"), PolicyConfig(), behav_state, predictor, testable_map_api,
    #                        max_velocity)

    grid = {}
    grid[(-1, -1)] = []
    grid[(0, -1)] = []
    grid[(1, -1)] = []
    grid[(-1, 0)] = []
    grid[(0, 0)] = []
    grid[(1, 0)] = []
    grid[(-1, 1)] = [obs[0]]
    grid[(0, 1)] = [obs[1]]
    grid[(1, 1)] = [obs[2]]
    behav_state = NovDemoBehavioralState(grid, ego)

    semantic_actions = []
    semantic_actions.append(SemanticAction((-1, 1), obs[0], SemanticActionType.FOLLOW))
    semantic_actions.append(SemanticAction((0, 1), obs[1], SemanticActionType.FOLLOW))
    semantic_actions.append(SemanticAction((1, 1), obs[2], SemanticActionType.FOLLOW))
    actions_spec = []
    actions_spec.append(SemanticActionSpec(t=5, v=vel1, s_rel=obs[0].x-ego.x, d_rel=obs[0].y-ego.y))
    actions_spec.append(SemanticActionSpec(t=7, v=vel2, s_rel=obs[1].x-ego.x, d_rel=obs[1].y-ego.y))
    actions_spec.append(SemanticActionSpec(t=5, v=vel3, s_rel=obs[2].x-ego.x, d_rel=obs[2].y-ego.y))

    costs = NovDemoPolicy._eval_actions(behav_state, semantic_actions, actions_spec, max_velocity)
    print(costs)
