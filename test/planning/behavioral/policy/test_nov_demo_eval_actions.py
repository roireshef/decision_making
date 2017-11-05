from typing import Tuple, Dict, List

from decision_making.src.planning.behavioral.constants import SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LAT_LEFT, \
    SEMANTIC_CELL_LON_FRONT
from decision_making.src.planning.behavioral.policies.november_demo_semantic_policy import NovDemoPolicy, \
    NovDemoBehavioralState
from decision_making.src.planning.behavioral.policy import PolicyConfig
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticAction, SemanticActionType, \
    SemanticActionSpec
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State, EgoState, DynamicObject, ObjectSize
from decision_making.src.state.state_module import StateModule, Logger
from mapping.test.model.testable_map_fixtures import *


def test_novDemoEvalSemanticActions(testable_map_api):
    max_velocity = 10  # m/s

    ego_v = max_velocity
    pos1 = np.array([33, -3])
    pos2 = np.array([41, 0])
    pos3 = np.array([39, 3])

    vel1_list = [13, 10,  7, 13,  7, 10,  7,  7, 10, 7]
    vel2_list = [10, 13, 10,  7, 13,  7,  7, 10,  7, 7]
    vel3_list = [ 7,  7, 13, 10, 10, 13, 10,  7,  7, 7]
    result    = [ 0,  0,  1,  2,  1,  2,  2,  1,  1, 1]

    for test in range(len(vel1_list)):
        vel1 = vel1_list[test]
        vel2 = vel2_list[test]
        vel3 = vel3_list[test]

        ego_x = 0
        t = 5
        # t1 = (pos1[0]-ego_x) / (0.5*(ego_v+vel1))
        # t2 = (pos2[0]-ego_x) / (0.5*(ego_v+vel2))
        # t3 = (pos3[0]-ego_x) / (0.5*(ego_v+vel3))

        size = ObjectSize(1.5, 0.5, 0)
        obs = list([
            DynamicObject(obj_id=0, timestamp=0, x=pos1[0], y=pos1[1], z=0, yaw=0, size=size,
                          road_localization=DynamicObject.compute_road_localization(pos1, 0, testable_map_api),
                          confidence=1.0, v_x=vel1, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
            DynamicObject(obj_id=1, timestamp=0, x=pos2[0], y=pos2[1], z=0, yaw=0, size=size,
                          road_localization=DynamicObject.compute_road_localization(pos2, 0, testable_map_api),
                          confidence=1.0, v_x=vel2, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
            DynamicObject(obj_id=2, timestamp=0, x=pos3[0], y=pos3[1], z=0, yaw=0, size=size,
                          road_localization=DynamicObject.compute_road_localization(pos3, 0, testable_map_api),
                          confidence=1.0, v_x=vel3, v_y=0, acceleration_lon=0.0, omega_yaw=0.0)
        ])

        ego = EgoState(obj_id=-1, timestamp=0, x=ego_x, y=0, z=0, yaw=0, size=size,
                       road_localization=DynamicObject.compute_road_localization(np.array([ego_x, 0]), 0.0, testable_map_api),
                       confidence=1.0, v_x=ego_v, v_y=0, steering_angle=0.0, acceleration_lon=0.0, omega_yaw=0.0)

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
        predictor = Predictor(testable_map_api)
        policy = NovDemoPolicy(Logger("NovDemoTest"), PolicyConfig(), predictor, testable_map_api)

        semantic_actions = []
        semantic_actions.append(SemanticAction((-1, 1), obs[0], SemanticActionType.FOLLOW))
        semantic_actions.append(SemanticAction((0, 1), obs[1], SemanticActionType.FOLLOW))
        semantic_actions.append(SemanticAction((1, 1), obs[2], SemanticActionType.FOLLOW))
        actions_spec = []
        actions_spec.append(SemanticActionSpec(t=t, v=vel1, s_rel=obs[0].x-ego.x, d_rel=obs[0].y-ego.y))
        actions_spec.append(SemanticActionSpec(t=t, v=vel2, s_rel=obs[1].x-ego.x, d_rel=obs[1].y-ego.y))
        actions_spec.append(SemanticActionSpec(t=t, v=vel3, s_rel=obs[2].x-ego.x, d_rel=obs[2].y-ego.y))

        costs = policy._eval_actions(behav_state, semantic_actions, actions_spec)
        assert(costs[result[test]] == 1)

    # from bokeh.plotting import figure, show, output_file
    # p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")
    # p1.circle(pos1[0], pos1[1])
    # output_file("legend.html", title="legend.py example")
    # show(p1)  # open a browser
    # print(costs)

def test_get_actionIndByLane(testable_map_api):
    semantic_action = SemanticAction((SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LON_FRONT), None, SemanticActionType(1))
    spec1 = SemanticActionSpec(t=5, v=10, s_rel=30, d_rel=-3)
    policy = NovDemoPolicy(Logger("NovDemoTest"), PolicyConfig(), Predictor(testable_map_api), testable_map_api)
    action_ind = policy._get_action_ind_by_lane([semantic_action], [spec1], SEMANTIC_CELL_LAT_RIGHT)
    assert action_ind == 0
    action_ind = policy._get_action_ind_by_lane([semantic_action], [spec1], SEMANTIC_CELL_LAT_LEFT)
    assert action_ind is None
