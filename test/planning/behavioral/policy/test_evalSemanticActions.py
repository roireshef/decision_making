import numpy as np

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT
from decision_making.src.planning.behavioral.constants import SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LAT_LEFT, \
    SEMANTIC_CELL_LON_FRONT
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_policy import SemanticActionsGridPolicy
from decision_making.src.planning.behavioral.policies.semantic_actions_grid_state import \
    SemanticActionsGridState
from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticAction, SemanticActionType, \
    SemanticActionSpec
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import EgoState, DynamicObject, ObjectSize
from decision_making.src.state.state_module import Logger
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.test.model.testable_map_fixtures import testable_map_api
from rte.python.logger.AV_logger import AV_Logger

from unittest.mock import patch


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new_callable=testable_map_api)
def test_novDemoEvalSemanticActions(testable_map_api):
    max_velocity = v = BEHAVIORAL_PLANNING_DEFAULT_SPEED_LIMIT  # m/s

    ego_v = max_velocity
    pos1 = np.array([33, -3])
    pos2 = np.array([41, 0])
    pos3 = np.array([39, 3])

    vel1_list = [v+3, v+0, v-3, v+3, v-3, v+0, v-3, v-3, v+0, v-3]
    vel2_list = [v+0, v+3, v+0, v-3, v+3, v-3, v-3, v+0, v-3, v-3]
    vel3_list = [v-3, v-3, v+3, v+0, v+0, v+3, v+0, v-3, v-3, v-3]
    result =    [  0,   0,   1,   0,   1,   0,   2,   1,   0,   1]

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
                          # road_localization=testable_map_api.compute_road_localization(pos1, 0),
                          confidence=1.0, v_x=vel1, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
            DynamicObject(obj_id=1, timestamp=0, x=pos2[0], y=pos2[1], z=0, yaw=0, size=size,
                          # road_localization=testable_map_api.compute_road_localization(pos2, 0),
                          confidence=1.0, v_x=vel2, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
            DynamicObject(obj_id=2, timestamp=0, x=pos3[0], y=pos3[1], z=0, yaw=0, size=size,
                          # road_localization=testable_map_api.compute_road_localization(pos3, 0),
                          confidence=1.0, v_x=vel3, v_y=0, acceleration_lon=0.0, omega_yaw=0.0)
        ])

        ego = EgoState(obj_id=-1, timestamp=0, x=ego_x, y=0, z=0, yaw=0, size=size,
                       # road_localization=testable_map_api.compute_road_localization(np.array([ego_x, 0]), 0.0),
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
        behav_state = SemanticActionsGridState(grid, ego)
        logger = Logger("NovDemoTest")
        predictor = Predictor(logger)
        policy = SemanticActionsGridPolicy(logger, predictor)

        semantic_actions = []
        semantic_actions.append(SemanticAction((-1, 1), obs[0], SemanticActionType.FOLLOW))
        semantic_actions.append(SemanticAction((0, 1), obs[1], SemanticActionType.FOLLOW))
        semantic_actions.append(SemanticAction((1, 1), obs[2], SemanticActionType.FOLLOW))
        actions_spec = []
        actions_spec.append(SemanticActionSpec(t=t, v=vel1, s_rel=obs[0].x - ego.x, d_rel=obs[0].y - ego.y))
        actions_spec.append(SemanticActionSpec(t=t, v=vel2, s_rel=obs[1].x - ego.x, d_rel=obs[1].y - ego.y))
        actions_spec.append(SemanticActionSpec(t=t, v=vel3, s_rel=obs[2].x - ego.x, d_rel=obs[2].y - ego.y))

        costs = policy._eval_actions(behav_state, semantic_actions, actions_spec)
        assert (costs[result[test]] == 1)

        # from bokeh.plotting import figure, show, output_file
        # p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")
        # p1.circle(pos1[0], pos1[1])
        # output_file("legend.html", title="legend.py example")
        # show(p1)  # open a browser
        # print(costs)


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new_callable=testable_map_api)
def test_get_actionIndByLane(testable_map_api):
    logger = AV_Logger.get_logger('test_get_actionIndByLane')
    semantic_action = SemanticAction((SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LON_FRONT), None, SemanticActionType(1))
    spec1 = SemanticActionSpec(t=5, v=10, s_rel=30, d_rel=-3)
    policy = SemanticActionsGridPolicy(Logger("NovDemoTest"), Predictor(logger=logger))
    action_ind = policy._get_action_ind([semantic_action], (SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LON_FRONT))
    assert action_ind == 0
    action_ind = policy._get_action_ind([semantic_action], (SEMANTIC_CELL_LAT_LEFT, SEMANTIC_CELL_LON_FRONT))
    assert action_ind is None
