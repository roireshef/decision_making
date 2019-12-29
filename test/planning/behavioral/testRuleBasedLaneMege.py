import numpy as np
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeActorState, LaneMergeState
from rte.python.logger.AV_logger import AV_Logger


def test_createSafeActions():
    ego_fstate = np.array([100, 19.1627779, -0.2685739])
    ego_len = 5
    actor1 = LaneMergeActorState(-39.61501694, 22.222, ego_len)
    actor2 = LaneMergeActorState(20.25832367, 22.222, ego_len)
    actors = [actor1, actor2]
    merge_from_s = 0
    red_line_s = np.inf
    front_actor = LaneMergeActorState(70.43085479736328, 11.11, ego_len)
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, front_actor, merge_from_s, red_line_s)
    logger = AV_Logger.get_logger("test")
    planner = RuleBasedLaneMergePlanner(logger)
    actions = planner._create_action_specs(state, None)
    costs = planner._evaluate_actions(state, None, actions)
    best_idx = np.argmin(costs)
    print('best_idx=', best_idx)
    [print(spec) for spec in actions[best_idx].action_specs]
