import numpy as np
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeActorState, LaneMergeState
from rte.python.logger.AV_logger import AV_Logger


def test_createSafeActions():
    ego_fstate = np.array([100, 17.55576896812154, 0.0989034929246979])
    ego_len = 5
    actor1 = LaneMergeActorState(-169.54778223, 25, ego_len)
    actor2 = LaneMergeActorState(-71.1381505, 25, ego_len)
    actor3 = LaneMergeActorState(27.34771223, 25, ego_len)
    actors = [actor1, actor2, actor3]
    merge_from_s = 302.96729536355247
    red_line_s = 342.96729536355247
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, merge_from_s, red_line_s)
    logger = AV_Logger.get_logger("test")
    planner = RuleBasedLaneMergePlanner(logger)
    actions = planner._create_action_specs(state)
    costs = planner._evaluate_actions(state, None, actions)
    best_idx = np.argmin(costs)
    print('best_idx=', best_idx)
    [print(spec) for spec in actions[best_idx].action_specs]
