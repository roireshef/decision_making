import numpy as np
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeActorState, LaneMergeState
from rte.python.logger.AV_logger import AV_Logger


def test_createSafeActions():
    ego_fstate = np.array([100, 24.09309369081506, 0.])
    ego_len = 5
    actor1 = LaneMergeActorState(-25.86602217, 25, ego_len)
    actor2 = LaneMergeActorState(3.43018145, 25, ego_len)
    actors = [actor1, actor2]
    merge_from_s = 103.104162160894248
    red_line_s = 163.10416216089425
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, merge_from_s, red_line_s)
    logger = AV_Logger.get_logger("test")
    planner = RuleBasedLaneMergePlanner(logger)
    actions = planner._create_action_specs(state)
    costs = planner._evaluate_actions(state, None, actions)
