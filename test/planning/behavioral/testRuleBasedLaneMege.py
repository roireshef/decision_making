import numpy as np
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeActorState, LaneMergeState
from rte.python.logger.AV_logger import AV_Logger


def test_createSafeActions():
    ego_fstate = np.array([100, 13.485821280998735, 1.])
    ego_len = 5
    actor1 = LaneMergeActorState(-16.68407334, 25, ego_len)
    actor2 = LaneMergeActorState(37.58871376, 25, ego_len)
    actors = [actor1, actor2]
    merge_from_s = 0
    red_line_s = 153.748177641319444
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, merge_from_s, red_line_s)
    logger = AV_Logger.get_logger("test")
    planner = RuleBasedLaneMergePlanner(logger)
    actions = planner._create_action_specs(state)
    costs = planner._evaluate_actions(state, None, actions)
