import numpy as np
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeActorState, LaneMergeState


def test_canSolveByRuleBased_closeFrontCarRequiresStrongBrake_success():
    ego_fstate = np.array([37.55195556, 18.16185665,  4.42184403])
    ego_len = 5
    actor1 = LaneMergeActorState(-32.551955563549306, 25, ego_len)
    actor2 = LaneMergeActorState(92.6480444364507, 25, ego_len)
    actors = [actor1, actor2]
    red_line_s = 240
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.choose_max_vel_quartic_trajectory(state)[0]
    assert ret


