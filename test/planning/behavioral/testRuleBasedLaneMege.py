import numpy as np
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeActorState, LaneMergeState


def test_canSolveByRuleBased_closeFrontCarRequiresStrongBrake_success():
    ego_fstate = np.array([200, 18.16185665,  4.42184403])
    ego_len = 5
    actor1 = LaneMergeActorState(-72.551955563549306, 25, ego_len)
    actor2 = LaneMergeActorState(68.6480444364507, 25, ego_len)
    actors = [actor1, actor2]
    red_line_s = 240
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.choose_max_vel_quartic_trajectory(state)
    assert len(ret) > 0


def test_createSafeActions():
    ego_fstate = np.array([100, 10., 1.])
    ego_len = 5
    actor1 = LaneMergeActorState(-72.551955563549306, 25, ego_len)
    actor2 = LaneMergeActorState(68.6480444364507, 25, ego_len)
    actor3 = LaneMergeActorState(108.6480444364507, 25, ego_len)
    actors = [actor1, actor2, actor3]
    red_line_s = 240
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.create_safe_actions(state)
