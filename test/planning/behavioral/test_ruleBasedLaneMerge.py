import numpy as np
from decision_making.src.planning.behavioral.rule_based_lane_merge import LaneMergeState, ActorState, RuleBasedLaneMerge
from decision_making.src.state.state import ObjectSize
from unittest.mock import patch


@patch('decision_making.src.planning.behavioral.rule_based_lane_merge.VELOCITY_LIMITS', [0, 18])
def test_createSafeMergeActions_useMaxVel():
    ego_fstate = np.array([-60, 15, 0])
    ego_size = ObjectSize(5, 2, 0)
    actor1 = ActorState(ego_size, np.array([-120, 20, 0]))
    actor2 = ActorState(ego_size, np.array([0, 20, 0]))
    actors = [actor1, actor2]
    state = LaneMergeState(ego_fstate, ego_size, actors, merge_point_red_line_dist=20)
    actions, jerks, times = RuleBasedLaneMerge.create_safe_actions(state, 500)
    assert len(actions) > 0
    vT = np.array([spec.v for spec in actions])
    assert (np.logical_and(17 <= vT, vT <= 18)).all()
    assert (np.logical_and(3 <= times, times <= 4)).all()


def test_createSafeMergeActions_useStop():
    ego_fstate = np.array([-40, 10, 0])
    ego_size = ObjectSize(5, 2, 0)
    actor1 = ActorState(ego_size, np.array([-120, 10, 0]))
    actor2 = ActorState(ego_size, np.array([-30, 20, 0]))
    actors = [actor1, actor2]
    state = LaneMergeState(ego_fstate, ego_size, actors, merge_point_red_line_dist=20)
    actions, jerks, times = RuleBasedLaneMerge.create_safe_actions(state, 500)
    assert len(actions) > 0
    vT = np.array([spec.v for spec in actions])
    T = np.array([spec.t for spec in actions])
