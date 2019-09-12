import numpy as np
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner, \
    ScenarioParams
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.src.planning.behavioral.state.state import ObjectSize
from unittest.mock import patch


@patch('decision_making.src.planning.behavioral.rule_based_lane_merge.VELOCITY_LIMITS', [0, 18])
def test_createSafeMergeActions_useMaxVel():
    ego_fstate = np.array([-60, 15, 0])
    ego_size = ObjectSize(5, 2, 0)
    actor1 = ActorState(ego_size, np.array([-120, 20, 0]))
    actor2 = ActorState(ego_size, np.array([0, 20, 0]))
    actors = [actor1, actor2]
    state = LaneMergeState(ego_fstate, ego_size, actors, merge_point_red_line_dist=20, merge_point_s_in_gff=200)
    lane_merge_actions, action_specs = RuleBasedLaneMergePlanner.create_safe_actions(
        state, ScenarioParams(worst_case_back_car_accel=0, worst_case_front_car_decel=3))
    assert len(lane_merge_actions) > 0
    final_v = np.array([action.action_specs[-1].v_T for action in lane_merge_actions])
    full_times = np.array([sum([spec.t for spec in action.action_specs]) for action in lane_merge_actions])
    assert (np.logical_and(16 <= final_v, final_v <= 18)).all()
    assert (np.logical_and(3 <= full_times, full_times <= 4)).all()


def test_createSafeMergeActions_useStop():
    ego_fstate = np.array([-40., 10., 0.])
    ego_size = ObjectSize(5., 2., 0.)
    actor1 = ActorState(ego_size, np.array([-80., 10., 0.]))
    actor2 = ActorState(ego_size, np.array([-30., 20., 0.]))
    actors = [actor1, actor2]
    state = LaneMergeState(ego_fstate, ego_size, actors, merge_point_red_line_dist=20., merge_point_s_in_gff=200.)
    lane_merge_actions, action_specs = RuleBasedLaneMergePlanner.create_safe_actions(
        state, ScenarioParams(worst_case_back_car_accel=0, worst_case_front_car_decel=0))
    assert len(action_specs) > 0
    costs = RuleBasedLaneMergePlanner.evaluate(lane_merge_actions)
    best_action = lane_merge_actions[np.argmin(costs)]
    assert len(best_action.action_specs) == 1
    best_spec = best_action.action_specs[0]
    assert best_spec.v_0 == best_spec.v_T
    assert best_spec.ds < best_spec.v_0 * best_spec.t <= 1.1 * best_spec.ds
