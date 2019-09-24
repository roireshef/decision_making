import numpy as np
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, VELOCITY_LIMITS, EGO_LENGTH
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner, \
    ScenarioParams, SimpleLaneMergeState
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.src.planning.behavioral.state.state import ObjectSize
from unittest.mock import patch

from decision_making.src.planning.types import LIMIT_MAX
import matplotlib.pyplot as plt


def test_quinticApproximator_plotVelocityProfiles():

    f = plt.figure(1)
    for i in range(10):
        v_0 = np.random.uniform(0, 30)
        v_T = np.random.uniform(0, 30)
        ds = np.random.uniform(50, 300)
        T = 16
        r = ds / T - v_0
        dv = v_0 - v_T
        sign = np.sign(ds - 0.5 * (v_0 + v_T) * T)  # 1 if required average velocity ds/T > average_vel, -1 otherwise
        c = 3. ** sign  # ratio between deceleration and acceleration
        first_dv = r + sign * np.sqrt(r * r + 2 * dv / (c + 1) * (r + 0.5 * dv))  # first_dv = v_t - v_0

        v_t = first_dv + v_0
        t = c * first_dv * T / (first_dv * (c+1) + v_0 - v_T)
        if ds > 0.5 * (v_0 + v_T) * T:
            assert np.isclose(3 * (v_t - v_0) / t, (v_t - v_T) / (T - t))
        else:
            assert np.isclose((v_0 - v_t) / t, 3 * (v_T - v_t) / (T - t))

        plt.plot(np.array([0, t, T]), np.array([v_0, v_t, v_T]))
    plt.xlabel('T')
    plt.ylabel('v_T')
    plt.show(f)


def test_calculateSafeTargetPoints_plotSafePoints():
    ego_len = 5
    ds = 60
    actor1 = np.array([-90, 20, ego_len])
    actor2 = np.array([60, 20, ego_len])
    actors = np.vstack((actor1, actor2))
    #actors = actor1[np.newaxis]
    import time
    st = time.time()
    v_T, T = RuleBasedLaneMergePlanner._calculate_safe_target_points(EGO_LENGTH, actors, ds, ScenarioParams())
    print('\ntime=', time.time() - st)
    f = plt.figure(1)
    axes = plt.gca()
    axes.set_xlim([0, BP_ACTION_T_LIMITS[LIMIT_MAX]])
    axes.set_ylim([0, VELOCITY_LIMITS[LIMIT_MAX]])
    plt.xlabel('T')
    plt.ylabel('v_T')
    plt.scatter(T, v_T)
    plt.show(f)


def test_canSolveByRuleBased_fasterBackCar():
    ego_fstate = np.array([224.49009284,  19.52321304,   1.08213813])
    ego_len = 5
    actor1 = np.array([-31.79009284, 25, ego_len])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert ret is None


def test_canSolveByRuleBased_fastEgo():
    ego_fstate = np.array([189.49494949,  25,   1.08213813])
    ego_len = 5
    actor1 = np.array([-189.49494949, 25., 5])
    actor2 = np.array([90.70505051, 25., 5])
    actor3 = np.array([43.20505051, 25., 5])
    actor4 = np.array([-134.29494949, 25., 5])
    actors = np.vstack((actor1, actor2, actor3, actor4))
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert ret is not None


def test_canSolveByRuleBased_slowEgoSafe():
    ego_fstate = np.array([228.72611638,  0.30320583,   0.59706981])
    ego_len = 5
    actor1 = np.array([-186.02611638, 25., 5])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert ret is not None


def test_canSolveByRuleBased_slowEgoUnsafe():
    ego_fstate = np.array([229.45172369,  1.156745,   1.11542085])
    ego_len = 5
    actor1 = np.array([-161.75172369, 25., 5])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert ret is None


def test_canSolveByRuleBased_safe():
    ego_fstate = np.array([2.25006479e+02, 1.83408332e-03, 0.00000000e+00])
    ego_len = 5
    actor1 = np.array([-74.80647901,   25., 5])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert ret is None


def test_canSolveByRuleBased_unsafe():
    ego_fstate = np.array([225.11986475,   0.30318166,   0.59706987])
    ego_len = 5
    actor1 = np.array([-1.97419865e+02,    25., 5])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert ret is None


def test_canSolveByRuleBased_fasterBackCarIsFar_failure():
    ego_fstate = np.array([60, 15, 0])
    ego_len = 5
    actor1 = np.array([-80, 20, ego_len])
    actor2 = np.array([60, 20, ego_len])
    actors = np.vstack((actor1, actor2))
    red_line_s = 120
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert ret is None


def test_canSolveByRuleBased_fasterBackCarIsNotFarEnough_success():
    ego_fstate = np.array([60, 23, 0])
    ego_len = 5
    actor1 = np.array([-60, 25, ego_len])
    actor2 = np.array([60, 25, ego_len])
    actors = np.vstack((actor1, actor2))
    red_line_s = 120
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert ret is not None


def test_canSolveByRuleBased_closeFrontCarRequiresStrongBrake_failure():
    ego_fstate = np.array([60, 20, 0])
    ego_len = 5
    actor1 = np.array([-86, 25, ego_len])
    actor2 = np.array([30, 25, ego_len])
    actors = np.vstack((actor1, actor2))
    red_line_s = 90
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    assert not RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)


def test_canSolveByRuleBased_closeFrontCarRequiresBrake_success():
    ego_fstate = np.array([60, 20, 0])
    ego_len = 5
    actor1 = np.array([-87, 25, ego_len])
    actor2 = np.array([30, 25, ego_len])
    actors = np.vstack((actor1, actor2))
    red_line_s = 90
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    assert RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)


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
