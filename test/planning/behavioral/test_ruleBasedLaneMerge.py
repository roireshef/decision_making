import numpy as np
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, VELOCITY_LIMITS, EGO_LENGTH, \
    BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner, \
    ScenarioParams
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeActorState, LaneMergeState
from decision_making.src.planning.types import LIMIT_MAX
import matplotlib.pyplot as plt
from logging import Logger

from decision_making.src.planning.utils.kinematics_utils import BrakingDistances


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
    actor1 = LaneMergeActorState(-90, 20, ego_len)
    actor2 = LaneMergeActorState(60, 20, ego_len)
    actors = [actor1, actor2]
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


def test_canSolveByRuleBased_fasterBackCarIsNotFarEnough_success():
    ego_fstate = np.array([60, 23, 0])
    ego_len = 5
    actor1 = LaneMergeActorState(-30, 25, ego_len)
    actor2 = LaneMergeActorState(40, 25, ego_len)
    actors = [actor1, actor2]
    red_line_s = 120
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.choose_max_vel_quartic_trajectory(state)[0]
    assert ret


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


def test_canSolveByRuleBased_success():
    w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.AGGRESSIVE.value]
    braking_dist, _ = BrakingDistances.calc_quartic_action_distances(w_T, w_J, np.array([25.]), np.array([0.]))
    red_line_s = 240
    ego_fstate = np.array([red_line_s - braking_dist[0], 25, 0])
    ego_len = 5
    actor1 = LaneMergeActorState(-26, 25, ego_len)
    actor2 = LaneMergeActorState(88, 25, ego_len)
    actors = [actor1, actor2]
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, red_line_s)
    actions = RuleBasedLaneMergePlanner.create_max_vel_quartic_actions(state)
    assert len(actions) > 0


def test_optimalPolicy():
    ego_fstate = np.array([60, 20, 0])
    ego_len = 5
    actor1 = LaneMergeActorState(-250, 25, ego_len)
    actor2 = LaneMergeActorState(-45, 25, ego_len)
    actor3 = LaneMergeActorState(30, 25, ego_len)
    actors = [actor1, actor2, actor3]
    red_line_s = 120
    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, red_line_s)
    success, rule_based_trajectory = RuleBasedLaneMergePlanner.get_optimal_action_trajectory(
        state, ScenarioParams(worst_case_back_actor_accel=0, worst_case_front_actor_decel=0))
    assert success


def test_optimalPolicy1():
    ego_fstate = np.array([210, 0, 0])
    red_line_s = 240
    ego_len = 5
    actors_vel = 25

    actors = []
    for s in range(-100, 0, actors_vel):
        actors.append(LaneMergeActorState(s, actors_vel, ego_len))
    for s in range(ego_fstate[0], actors_vel*32, 2*actors_vel):
        actors.append(LaneMergeActorState(-s, actors_vel, ego_len))

    state = LaneMergeState.create_thin_state(ego_len, ego_fstate, actors, red_line_s)
    success, rule_based_trajectory = RuleBasedLaneMergePlanner.get_optimal_action_trajectory(
        state, ScenarioParams(worst_case_back_actor_accel=0, worst_case_front_actor_decel=0))
    success=success


def test_statistics():
    actors_vel = 15
    actors_density = 0.3
    length = 5
    road_length = 270
    max_cars_num = int(3*road_length / actors_vel)
    states_num = 500
    red_line_s = 240
    ego_range = np.arange(0, 230.001, 5)
    results_per_ego_s = np.zeros(len(ego_range), dtype=float)
    actors_sum = 0

    params = ScenarioParams(ego_max_velocity=actors_vel, actors_max_velocity=actors_vel+1,
                            worst_case_front_actor_decel=0.1, worst_case_back_actor_accel=0)

    for i, ego_s in enumerate(ego_range):
        ego_fstate = np.array([ego_s, 0, 0])
        for _ in range(states_num):
            # generate actors' state
            shift = np.random.uniform(actors_vel)
            rand = np.random.uniform(size=max_cars_num)
            actors_num = np.sum(rand < actors_density) + 1
            actors_s = np.zeros(actors_num)
            actors_s[1:] = -2*road_length + shift + (np.arange(max_cars_num) * actors_vel)[rand < actors_density]
            actors = [LaneMergeActorState(actor_s - ego_s, actors_vel, length) for actor_s in actors_s]

            state = LaneMergeState.create_thin_state(length, ego_fstate, actors, red_line_s)
            success, ret = RuleBasedLaneMergePlanner.choose_max_vel_quartic_trajectory(state, params)
            results_per_ego_s[i] += (ret.shape[0] > 0)
            actors_sum += actors_num

    print('%s' % list(results_per_ego_s/states_num))
    plt.plot(ego_range, results_per_ego_s/states_num)
    plt.show()
