import numpy as np
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, VELOCITY_LIMITS, EGO_LENGTH
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner, \
    ScenarioParams, SimpleLaneMergeState
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState

from decision_making.src.planning.types import LIMIT_MAX
import matplotlib.pyplot as plt
from rte.python.logger.AV_logger import AV_Logger


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
    assert len(ret) == 0


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
    assert len(ret) > 0


def test_canSolveByRuleBased_slowEgoSafe():
    ego_fstate = np.array([228.72611638,  0.30320583,   0.59706981])
    ego_len = 5
    actor1 = np.array([-186.02611638, 25., 5])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert len(ret) > 0


def test_canSolveByRuleBased_slowEgoUnsafe():
    ego_fstate = np.array([229.45172369,  1.156745,   1.11542085])
    ego_len = 5
    actor1 = np.array([-161.75172369, 25., 5])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert len(ret) == 0


def test_canSolveByRuleBased_safe():
    ego_fstate = np.array([2.25006479e+02, 1.83408332e-03, 0.00000000e+00])
    ego_len = 5
    actor1 = np.array([-74.80647901,   25., 5])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert len(ret) == 0


def test_canSolveByRuleBased_unsafe():
    ego_fstate = np.array([225.11986475,   0.30318166,   0.59706987])
    ego_len = 5
    actor1 = np.array([-1.97419865e+02,    25., 5])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert len(ret) == 0


def test_canSolveByRuleBased_safe1():
    ego_fstate = np.array([2.24774265e+02, 1.49699233e-03, 0.00000000e+00])
    ego_len = 5
    actor1 = np.array([-202.0742649,    25., 5])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    safe, acc = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert safe


def test_canSolveByRuleBased_unsafe1():
    ego_fstate = np.array([224.88956723698357,   0.3388668403495103,    0.6279345])
    ego_len = 5
    actor1 = np.array([-177.20570722,    25., 5])
    actors = np.array([actor1])
    red_line_s = 240
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    safe, acc = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert safe


def test_canSolveByRuleBased_fasterBackCarIsFar_failure():
    ego_fstate = np.array([60, 15, 0])
    ego_len = 5
    actor1 = np.array([-80, 20, ego_len])
    actor2 = np.array([60, 20, ego_len])
    actors = np.vstack((actor1, actor2))
    red_line_s = 120
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert len(ret) == 0


def test_canSolveByRuleBased_fasterBackCarIsNotFarEnough_success():
    ego_fstate = np.array([60, 23, 0])
    ego_len = 5
    actor1 = np.array([-60, 25, ego_len])
    actor2 = np.array([60, 25, ego_len])
    actors = np.vstack((actor1, actor2))
    red_line_s = 120
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert len(ret) > 0


def test_canSolveByRuleBased_closeFrontCarRequiresStrongBrake_failure():
    ego_fstate = np.array([60, 20, 0])
    ego_len = 5
    actor1 = np.array([-86, 25, ego_len])
    actor2 = np.array([30, 25, ego_len])
    actors = np.vstack((actor1, actor2))
    red_line_s = 90
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)
    assert len(ret) == 0


def test_canSolveByRuleBased_closeFrontCarRequiresBrake_success():
    ego_fstate = np.array([60, 20, 0])
    ego_len = 5
    actor1 = np.array([-87, 25, ego_len])
    actor2 = np.array([30, 25, ego_len])
    actors = np.vstack((actor1, actor2))
    red_line_s = 90
    state = SimpleLaneMergeState(ego_len, ego_fstate, actors, red_line_s)
    assert RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state)


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
            actors = np.c_[actors_s - ego_s, np.ones(actors_num) * actors_vel, np.ones(actors_num) * length]

            state = SimpleLaneMergeState(length, ego_fstate, actors, red_line_s)
            ret = RuleBasedLaneMergePlanner.acceleration_to_max_vel_is_safe(state, params)
            results_per_ego_s[i] += (ret.shape[0] > 0)
            actors_sum += actors_num

    print('%s' % list(results_per_ego_s/states_num))
    plt.plot(ego_range, results_per_ego_s/states_num)
    plt.show()


from decision_making.src.planning.behavioral.planner.RL_lane_merge_planner import RL_LaneMergePlanner
from gym.spaces.tuple_space import Tuple as GymTuple
from ray.rllib.evaluation import SampleBatch
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.test.planning.behavioral.behavioral_state_fixtures import state_with_objects_before_merge, route_plan_1_2


def test_load_model(state_with_objects_before_merge, route_plan_1_2):

    model = RL_LaneMergePlanner.load_model()

    logger = AV_Logger.get_logger()
    behavioral_state = BehavioralGridState.create_from_state(state_with_objects_before_merge, route_plan_1_2, logger)
    lane_merge_state = LaneMergeState.create_from_behavioral_state(behavioral_state)
    encoded_state: GymTuple = lane_merge_state.encode_state_for_RL()
    logits = model._forward({SampleBatch.CUR_OBS: encoded_state}, [])[0]
    ret = logits