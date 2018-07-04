from logging import Logger

import numpy as np
import time

from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING, \
    DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.types import FS_SX, FS_SV, C_A, C_K, C_X, C_Y, C_YAW, C_V
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import ObjectSize, State, DynamicObject, EgoState
from mapping.src.service.map_service import MapService


class SafetyUtilsTrajectoriesFixture:

    @staticmethod
    def create_simple_trajectories():
        ego_size = np.array([4, 2])
        lane_wid = 3.5

        T = 10.
        times_step = 0.1
        times_num = int(T / times_step) + 1
        zeros = np.zeros(times_num)
        time_range = np.arange(0, T + 0.001, times_step)

        # ego trajectories (Quintic polynomials):
        #   traj[0] moves with constant velocity 10 m/s
        #   traj[1] moves with constant velocity 20 m/s
        #   traj[2] moves with constant velocity 30 m/s and moves laterally to the left from lane 0 to lane 1.
        #   traj[3] moves with constant velocity 30 m/s and moves laterally to the right from lane 2 to lane 1.
        ego_lon = 200
        v = np.array([10, 20, 30])
        lane = lane_wid/2 + np.arange(0, 2*lane_wid + 0.001, lane_wid)  # 3 lanes
        constraints_s = np.array([[ego_lon, v[0], 0, ego_lon + T * v[0], v[0], 0],
                                  [ego_lon, v[1], 0, ego_lon + T * v[1], v[1], 0],
                                  [ego_lon, v[2], 0, ego_lon + T * v[2], v[2], 0],
                                  [ego_lon, v[2], 0, ego_lon + T * v[2], v[2], 0]])
        poly_s = WerlingPlanner._solve_1d_poly(constraints_s, T, QuinticPoly1D)
        fstates_s = QuinticPoly1D.polyval_with_derivatives(poly_s, time_range)

        constraints_d = np.array([[lane[0], 0, 0, lane[0], 0, 0],
                                  [lane[0], 0, 0, lane[0], 0, 0],
                                  [lane[0], 0, 0, lane[1], 0, 0],
                                  [lane[2], 0, 0, lane[1], 0, 0]])

        poly_d = WerlingPlanner._solve_1d_poly(constraints_d, T, QuinticPoly1D)
        fstates_d = QuinticPoly1D.polyval_with_derivatives(poly_d, time_range)

        ego_ftraj = np.dstack((fstates_s, fstates_d))

        # object[0] moves with velocity 10 m/s on the rightest lane
        # object[1] object moves with velocity 20 m/s on the second lane
        # object[2] moves with velocity 30 m/s on the right lane
        # object[3] moves with velocity 20 m/s on the right lane, and starts from lon=135
        # object[4] moves with velocity 20 m/s on the right lane, and starts from lon=165
        # object[5] moves longitudinally as ego_traj[2], and moves laterally from lane 2 to lane 1
        # object[6] moves longitudinally as ego_traj[2], and moves laterally from lane 1 to lane 2
        # object[7] moves longitudinally as ego_traj[2], and moves laterally from lane 1 to lane 0
        sv1 = np.repeat(v[0], times_num)
        sv2 = np.repeat(v[1], times_num)
        sv3 = np.repeat(v[2], times_num)
        sx1 = ego_lon + time_range * sv1[0]
        sx2 = ego_lon + time_range * sv2[0]
        sx3 = ego_lon + time_range * sv3[0]
        dx = np.repeat(lane_wid/2, times_num)
        dv3 = np.repeat(lane_wid / (T - times_step), times_num)

        obj_size = np.array([4, 2])
        obj_ftraj = np.array([np.c_[sx1 + sv1[0] + (ego_size[0] + obj_size[0]) / 2 + 1, sv1, zeros, dx, zeros, zeros],
                              np.c_[sx2, sv2, zeros, dx + lane_wid, zeros, zeros],
                              np.c_[sx3 + sv3[0] + (ego_size[0] + obj_size[0]) / 2 + 1, sv3, zeros, dx, zeros, zeros],
                              np.c_[sx2 + 4.5 * sv3[0], sv2, zeros, dx, zeros, zeros],
                              np.c_[sx2 + 5.5 * sv3[0], sv2, zeros, dx, zeros, zeros],
                              np.c_[sx3, sv3, zeros, dx + lane_wid * 2 - time_range * dv3[0], -dv3,
                                    zeros],
                              np.c_[sx3, sv3, zeros, dx + lane_wid + time_range * dv3[0], dv3, zeros],
                              np.c_[sx3, sv3, zeros, dx + lane_wid - time_range * dv3[0], -dv3, zeros]])
        obj_sizes = np.tile(obj_size, obj_ftraj.shape[0]).reshape(obj_ftraj.shape[0], 2)

        return ego_ftraj, ego_size, obj_ftraj, obj_sizes

    @staticmethod
    def create_trajectories_for_F():
        ego_size = np.array([4, 2])
        lane_wid = 3.5

        T = 10.
        times_step = 0.1
        times_num = int(T / times_step) + 1
        zeros = np.zeros(times_num)
        time_range = np.arange(0, T + 0.001, times_step)

        # all ego trajectories start from lon=0 and from the rightest lane (quintic polynomials)
        # traj[0] moves with constant velocity 30 m/s and moves laterally to the left from lane 0 to lane 1. T_d = 5
        # traj[1] is like traj[0], but with T_d = 2.6
        # traj[2] is like traj[0], but with T_d = 2.5
        ego_lon = 200
        v = np.array([10, 20, 30])
        lane = lane_wid/2 + np.arange(0, 2*lane_wid + 0.001, lane_wid)  # 3 lanes
        constraints_s = np.array([[ego_lon, v[2], 0, ego_lon + T * v[2], v[2], 0],
                                  [ego_lon, v[2], 0, ego_lon + T * v[2], v[2], 0],
                                  [ego_lon, v[2], 0, ego_lon + T * v[2], v[2], 0]])
        poly_s = WerlingPlanner._solve_1d_poly(constraints_s, T, QuinticPoly1D)
        fstates_s = QuinticPoly1D.polyval_with_derivatives(poly_s, time_range)

        constraints_d = np.array([[lane[0], 0, 0, lane[1], 0, 0],
                                  [lane[0], 0, 0, lane[1], 0, 0],
                                  [lane[0], 0, 0, lane[1], 0, 0]])

        poly_d1 = WerlingPlanner._solve_1d_poly(np.array([constraints_d[0]]), T=5, poly_impl=QuinticPoly1D)
        poly_d2 = WerlingPlanner._solve_1d_poly(np.array([constraints_d[1]]), T=2.6, poly_impl=QuinticPoly1D)
        poly_d3 = WerlingPlanner._solve_1d_poly(np.array([constraints_d[2]]), T=2.5, poly_impl=QuinticPoly1D)

        fstates_d1 = QuinticPoly1D.polyval_with_derivatives(poly_d1, time_range)
        fstates_d2 = QuinticPoly1D.polyval_with_derivatives(poly_d2, time_range)
        fstates_d3 = QuinticPoly1D.polyval_with_derivatives(poly_d3, time_range)
        fstates_d = np.concatenate((fstates_d1, fstates_d2, fstates_d3))

        ego_ftraj = np.dstack((fstates_s, fstates_d))

        # all objects move on the right lane
        #   object[0] moves with velocity 20 m/s and starts from safe lon=127
        #   object[1] moves with velocity 10 m/s and starts from safe lon=165
        #   object[2] moves with velocity 10 m/s and starts from safe lon=193 (time delay=2.9)
        #   object[3] moves with velocity 10 m/s and starts from safe lon=195 (time delay=3)
        sv1 = np.repeat(v[0], times_num)
        sv2 = np.repeat(v[1], times_num)
        sv3 = np.repeat(v[2], times_num)
        sx1 = ego_lon + time_range * sv1[0]
        sx2 = ego_lon + time_range * sv2[0]
        dx = np.repeat(lane_wid/2, times_num)

        obj_size = np.array([4, 2])
        obj_ftraj = np.array([np.c_[sx2 + 127, sv2, zeros, dx, zeros, zeros],
                              np.c_[sx1 + 165, sv1, zeros, dx, zeros, zeros],
                              np.c_[sx1 + 192, sv1, zeros, dx, zeros, zeros],
                              np.c_[sx1 + 195, sv1, zeros, dx, zeros, zeros],
                              np.c_[sx1 - 100, sv3, zeros, dx + lane_wid, zeros, zeros]])
        obj_sizes = np.tile(obj_size, obj_ftraj.shape[0]).reshape(obj_ftraj.shape[0], 2)

        return ego_ftraj, ego_size, obj_ftraj, obj_sizes

    @staticmethod
    def create_trajectories_for_LB_LF():
        ego_size = np.array([4, 2])
        lane_wid = 3.5

        T = 10.
        times_step = 0.1
        times_num = int(T / times_step) + 1
        zeros = np.zeros(times_num)
        time_samples = np.arange(0, T + 0.001, times_step)

        # all ego trajectories start from the rightest lane (quartic polynomials)
        ego_lon = 200
        v = np.array([10, 20, 30])
        lane = lane_wid/2 + np.arange(0, 2*lane_wid + 0.001, lane_wid)  # 3 lanes
        constraints_s = np.array([[ego_lon, v[0], 0, v[0], 0],  # from 10 to 10 m/s
                                  [ego_lon, v[0], 0, v[1], 0],  # from 10 to 20 m/s
                                  [ego_lon, v[0], 0, v[2], 0],  # from 10 to 30 m/s
                                  [ego_lon, v[1], 0, v[2], 0],  # from 20 to 30 m/s
                                  [ego_lon, v[2], 0, v[2], 0],  # from 30 to 30 m/s
                                  [ego_lon, v[0], 0, v[1], 0]]) # from 10 to 20 m/s
        poly_s = WerlingPlanner._solve_1d_poly(constraints_s, T, QuarticPoly1D)
        fstates_s = QuinticPoly1D.polyval_with_derivatives(poly_s, time_samples)

        constraints_d = np.array([[lane[0], 0, 0, lane[1], 0, 0]])

        T_d = np.array([6, 3])
        # all trajectories except the last one have normal lane change time (T_d = 6 sec)
        poly_d1 = WerlingPlanner._solve_1d_poly(np.tile(constraints_d, constraints_s.shape[0] - 1).
                                                reshape(constraints_s.shape[0] - 1, 6), T=T_d[0], poly_impl=QuinticPoly1D)
        # the last trajectory has fast lane change time (T_d = 3 sec)
        poly_d2 = WerlingPlanner._solve_1d_poly(np.array([constraints_d[-1]]), T=T_d[1], poly_impl=QuinticPoly1D)
        fstates_d1 = QuinticPoly1D.polyval_with_derivatives(poly_d1, time_samples)
        fstates_d2 = QuinticPoly1D.polyval_with_derivatives(poly_d2, time_samples)

        # fill all elements of ftraj_d beyond T_d by the values of ftraj_d at T_d
        last_sample = np.where(time_samples >= T_d[0])[0][0]
        fstates_d1[:, last_sample+1:, :] = fstates_d1[:, last_sample:last_sample+1, :]
        last_sample = np.where(time_samples >= T_d[1])[0][0]
        fstates_d2[:, last_sample+1:, :] = fstates_d2[:, last_sample:last_sample+1, :]
        fstates_d = np.concatenate((fstates_d1, fstates_d2))

        ego_ftraj = np.dstack((fstates_s, fstates_d))

        # objects start lon=0 m behind ego
        sv0 = zeros + 9
        sx0 = ego_lon + time_samples * sv0
        sv2 = zeros + v[2]
        sx2 = ego_lon + time_samples * sv2
        dx = zeros + 3*lane_wid/2  # middle lane

        obj_size = np.array([4, 2])
        # objects start behind and ahead ego and move with velocity 30 m/s
        obj_ftraj = np.array([np.c_[sx2 - 120, sv2, zeros, dx, zeros, zeros],     # LB
                              np.c_[sx2 + 20, sv2, zeros, dx, zeros, zeros],      # close LF
                              np.c_[sx2 - 220, sv2, zeros, dx, zeros, zeros],     # far LB
                              np.c_[sx2 + 40, sv2, zeros, dx, zeros, zeros],      # LF
                              np.c_[sx0 - 20, sv0, zeros, dx, zeros, zeros]])     # close slow LB (vel=9)
        obj_sizes = np.tile(obj_size, obj_ftraj.shape[0]).reshape(obj_ftraj.shape[0], 2)

        return ego_ftraj, ego_size, obj_ftraj, obj_sizes


def test_calcSafetyForTrajectories_egoAndSomeObjectsMoveLaterally_checkSafetyCorrectnessForManyScenarios():
    """
    Test safety of 4 different ego trajectories w.r.t. 8 different objects moving on three lanes with different
    velocities and starting from different latitudes and longitudes. All velocities are constant along trajectories.
    In two trajectories ego changes lane. 3 last objects change lane.
    The test checks safety for whole trajectories.
    """
    ego_ftraj, ego_size, obj_ftraj, obj_sizes = SafetyUtilsTrajectoriesFixture.create_simple_trajectories()

    safe_times = SafetyUtils.calc_safety_for_trajectories(ego_ftraj, ego_size, obj_ftraj, obj_sizes)

    assert safe_times[0][0].all()       # move with the same velocity and start on the safe distance
    assert safe_times[0][1].all()       # object is ahead and faster, therefore safe
    assert not safe_times[1][0].all()   # the object ahead is slower, therefore unsafe
    assert safe_times[1][1].all()       # move on different lanes, then safe

    assert not safe_times[2][0].all()   # ego is faster
    assert not safe_times[2][1].all()   # ego is faster
    assert safe_times[2][2].all()       # move with the same velocity
    assert not safe_times[2][3].all()   # obj becomes unsafe longitudinally at time 4, before it becomes safe laterally
    assert safe_times[2][4].all()       # obj becomes unsafe longitudinally at time 7, exactly when it becomes safe laterally
    assert not safe_times[2][5].all()   # obj & ego move laterally one to another, becomes unsafe laterally at the end
    assert safe_times[2][6].all()       # obj & ego move laterally to the left, keeping lateral distance, and always safe
    assert not safe_times[2][7].all()   # obj & ego move laterally in opposite directions and don't keep safe distance
    assert safe_times[3][7].all()       # obj & ego move laterally to the right, keeping lateral distance, and always safe


def test_calcSafetyForTrajectories_overtakeOfSlowF_safeOnlyIfObjectFisFar():

    ego_ftraj, ego_size, obj_ftraj, obj_sizes = SafetyUtilsTrajectoriesFixture.create_trajectories_for_F()

    safe_times = SafetyUtils.calc_safety_for_trajectories(ego_ftraj, ego_size, obj_ftraj, obj_sizes)

    assert safe_times[0][0].all()       # ego_v=30 overtake obj_v=20, time_delay = 2   (127 m), T_d = 5     safe
    assert not safe_times[1][1].all()   # ego_v=30 overtake obj_v=10, time_delay = 2   (165 m), T_d = 2.6   unsafe
    assert safe_times[2][1].all()       # ego_v=30 overtake obj_v=10, time_delay = 2   (165 m), T_d = 2.5   safe
    assert not safe_times[0][2].all()   # ego_v=30 overtake obj_v=10, time_delay = 2.9 (192 m), T_d = 5     unsafe
    assert safe_times[0][3].all()       # ego_v=30 overtake obj_v=10, time_delay = 3.0 (195 m), T_d = 5     safe


def test_calcSafetyForTrajectories_safetyWrtLBLF_safeOnlyIfObjectLBisFar():

    ego_ftraj, ego_size, obj_ftraj, obj_sizes = SafetyUtilsTrajectoriesFixture.create_trajectories_for_LB_LF()

    safe_times = SafetyUtils.calc_safety_for_trajectories(ego_ftraj, ego_size, obj_ftraj, obj_sizes)

    assert not safe_times[0][0].all()   # slow ego (10 m/s) is unsafe w.r.t. fast LB (30 m/s, 120 m behind ego)
    assert safe_times[0][1].all()       # slow ego (10 m/s) is safe w.r.t. close and fast LF (30 m/s)
    assert not safe_times[1][0].all()   # slowly accelerating ego (10-20 m/s) is unsafe w.r.t. LB (30 m/s, 120 m behind ego)
    assert safe_times[1][1].all()       # mid-vel ego (10-20 m/s) is safe w.r.t. close and fast LF (30 m/s)
    assert safe_times[4][0].all()       # fast ego (30 m/s) is safe w.r.t. fast LB (30 m/s, 120 m behind ego)
    assert not safe_times[4][1].all()   # fast ego (30 m/s) is unsafe w.r.t. close LF with same velocity
    assert safe_times[4][3].all()       # fast ego (30 m/s) is safe w.r.t. LF with same velocity
    assert not safe_times[3][0].all()   # accelerating ego (20->30 m/s) is unsafe w.r.t. LB (30 m/s, 120 m behind ego)
    assert safe_times[2][1].all()       # accelerating ego (10->30 m/s) is safe w.r.t. close LF (30 m/s)
    assert safe_times[3][1].all()       # accelerating ego (20->30 m/s) is safe w.r.t. close LF (30 m/s)
    assert safe_times[3][2].all()       # accelerating ego (20->30 m/s) is safe w.r.t. far LB (220 m behind ego)
    assert not safe_times[1][2].all()   # slow accelerating ego (10->20 m/s, T_d = 6) is unsafe w.r.t. far LB
    # In this version if ego becomes unsafe w.r.t. B/LB, without lateral movement of ego or after completing the lane
    # change, it's considered unsafe. TODO: insert considering of lateral movement
    assert not safe_times[5][2].all()   # slowly accelerating ego (10->20 m/s, T_d = 3) is unsafe safe w.r.t. far LB
    # in the following test ego starts longitudinally unsafe and becomes safe before it becomes unsafe laterally
    assert safe_times[2][4].all()       # ego (10->30 m/s, T_d = 6) is safe wrt close slow LB (9 m/s, 20 m behind ego)


def test_calcSafetyForTrajectories_egoAndSingleObject_checkSafetyCorrectnessForManyScenarios():
    """
    Test safety of different ego trajectories w.r.t. a SINGLE object moving on the rightest lane.
    All velocities are constant along trajectories.
    In two trajectories ego changes lane.
    The test checks safety for whole trajectories.
    """
    ego_ftraj, ego_size, obj_ftraj, obj_sizes = SafetyUtilsTrajectoriesFixture.create_simple_trajectories()

    # test with a single object
    safe_times = SafetyUtils.calc_safety_for_trajectories(ego_ftraj, ego_size, obj_ftraj[0], obj_sizes[0])
    assert safe_times[0].all()      # move with the same velocity and start on the safe distance
    assert not safe_times[1].all()  # the object ahead is slower, therefore unsafe
    assert not safe_times[2].all()  # ego is faster
    assert safe_times[3].all()      # ego moves from lane 2 to lane 1, and the object on lane 0


def test_calcSafeTd():

    logger = Logger("test_calc_safe_T_d")
    lane_width = 3.6
    ego_size = ObjectSize(4, 2, 0)
    ego_lon = 400
    ego_lat = lane_width / 2
    ego_vel = 10
    road_id = 20
    road_frenet = get_road_rhs_frenet_by_road_id(road_id)

    predictor = RoadFollowingPredictor(logger)
    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING),
                                                 DynamicActionSpace(logger, predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING)])
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)

    t = 20.
    times_step = 0.1
    time_samples = np.arange(0, t + 0.001, times_step)
    samples_num = time_samples.shape[0]

    ego = create_canonic_ego(0, ego_lon, ego_lat, ego_vel, ego_size, road_frenet)
    ego_fstate = np.array([ego.road_localization.road_lon, ego.v_x, 0, ego.road_localization.intra_road_lat, 0, 0])

    obj_sizes = np.array([ObjectSize(4, 2, 0), ObjectSize(6, 2, 0), ObjectSize(6, 2, 0), ObjectSize(4, 2, 0)])

    F = create_canonic_object(1, 0, ego_lon + 25, ego_lat, ego_vel, obj_sizes[0], road_frenet)
    LF = create_canonic_object(2, 0, ego_lon + 20, ego_lat + lane_width, ego_vel + 6, obj_sizes[1], road_frenet)
    LB = create_canonic_object(4, 0, ego_lon - 40, ego_lat + lane_width, ego_vel + 2, obj_sizes[3], road_frenet)
    objects = [F, LF, LB]

    predictions = []
    obj_sizes_mat = []
    for i, obj in enumerate(objects):
        fstate = np.array([obj.road_localization.road_lon, obj.v_x, 0, obj.road_localization.intra_road_lat, 0, 0])
        prediction = np.tile(fstate, samples_num).reshape(samples_num, 6)
        prediction[:, 0] = fstate[FS_SX] + time_samples * fstate[FS_SV]
        predictions.append(prediction)
        obj_sizes_mat.append(np.array([obj.size.length, obj.size.width]))
    predictions = np.array(predictions)
    obj_sizes_mat = np.array(obj_sizes_mat)

    state = State(None, objects, ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)
    recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(recipes, behavioral_state)

    # Action specification
    specs = np.full(recipes.__len__(), None)
    valid_action_recipes = [recipe for i, recipe in enumerate(recipes) if recipes_mask[i]]
    specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

    specs_mask = action_spec_validator.filter_action_specs(list(specs), behavioral_state)

    T_d = SafetyUtils.calc_safe_T_d(ego_fstate, np.array([ego_size.length, ego_size.width]), specs, specs_mask,
                                    time_samples, predictions, obj_sizes_mat)
    # TODO: after considering the rear object in RSS, verify the action 63 is safe for T_d = 3
    assert T_d[63] == 0     # static action T_s=16 v_T=13.9 with lane change: unsafe wrt faster LB since T_s is long
    assert T_d[65] == 5     # static action T_s=6.4 v_T=13.9 with lane change: T_d is affected by close F
    assert T_d[68] == 3     # static action T_s=6.4 v_T=16.7 with lane change: T_d is short because of close F
    assert T_d[120] == 3    # dynamic action T_s=7.6 with lane change: T_d is short because of close F


def create_canonic_ego(timestamp: int, lon: float, lat: float, vel: float, size: ObjectSize,
                       road_frenet: FrenetSerret2DFrame) -> EgoState:
    """
    Create ego with zero lateral velocity and zero accelerations
    """
    fstate = np.array([lon, vel, 0, lat, 0, 0])
    cstate = road_frenet.fstate_to_cstate(fstate)
    return EgoState(0, timestamp, cstate[C_X], cstate[C_Y], 0, cstate[C_YAW], size, 0, cstate[C_V], 0,
                    cstate[C_A], cstate[C_K])


def create_canonic_object(obj_id: int, timestamp: int, lon: float, lat: float, vel: float, size: ObjectSize,
                          road_frenet: FrenetSerret2DFrame) -> DynamicObject:
    """
    Create object with zero lateral velocity and zero accelerations
    """
    fstate = np.array([lon, vel, 0, lat, 0, 0])
    cstate = road_frenet.fstate_to_cstate(fstate)
    return DynamicObject(obj_id, timestamp, cstate[C_X], cstate[C_Y], 0, cstate[C_YAW], size, 0, cstate[C_V], 0,
                         cstate[C_A], cstate[C_K])


def get_road_rhs_frenet_by_road_id(road_id: int):
    return MapService.get_instance()._rhs_roads_frenet[road_id]
