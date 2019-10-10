from logging import Logger
from typing import List
import numpy as np
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS, LON_ACC_LIMITS, VELOCITY_LIMITS, \
    EGO_LENGTH, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, BP_ACTION_T_LIMITS, EPS, \
    TRAJECTORY_TIME_RESOLUTION

from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, ActionSpec, ActionRecipe, \
    RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.default_config import DEFAULT_ACTION_SPEC_FILTERING
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.types import FS_SX, FS_SV, FS_SA, FrenetState1D, LIMIT_MIN, LIMIT_MAX, FS_1D_LEN, \
    BoolArray
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D, Poly1D
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState

WORST_CASE_FRONT_CAR_DECEL = 2.  # [m/sec^2]
WORST_CASE_BACK_CAR_ACCEL = 1.  # [m/sec^2]


class ScenarioParams:
    def __init__(self, worst_case_back_actor_accel: float = WORST_CASE_BACK_CAR_ACCEL,
                 worst_case_front_actor_decel: float = WORST_CASE_FRONT_CAR_DECEL,
                 ego_reaction_time: float = SAFETY_HEADWAY, back_actor_reaction_time: float = SAFETY_HEADWAY,
                 front_rss_decel: float = -LON_ACC_LIMITS[LIMIT_MIN], back_rss_decel: float = 5.0,
                 ego_max_velocity: float = 25., actors_max_velocity: float = 28.):
        """
        :param worst_case_back_actor_accel: worst case braking deceleration of front actor prior the merge
        :param worst_case_front_actor_decel: worst case acceleration of back actor prior the merge
        :param ego_reaction_time: [sec] reaction delay of ego
        :param back_actor_reaction_time: [sec] reaction delay of actor
        :param front_rss_decel: [m/s^2] maximal braking deceleration of ego & front actor
        :param back_rss_decel: [m/s^2] maximal braking deceleration of ego & back actor (may be lower)
        :param ego_max_velocity: [m/sec] maximal velocity of ego
        :param actors_max_velocity: [m/sec] maximal velocity of actors
        """
        # prediction parameters
        self.worst_case_back_actor_accel = worst_case_back_actor_accel
        self.worst_case_front_actor_decel = worst_case_front_actor_decel

        # RSS parameters
        self.ego_reaction_time = ego_reaction_time
        self.back_actor_reaction_time = back_actor_reaction_time
        self.front_rss_decel = front_rss_decel
        self.back_rss_decel = back_rss_decel
        self.ego_max_velocity = ego_max_velocity
        self.actors_max_velocity = actors_max_velocity


class LaneMergeSpec:
    def __init__(self, t: float, v_0: float, a_0: float, v_T: float, ds: float, poly_coefs_num: int):
        """
        lane-merge sub-action specification
        :param t: time period
        :param v_0: initial velocity
        :param a_0: initial acceleration
        :param v_T: end velocity
        :param ds: distance in s
        :param poly_coefs_num: number of coefficients in the spec's polynomial
        """
        self.t = t
        self.v_0 = v_0
        self.a_0 = a_0
        self.v_T = v_T
        self.ds = ds
        self.poly_coefs_num = poly_coefs_num


class LaneMergeSequence:
    def __init__(self, action_specs: List[LaneMergeSpec]):
        self.action_specs = action_specs


class SimpleLaneMergeState:
    def __init__(self, ego_length: float, ego_fstate: FrenetState1D, actors_s_vel_length: np.array, red_line_s: float):
        """
        simple lane merge state
        :param ego_length: [m] length of the host car
        :param ego_fstate: ego longitudinal Frenet state
        :param actors_s_vel_length: Nx3 array of actors' data. Each row contains: s (relative to ego), velocity, size.length
        :param red_line_s: s of the red line
        """
        self.ego_length = ego_length
        self.ego_fstate = ego_fstate
        self.actors_s_vel_length = actors_s_vel_length
        self.red_line_s = red_line_s

    def to_string(self):
        return f'EGO: {self.ego_fstate} + \r\nACTORS:{self.actors_s_vel_length}'

    @classmethod
    def create_from_lane_merge_state(cls, lane_merge_state: LaneMergeState):
        # extract main road actors
        actors_data = []
        for lon_pos in RelativeLongitudinalPosition:
            for obj in lane_merge_state.road_occupancy_grid[(lane_merge_state.merge_side, lon_pos)]:
                actors_data.append([obj.longitudinal_distance, obj.dynamic_object.velocity, obj.dynamic_object.size.length])

        ego_fstate = lane_merge_state.projected_ego_fstates[RelativeLane.SAME_LANE][:FS_1D_LEN]
        return cls(lane_merge_state.ego_state.size.length, ego_fstate, np.array(actors_data), lane_merge_state.red_line_s)


class RuleBasedLaneMergePlanner(BasePlanner):

    TIME_GRID_RESOLUTION = 0.2
    VEL_GRID_RESOLUTION = 0.5
    MAX_TARGET_S_ERROR = 0.5  # [m] maximal allowed error for actions' terminal s

    def __init__(self, lane_merge_state: LaneMergeState, actions: List[LaneMergeSequence], logger: Logger):
        super().__init__(lane_merge_state, logger)
        self.actions = actions

    @staticmethod
    def acceleration_to_max_vel_is_safe(state: SimpleLaneMergeState, params: ScenarioParams = ScenarioParams()) -> \
            np.array:
        """
        Check existence of rule-based solution that can merge safely, assuming the worst case scenario of
        main road actors. The function tests a single static action toward maximal velocity (ScenarioParams.ego_max_velocity).
        If the action is safe (longitudinal RSS) during crossing red line w.r.t. all main road actors, return True.
        :param state: simple lane merge state, containing data about host and the main road vehicles
        :param params: scenario parameters, describing the worst case actors' behavior and RSS safety parameters
        :return: accelerations array or None if there is no safe action
        """
        import time
        st = time.time()

        s_0, v_0, a_0 = state.ego_fstate
        rel_red_line_s = state.red_line_s - s_0
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS.T
        specs_t, specs_s = RuleBasedLaneMergePlanner._specify_quartic(v_0, a_0, params.ego_max_velocity, w_T, w_J)

        # validate lon. acceleration limits and choose the most calm valid action
        valid_acc, poly_s = RuleBasedLaneMergePlanner._validate_acceleration(v_0, a_0, params.ego_max_velocity, specs_t)
        poly_s = poly_s[valid_acc]
        specs_s = specs_s[valid_acc]
        specs_t = specs_t[valid_acc]

        target_t = specs_t + np.maximum(0, rel_red_line_s - specs_s) / params.ego_max_velocity
        target_v = np.full(specs_t.shape[0], params.ego_max_velocity, dtype=float)
        target_s = np.full(specs_t.shape[0], np.maximum(specs_s, rel_red_line_s), dtype=float)

        overflow_actions = rel_red_line_s < specs_s

        if overflow_actions.any():
            times = np.arange(0, np.max(specs_t[overflow_actions]) + EPS, TRAJECTORY_TIME_RESOLUTION)
            s_values = Math.polyval2d(poly_s[overflow_actions], times)
            poly_vel = Math.polyder2d(poly_s[overflow_actions], m=1)
            red_line_idxs = np.argmin(np.abs(s_values - rel_red_line_s), axis=1)
            target_t[overflow_actions] = red_line_idxs * TRAJECTORY_TIME_RESOLUTION
            target_v[overflow_actions] = Math.zip_polyval2d(poly_vel, target_t[overflow_actions, np.newaxis])[:, 0]
            target_s[overflow_actions] = s_values[np.arange(s_values.shape[0]), red_line_idxs]

        actors_s = state.actors_s_vel_length[:, 0, np.newaxis]
        actors_v = state.actors_s_vel_length[:, 1, np.newaxis]
        margins = 0.5 * (state.actors_s_vel_length[:, 2, np.newaxis] + state.ego_length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT

        # check safety assuming the worst case scenario
        # here safety_matrix is 1x1, since we check only one terminal point (v_T, T) at the red line
        is_safe_action = RuleBasedLaneMergePlanner._create_RSS_matrix(
            actors_s, actors_v, margins, target_v, target_t, target_s,
            params.worst_case_front_actor_decel, params.worst_case_back_actor_accel,
            params.ego_reaction_time, params.back_actor_reaction_time, params.front_rss_decel, params.back_rss_decel,
            params.actors_max_velocity)  # enable back actor to exceed max_vel to remain consistent with control errors

        # if overflow_actions.any():
        #     times = np.arange(0, np.max(specs_t[overflow_actions]) + EPS, TRAJECTORY_TIME_RESOLUTION)
        #     s_values = Math.polyval2d(poly_s[overflow_actions], times)
        #     poly_vel = Math.polyder2d(poly_s[overflow_actions], m=1)
        #     red_line_idxs = np.argmin(np.abs(s_values - rel_red_line_s), axis=1)
        #     overflow_target_t = red_line_idxs * TRAJECTORY_TIME_RESOLUTION
        #     overflow_target_v = Math.zip_polyval2d(poly_vel, overflow_target_t[:, np.newaxis])[:, 0]
        #     overflow_target_s = s_values[np.arange(s_values.shape[0]), red_line_idxs]
        #     target_t = np.concatenate((target_t, overflow_target_t))
        #     target_v = np.concatenate((target_v, overflow_target_v))
        #     target_s = np.concatenate((target_s, overflow_target_s))
        #
        # actors_s = state.actors_s_vel_length[:, 0, np.newaxis]
        # actors_v = state.actors_s_vel_length[:, 1, np.newaxis]
        # margins = 0.5 * (state.actors_s_vel_length[:, 2, np.newaxis] + state.ego_length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
        #
        # # check safety assuming the worst case scenario
        # # here safety_matrix is 1x1, since we check only one terminal point (v_T, T) at the red line
        # is_safe_concatenated = RuleBasedLaneMergePlanner._check_headway_safety(
        #     actors_s, actors_v, margins, target_v, target_t, target_s,
        #     params.worst_case_front_actor_decel, params.worst_case_back_actor_accel,
        #     params.ego_reaction_time, params.back_actor_reaction_time, params.actors_max_velocity)
        #
        # is_safe_action = is_safe_concatenated[:len(specs_t)]
        # is_safe_action[overflow_actions] &= is_safe_concatenated[len(specs_t):]

        OUTPUT_LENGTH = 10
        if is_safe_action.any() or rel_red_line_s < 0:
            action_idx = np.argmax(is_safe_action)  # the most calm safe action
            poly_acc = np.polyder(poly_s[action_idx], m=2)
            times = np.arange(0, min(OUTPUT_LENGTH * TRAJECTORY_TIME_RESOLUTION, specs_t[action_idx]), TRAJECTORY_TIME_RESOLUTION)
            accelerations = np.zeros(OUTPUT_LENGTH)
            accelerations[:times.shape[0]] = np.polyval(poly_acc, times)
            #print('\ntime=', time.time() - st)
            return accelerations

        #print('\ntime=', time.time() - st)
        return np.array([])

    @staticmethod
    def can_solve_by_two_constant_accelerations(state: SimpleLaneMergeState, params: ScenarioParams = ScenarioParams()) -> bool:
        """
        Check existence of rule-based solution that can merge safely, assuming the worst case scenario.
        :param state: simple lane merge state, containing data about host and the main road vehicles
        :param params: scenario parameters, describing the worst case actors' behavior and RSS safety parameters
        :return: True if a rule-based solution exists
        """
        import time
        st = time.time()
        s_0, v_0, a_0 = state.ego_fstate
        # calculate the actions' distance (the same for all actions)
        ds = state.red_line_s - s_0

        v_T, T = RuleBasedLaneMergePlanner._calculate_safe_target_points(state.ego_length, state.actors_s_vel_length, ds, params)

        # Assume that T is divided into two parts with constant acceleration in the each part.
        # Acceleration should be lower than deceleration by a constant factor.
        ACCELERATION_SIGN_RATIO = 3.  # the ratio between deceleration and acceleration
        sign = 2 * (ds > 0.5 * (v_0 + v_T) * T) - 1  # 1 if required average velocity ds/T > average_vel, -1 otherwise
        c = ACCELERATION_SIGN_RATIO ** sign  # 3 for initial acceleration, 1/3 for initial deceleration
        # Solve system of two equations with two unknowns: vt and t, where t is the first part ending time,
        # and vt is the velocity at time t:
        # 1. Area under the graph should be s:
        #       t * (v0 + vt)/2 + (T - t)(vt + vT)/2 = ds
        # 2. If sign==1, initial acceleration should be 3 times weaker than final acceleration (c=3).
        #    Otherwise initial acceleration should be 3 times stronger than final acceleration (c=1/3).
        #       c * (vt - v0) / t = (vt - vT) / (T - t)
        # Substitute variables rv and delta_v for more convenient calculation:
        rv = ds / T - v_0  # required average velocity relative to the initial velocity
        delta_v = v_0 - v_T
        # Let x = init_delta_v = vt - v0. Solve quadratic equation for x:
        #       x^2 - 2*x * (ds/T - v0) - delta_v/(c+1) * (2*ds/T - v0 - vT) = 0
        init_delta_v = rv + sign * np.sqrt(rv * rv + 2 * delta_v / (c + 1) * (rv + 0.5 * delta_v))  # vt - v0
        init_T = c * init_delta_v * T / (init_delta_v * (c + 1) + delta_v)  # t (time of the initial part)
        init_accel = init_delta_v / (init_T + EPS)  # acceleration of the first part
        end_accel = (init_delta_v + delta_v) / (T - init_T + EPS)  # acceleration of the second part

        # calculate init & end acceleration normalized by ACCELERATION_SIGN_RATIO
        normalized_init_accel = np.maximum(c, 1) * np.abs(init_accel)
        normalized_end_accel = np.maximum(1./c, 1) * np.abs(end_accel)
        normalized_accel = np.maximum(normalized_init_accel, normalized_end_accel)
        best_safe_point_idx = np.argmin(normalized_accel)

        # # for the found safe point check if there is a quitic action with valid accelerations and velocities
        # best_v_T, best_T = v_T[best_safe_point_idx], T[best_safe_point_idx]
        # qintic_actions, _ = RuleBasedLaneMergePlanner._create_single_actions(
        #       state.ego_fstate, best_v_T[np.newaxis], best_T[np.newaxis], ds)
        #return len(qintic_actions) > 0

        print('time=', time.time() - st)
        return normalized_accel[best_safe_point_idx] < 4.5

    @staticmethod
    def create_safe_actions(state: SimpleLaneMergeState, params: ScenarioParams = ScenarioParams()) -> \
            List[LaneMergeSequence]:
        """
        Create all possible actions to the merge point, filter unsafe actions, filter actions exceeding vel-acc limits,
        calculate time-jerk cost for the remaining actions.
        :param state: SimpleLaneMergeState containing distance to merge and actors
        :param params: scenario params, e.g. worst-case cars accelerations
        :return: list of initial action specs, array of jerks until the merge, array of times until the merge
        """
        import time
        st_tot = time.time()

        st = time.time()
        ego_fstate = state.ego_fstate
        # calculate the actions' distance (the same for all actions)
        ds = state.red_line_s - ego_fstate[FS_SX]

        v_T, T = RuleBasedLaneMergePlanner._calculate_safe_target_points(state.ego_length, state.actors_s_vel_length, ds, params)

        time_safety = time.time() - st

        st = time.time()
        # create single-spec quintic safe actions
        single_actions, violates_max_vel = RuleBasedLaneMergePlanner._create_single_actions(ego_fstate, v_T, T, ds)

        max_vel_actions = stop_actions = []
        time_quintic = time.time() - st

        st = time.time()
        # create composite "max_velocity" actions, such that each sequence includes quartic + const_max_vel + quartic
        if violates_max_vel:
            max_vel_actions = RuleBasedLaneMergePlanner._create_composite_actions(ego_fstate, T, VELOCITY_LIMITS[1], v_T, ds)

        # create composite "stop" actions, such that each sequence includes quartic + zero_vel + quartic
        # create stop actions only if there are no single-spec actions, since quintic actions have lower time
        # and lower jerk than composite stop action
        if len(single_actions) == 0:
            stop_actions = RuleBasedLaneMergePlanner._create_composite_actions(ego_fstate, T, 0, v_T, ds)
        time_quartic = time.time() - st

        lane_merge_actions = single_actions + max_vel_actions + stop_actions

        print('\ntime = %f: safety=%f quintic=%f quartic=%f' % (time.time() - st_tot, time_safety, time_quintic, time_quartic))
        return lane_merge_actions

    @staticmethod
    def _create_single_actions(ego_fstate: FrenetState1D, v_T: np.array, T: np.array, ds: float) -> \
            [List[LaneMergeSequence], bool]:

        # calculate s_profile coefficients for all actions
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
        poly_coefs = QuinticPoly1D.s_profile_coefficients(a_0, v_0, v_T, ds, T)

        # the fast analytic kinematic filter reduce the load on the regular kinematic filter
        # calculate actions that don't violate acceleration limits
        valid_acc = QuinticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)
        valid_vel = QuinticPoly1D.are_velocities_in_limits(poly_coefs[valid_acc], T[valid_acc], VELOCITY_LIMITS)
        valid_idxs = np.where(valid_acc)[0][valid_vel]

        actions = [LaneMergeSequence([LaneMergeSpec(t, v_0, a_0, vT, ds, QuinticPoly1D.num_coefs())])
                   for t, vT in zip(T[valid_idxs], v_T[valid_idxs])]

        # check if there are actions that try to violate maximal vel_limit
        not_too_high_vel = QuinticPoly1D.are_velocities_in_limits(poly_coefs[valid_acc], T[valid_acc], np.array([-np.inf, VELOCITY_LIMITS[1]]))
        return actions, (~not_too_high_vel).any()

    @staticmethod
    def _create_composite_actions(ego_fstate: FrenetState1D, T: np.array, v_mid: float, v_T: np.array, ds: float) \
            -> List[LaneMergeSequence]:

        # sort lexicographically first by v_T and then by descending order of T
        ordered_idxs = np.lexsort((-T, v_T))
        v_T, T = v_T[ordered_idxs], T[ordered_idxs]
        # get array of unique final velocities
        v_T_unique, unique_index, v_T_unique_inverse = np.unique(v_T, return_index=True, return_inverse=True)
        # for each unique v_T calculate the maximal appropriate T
        T_max = T[unique_index]

        # calculate initial and end quartic actions with all final velocities and all jerk-time weights:
        # from initial to maximal velocity and from maximal velocity to v_T_unique
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
        T_init, s_init, T_end_unique, s_end_unique = RuleBasedLaneMergePlanner._specify_quartic_actions(v_0, a_0, v_mid,
                                                                                                        v_T_unique, T_max, ds)
        T_end, s_end = T_end_unique[v_T_unique_inverse], s_end_unique[v_T_unique_inverse]

        # calculate T_mid & s_mid
        T_mid = T[:, np.newaxis] - T_init - T_end
        T_mid[np.less(T_mid, 0, where=~np.isnan(T_mid))] = np.nan
        if np.isnan(T_mid).all():
            return []
        s_mid = T_mid * v_mid

        # for each input pair (v_T, T) find w, for which the total s (s_init + s_mid + s_end) is closest to ds
        s_total = s_init + s_mid + s_end
        s_error = np.abs(s_total - ds)
        s_error[np.isnan(s_error)] = np.inf  # np.nanargmin returns error, when a whole line is nan
        best_w_idx = np.argmin(s_error, axis=1)  # for each (v_T, T) get w_J such that s_error is minimal

        # choose pairs (v_T, T), for which s_error are under the threshold
        min_s_error = s_error[np.arange(s_error.shape[0]), best_w_idx]  # lowest s_error for each input (v_T, T)
        chosen_idxs = np.where(min_s_error < RuleBasedLaneMergePlanner.MAX_TARGET_S_ERROR)[0]
        chosen_w_J_idxs = best_w_idx[chosen_idxs]
        chosen_T_init, chosen_s_init = T_init[chosen_w_J_idxs], s_init[chosen_w_J_idxs]
        chosen_T_end, chosen_s_end = T_end[chosen_idxs, chosen_w_J_idxs], s_end[chosen_idxs, chosen_w_J_idxs]
        chosen_T_mid = T[chosen_idxs] - chosen_T_init - chosen_T_end
        chosen_s_mid = chosen_T_mid * v_mid
        chosen_v_T = v_T[chosen_idxs]

        # verify that the middle action has non-negative time
        assert np.all(chosen_T_mid >= 0)

        actions = [LaneMergeSequence([LaneMergeSpec(t1, v_0, a_0, v_mid, s1, QuarticPoly1D.num_coefs()),
                                      LaneMergeSpec(t2, v_mid, 0, v_mid, s2, poly_coefs_num=2),
                                      LaneMergeSpec(t3, v_mid, 0, v_end, s3, QuarticPoly1D.num_coefs())])
                   for t1, s1, t2, s2, t3, s3, v_end in
                   zip(chosen_T_init, chosen_s_init, chosen_T_mid, chosen_s_mid, chosen_T_end, chosen_s_end, chosen_v_T)]
        return actions

    @staticmethod
    def _specify_quartic_actions(v_0: float, a_0: float, v_mid: float, v_T: np.array, T_max: np.array, s_max: float):

        # create grid for jerk-time weights
        w_J = np.geomspace(16, 0.001, 128)  # jumps of factor 1.37
        w_T = np.full(w_J.shape, BP_JERK_S_JERK_D_TIME_WEIGHTS[0, 2])

        # calculate initial quartic actions
        T_init, s_init = RuleBasedLaneMergePlanner._specify_quartic(v_0, a_0, v_mid, w_T, w_J)

        # filter out initial specs and weights with exceeding T, s and acceleration
        valid_s_T_idxs = np.where((T_init <= np.max(T_max)) & (s_init <= s_max))[0]
        valid_acc, _ = RuleBasedLaneMergePlanner._validate_acceleration(v_0, a_0, v_mid, T_init[valid_s_T_idxs])
        w_J, w_T = w_J[valid_s_T_idxs[valid_acc]], w_T[valid_s_T_idxs[valid_acc]]
        T_init, s_init = T_init[valid_s_T_idxs[valid_acc]], s_init[valid_s_T_idxs[valid_acc]]

        # calculate final quartic actions
        v_T_meshgrid, _ = np.meshgrid(v_T, w_J, indexing='ij')
        v_T_meshgrid = v_T_meshgrid.ravel()
        w_J_meshgrid = np.tile(w_J, v_T.shape[0])
        w_T_meshgrid = np.full(w_J_meshgrid.shape, w_T[0])
        T_end, s_end = RuleBasedLaneMergePlanner._specify_quartic(v_mid, 0, v_T_meshgrid, w_T_meshgrid, w_J_meshgrid)

        # validate final specs with exceeding T, s and acceleration
        valid_s_T = (np.tile(T_init, v_T.shape[0]) + T_end <= np.repeat(T_max, w_J.shape[0])) & \
                    (np.tile(s_init, v_T.shape[0]) + s_end <= s_max)
        valid_acc = np.zeros_like(T_end).astype(bool)
        valid_acc[valid_s_T], _ = RuleBasedLaneMergePlanner._validate_acceleration(v_mid, 0, v_T_meshgrid[valid_s_T], T_end[valid_s_T])
        s_end[~(valid_s_T & valid_acc)] = np.nan
        T_end[~(valid_s_T & valid_acc)] = np.nan

        T_end = T_end.reshape(v_T.shape[0], w_J.shape[0])
        s_end = s_end.reshape(v_T.shape[0], w_J.shape[0])
        return T_init, s_init, T_end, s_end

    @staticmethod
    def _specify_quartic(v_0: float, a_0: float, v_T: np.array, w_T: np.array, w_J: np.array) -> [np.array, np.array]:
        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuarticPoly1D.time_cost_function_derivative_coefs(w_T=w_T, w_J=w_J, a_0=a_0, v_0=v_0, v_T=v_T)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, BP_ACTION_T_LIMITS)
        T = np.fmin.reduce(roots_s, axis=-1)
        s = QuarticPoly1D.distance_profile_function(a_0=a_0, v_0=v_0, v_T=v_T, T=T)(T)
        s[np.isclose(T, 0)] = 0
        return T, s

    @staticmethod
    def _validate_acceleration(v_0: float, a_0: float, v_T: float, T: np.array) -> [np.array, np.array]:
        """
        Check acceleration in limits for quartic polynomials.
        * Use faster implementation than QuarticPoly1D.are_accelerations_in_limits
        :param v_0: initial velocity
        :param a_0: initial acceleration
        :param v_T: target velocity(es): either scalar or array of size len(T)
        :param T: array of planning times
        :return: boolean array of size len(T) of valid actions and matrix Nx5: s_profile polynomials for all T
        """
        # validate acceleration limits of the initial quartic action
        poly_s = np.zeros((T.shape[0], QuarticPoly1D.num_coefs()))
        validT = ~np.isnan(T)
        poly_s[validT] = QuarticPoly1D.position_profile_coefficients(a_0, v_0, v_T if np.isscalar(v_T) else v_T[validT], T[validT])
        # acc_poly = Math.polyder2d(poly_s[validT], m=2)
        # jerk_poly = Math.polyder2d(acc_poly, m=1)
        # roots = (-jerk_poly[:, 1] / jerk_poly[:, 0])[:, np.newaxis]
        # max_accelerations = Math.zip_polyval2d(acc_poly, roots)[:, 0]
        valid_acc = np.zeros_like(T).astype(bool)
        # valid_acc[validT] = NumpyUtils.is_in_limits(max_accelerations, LON_ACC_LIMITS)
        valid_acc[validT] = QuarticPoly1D.are_accelerations_in_limits(poly_s[validT], T[validT], LON_ACC_LIMITS)
        return valid_acc, poly_s

    @staticmethod
    def _calculate_safe_target_points(ego_length: float, actors_data: np.array, target_s: float,
                                      params: ScenarioParams) -> [np.array, np.array]:
        """
        Create boolean 2D matrix of actions that are longitudinally safe (RSS) between red line & merge point w.r.t.
        all actors.
        :param ego_length: [m] length of host car
        :param actors_data: Nx3 array of actors' data. Each row contains: s (relative to ego), velocity, size.length
        :return: two 1D arrays of v_T & T, where ego is safe at target_s relatively to all actors
        """
        # grid of possible planning times
        t_grid = np.arange(RuleBasedLaneMergePlanner.TIME_GRID_RESOLUTION, BP_ACTION_T_LIMITS[LIMIT_MAX] + EPS,
                           RuleBasedLaneMergePlanner.TIME_GRID_RESOLUTION)
        # grid of possible target ego velocities
        v_grid = np.arange(0, VELOCITY_LIMITS[LIMIT_MAX] + EPS, RuleBasedLaneMergePlanner.VEL_GRID_RESOLUTION)

        actors_s = actors_data[:, 0, np.newaxis, np.newaxis]
        actors_v = actors_data[:, 1, np.newaxis, np.newaxis]
        margins = 0.5 * (actors_data[:, 2, np.newaxis, np.newaxis] + ego_length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
        ego_v = v_grid[:, np.newaxis]  # target ego velocity should be in different dimension than planning time

        safety_matrix = RuleBasedLaneMergePlanner._create_RSS_matrix(
            actors_s, actors_v, margins, ego_v, t_grid, target_s,
            params.worst_case_front_actor_decel, params.worst_case_back_actor_accel,
            params.ego_reaction_time, params.back_actor_reaction_time, params.front_rss_decel, params.back_rss_decel,
            params.actors_max_velocity)

        vi, ti = np.where(safety_matrix)
        return v_grid[vi], t_grid[ti]

    @staticmethod
    def _create_RSS_matrix(actors_s: np.array, actors_v: np.array, margins: np.array,
                           target_v: np.array, target_t: np.array, target_s: np.array,
                           wc_front_decel: float, wc_back_accel: float, ego_hw: float, actor_hw: float,
                           rss_front_decel: float, rss_back_decel: float, actors_max_vel: float) -> BoolArray:
        """
        Given an actor on the main road and two grids of planning times and target velocities, create boolean matrix
        of all possible actions that are longitudinally safe (RSS) at actions' endpoint target_s.
        :param actors_s: current s of actor relatively to the merge point (negative or positive)
        :param actors_v: current actor's velocity
        :param margins: half sum of cars' lengths + safety margin
        :param target_t: array of planning times
        :param target_v: array of target velocities
        :param target_s: array of target s
        :param wc_front_decel: worst case predicted deceleration of front actor before target_t
        :param wc_back_accel: worst case predicted acceleration of back actor before target_t
        :param ego_hw: safety headway of ego
        :param actor_hw: safety headway of an actor
        :param rss_front_decel: maximal deceleration of front actor at target_t
        :param rss_back_decel: maximal deceleration of back actor at target_t (to enable yield use rss_back_decel < rss_front_decel)
        :param actors_max_vel: maximal velocity for actors
        :return: True if the action is safe relatively to the given actor
        """
        front_braking_time = np.minimum(actors_v / wc_front_decel, target_t)
        front_v = actors_v - front_braking_time * wc_front_decel  # target velocity of the front actor
        front_s = actors_s + 0.5 * (actors_v + front_v) * front_braking_time

        back_accel_time = np.minimum((actors_max_vel - actors_v) / wc_back_accel, target_t)
        back_max_vel_time = target_t - back_accel_time  # time of moving with maximal velocity of the back actor
        back_v = actors_v + back_accel_time * wc_back_accel  # target velocity of the back actor
        back_s = actors_s + 0.5 * (actors_v + back_v) * back_accel_time + actors_max_vel * back_max_vel_time

        # calculate if ego is safe according to the longitudinal RSS formula
        front_side_safe = np.maximum(0, target_v * target_v - front_v * front_v) / (2 * rss_front_decel) + \
                          ego_hw * target_v < front_s - target_s - margins
        back_side_safe = np.maximum(0, back_v * back_v - target_v * target_v) / (2 * rss_back_decel) + \
                          actor_hw * back_v < target_s - back_s - margins
        safety_matrix = np.logical_or(front_side_safe, back_side_safe)
        return safety_matrix.all(axis=0)

    @staticmethod
    def _check_headway_safety(actors_s: np.array, actors_v: np.array, margins: np.array,
                              target_v: np.array, target_t: np.array, target_s: np.array,
                              wc_front_decel: float, wc_back_accel: float, ego_hw: float, actor_hw: float,
                              actors_max_vel: float) -> BoolArray:
        """
        Given an actor on the main road, target times and target velocities, create boolean matrix
        of all possible actions that are longitudinally safe at actions' endpoints target_s.
        :param actors_s: current s of actor relatively to the merge point (negative or positive)
        :param actors_v: current actor's velocity
        :param margins: half sum of cars' lengths + safety margin
        :param target_t: array of planning times
        :param target_v: array of target velocities
        :param target_s: array of target s
        :param wc_front_decel: worst case predicted deceleration of front actor before target_t
        :param wc_back_accel: worst case predicted acceleration of back actor before target_t
        :param ego_hw: safety headway of ego
        :param actor_hw: safety headway of an actor
        :param actors_max_vel: maximal velocity for actors
        :return: True if the host is safe during all target times
        """
        front_braking_time = np.minimum(actors_v / wc_front_decel, target_t)
        front_v = actors_v - front_braking_time * wc_front_decel  # target velocity of the front actor
        front_s = actors_s + 0.5 * (actors_v + front_v) * front_braking_time

        back_accel_time = np.minimum((actors_max_vel - actors_v) / wc_back_accel, target_t)
        back_max_vel_time = target_t - back_accel_time  # time of moving with maximal velocity of the back actor
        back_v = actors_v + back_accel_time * wc_back_accel  # target velocity of the back actor
        back_s = actors_s + 0.5 * (actors_v + back_v) * back_accel_time + actors_max_vel * back_max_vel_time

        # calculate if ego is safe according to the longitudinal RSS formula
        front_side_safe = ego_hw * target_v < front_s - target_s - margins
        back_side_safe = actor_hw * back_v < target_s - back_s - margins
        is_safe = np.logical_or(front_side_safe, back_side_safe)
        return is_safe.all(axis=0)

    def _create_actions(self) -> np.array:
        return self.actions

    def _filter_actions(self, actions: np.array) -> np.array:
        first_action_specs = [action[0] for action in actions]  # pick the first spec from each LaneMergeSequence
        action_specs_mask = DEFAULT_ACTION_SPEC_FILTERING.filter_action_specs(first_action_specs, self.behavioral_state)
        filtered_action_specs = np.full(len(first_action_specs), None)
        filtered_action_specs[action_specs_mask] = first_action_specs[action_specs_mask]
        return filtered_action_specs

    def _evaluate_actions(self, actions: np.array) -> np.array:
        """
        Calculate time-jerk costs for the given actions
        :param actions: list of LaneMergeActions, where each LaneMergeAction is a sequence of action specs
        :return: array of actions' time-jerk costs
        """
        # calculate full actions' jerks and times
        actions_jerks = np.zeros(len(actions))
        actions_times = np.zeros(len(actions))
        for poly1d in [QuinticPoly1D, QuarticPoly1D]:
            specs_list = [[idx, spec.t, spec.v_0, spec.a_0, spec.v_T, spec.ds]
                          for idx, action in enumerate(actions)
                          for spec in action.action_specs if spec.poly_coefs_num == poly1d.num_coefs()]
            specs_matrix = np.array(list(filter(None, specs_list)))
            action_idxs, T, v_0, a_0, v_T, ds = specs_matrix.T
            poly_coefs = QuinticPoly1D.s_profile_coefficients(a_0, v_0, v_T, ds, T) \
                if poly1d is QuinticPoly1D else QuarticPoly1D.s_profile_coefficients(a_0, v_0, v_T, T)
            spec_jerks = poly1d.cumulative_jerk(poly_coefs, T)

            # TODO: can be optimized ?
            for action_idx, spec_jerk, spec_t in zip(action_idxs.astype(int), spec_jerks, T):
                actions_jerks[action_idx] += spec_jerk
                actions_times[action_idx] += spec_t

        # calculate actions' costs according to the CALM jerk-time weights
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]
        action_costs = weights[0] * actions_jerks + weights[2] * actions_times
        return action_costs

    def _choose_action(self, actions: np.array, costs: np.array) -> [ActionRecipe, ActionSpec]:
        """
        pick the first action_spec from the best LaneMergeSequence having the minimal cost
        :param actions: array of LaneMergeSequence
        :param costs: array of actions' costs
        :return: [ActionSpec] the first action_spec in the best LaneMergeSequence
        """
        # choose the first spec of the best action having the minimal cost
        best_action_spec = actions[np.argmin(costs)][0]
        # convert spec.s from LaneMergeState to be relative to GFF
        best_action_spec.s += self.lane_merge_state.merge_point_in_gff
        return [None, best_action_spec]
