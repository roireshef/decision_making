from logging import Logger
from typing import Optional, List
import numpy as np
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS, LON_ACC_LIMITS, VELOCITY_LIMITS, \
    EGO_LENGTH, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, SPECIFICATION_HEADWAY, BP_ACTION_T_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState

from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, ActionSpec, ActionRecipe
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.types import FS_SX, FS_SV, FS_SA, FrenetState1D, LIMIT_MAX
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from decision_making.src.state.lane_merge_state import LaneMergeState, ActorState

WORST_CASE_FRONT_CAR_DECEL = 3  # [m/sec^2]
WORST_CASE_BACK_CAR_ACCEL = 0  # [m/sec^2]


class ScenarioParams:
    def __init__(self, worst_case_back_car_accel: float = WORST_CASE_BACK_CAR_ACCEL,
                 worst_case_front_car_decel: float = WORST_CASE_FRONT_CAR_DECEL):
        self.worst_case_back_car_accel = worst_case_back_car_accel
        self.worst_case_front_car_decel = worst_case_front_car_decel


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


class RuleBasedLaneMergePlanner(BasePlanner):

    TIME_GRID_RESOLUTION = 0.2
    VEL_GRID_RESOLUTION = 0.5
    MAX_TARGET_S_ERROR = 0.5  # [m] maximal allowed error for actions' terminal s

    def __init__(self, lane_merge_state: LaneMergeState, actions: List[LaneMergeSequence], logger: Logger):
        super().__init__(logger)
        self.lane_merge_state = lane_merge_state
        self.actions = actions

    @staticmethod
    def create_safe_actions(lane_merge_state: LaneMergeState, params: ScenarioParams) -> List[LaneMergeSequence]:
        """
        Create all possible actions to the merge point, filter unsafe actions, filter actions exceeding vel-acc limits,
        calculate time-jerk cost for the remaining actions.
        :param lane_merge_state: LaneMergeState containing distance to merge and actors
        :param params: scenario params, e.g. worst-case cars accelerations
        :return: list of initial action specs, array of jerks until the merge, array of times until the merge
        """
        import time
        st_tot = time.time()

        st = time.time()
        t_arr = np.arange(RuleBasedLaneMergePlanner.TIME_GRID_RESOLUTION, BP_ACTION_T_LIMITS[LIMIT_MAX],
                          RuleBasedLaneMergePlanner.TIME_GRID_RESOLUTION)
        v_arr = np.arange(0, VELOCITY_LIMITS[LIMIT_MAX], RuleBasedLaneMergePlanner.VEL_GRID_RESOLUTION)
        # TODO: check safety also on the red line, besides the merge point
        safety_matrix = RuleBasedLaneMergePlanner._create_safety_matrix(lane_merge_state.main_road_actors, t_arr, v_arr, params)
        vi, ti = np.where(safety_matrix)
        T, v_T = t_arr[ti], v_arr[vi]

        # calculate the actions' distance (the same for all actions)
        ds = -lane_merge_state.ego_fstate[FS_SX] - lane_merge_state.ego_size.length / 2
        ego_fstate = lane_merge_state.ego_fstate
        time_safety = time.time() - st

        st = time.time()
        # create single-spec quintic safe actions
        single_actions, violates_high_vel_limit = RuleBasedLaneMergePlanner._create_single_actions(ego_fstate, T, v_T, ds)

        max_vel_actions = stop_actions = []
        time_quintic = time.time() - st

        st = time.time()
        # create composite "max_velocity" actions, such that each sequence includes quartic + const_max_vel + quartic
        if violates_high_vel_limit:
            max_vel_actions = RuleBasedLaneMergePlanner._create_composite_actions(ego_fstate, T, VELOCITY_LIMITS[1], v_T, ds)

        # create composite "stop" actions, such that each sequence includes quartic + zero_vel + quartic
        # create stop actions only if there are no single-spec actions, since quintic actions have lower time
        # and lower jerk than composite stop action
        if len(single_actions) == 0:
            stop_actions = RuleBasedLaneMergePlanner._create_composite_actions(ego_fstate, T, 0, v_T, ds)
        time_quartic = time.time() - st

        lane_merge_actions = single_actions + max_vel_actions + stop_actions

        # # given the list of lane_merge_actions, create list of initial action_specs with s value in GFF
        # ego_s_in_gff = lane_merge_state.merge_point_in_gff + lane_merge_state.ego_fstate[FS_SX]
        # init_specs = [action.action_specs[0] for action in lane_merge_actions]
        # action_specs = [ActionSpec(spec.t, spec.v_T, ego_s_in_gff + spec.ds, d=0, recipe=None) for spec in init_specs]

        print('\ntime = %f: safety=%f quintic=%f quartic=%f' % (time.time() - st_tot, time_safety, time_quintic, time_quartic))
        return lane_merge_actions

    @staticmethod
    def _create_single_actions(ego_fstate: FrenetState1D, T: np.array, v_T: np.array, ds: float) -> \
            [List[LaneMergeSequence], bool]:

        # calculate s_profile coefficients for all actions
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
        poly_coefs = QuinticPoly1D.s_profile_coefficients(a_0, v_0, v_T, ds, T)

        # the fast analytic kinematic filter reduce the load on the regular kinematic filter
        # calculate actions that don't violate acceleration limits
        valid_acc = QuinticPoly1D.are_derivatives_in_limits_zero_coef(2, poly_coefs, T, LON_ACC_LIMITS)
        valid_vel = QuinticPoly1D.are_derivatives_in_limits_zero_coef(1, poly_coefs[valid_acc], T[valid_acc],
                                                                      VELOCITY_LIMITS)
        valid_idxs = np.where(valid_acc)[0][valid_vel]

        actions = [LaneMergeSequence([LaneMergeSpec(t, v_0, a_0, vT, ds, QuinticPoly1D.num_coefs())])
                   for t, vT in zip(T[valid_idxs], v_T[valid_idxs])]

        # check if there are actions that try to violate maximal vel_limit
        not_too_high_vel = QuinticPoly1D.are_derivatives_in_limits_zero_coef(1, poly_coefs[valid_acc], T[valid_acc],
                                                                             np.array([-np.inf, VELOCITY_LIMITS[1]]))
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
        T_init, s_init = RuleBasedLaneMergePlanner._specify(v_0, a_0, v_mid, w_T, w_J)

        # filter out initial specs and weights with exceeding T, s and acceleration
        valid_s_T_idxs = np.where((T_init <= np.max(T_max)) & (s_init <= s_max))[0]
        valid_acc = RuleBasedLaneMergePlanner._validate_acceleration(v_0, a_0, v_mid, T_init[valid_s_T_idxs])
        w_J, w_T = w_J[valid_s_T_idxs[valid_acc]], w_T[valid_s_T_idxs[valid_acc]]
        T_init, s_init = T_init[valid_s_T_idxs[valid_acc]], s_init[valid_s_T_idxs[valid_acc]]

        # calculate final quartic actions
        v_T_meshgrid, _ = np.meshgrid(v_T, w_J, indexing='ij')
        v_T_meshgrid = v_T_meshgrid.ravel()
        w_J_meshgrid = np.tile(w_J, v_T.shape[0])
        w_T_meshgrid = np.full(w_J_meshgrid.shape, w_T[0])
        T_end, s_end = RuleBasedLaneMergePlanner._specify(v_mid, 0, v_T_meshgrid, w_T_meshgrid, w_J_meshgrid)

        # validate final specs with exceeding T, s and acceleration
        valid_s_T = (np.tile(T_init, v_T.shape[0]) + T_end <= np.repeat(T_max, w_J.shape[0])) & \
                    (np.tile(s_init, v_T.shape[0]) + s_end <= s_max)
        valid_acc = np.zeros_like(T_end).astype(bool)
        valid_acc[valid_s_T] = RuleBasedLaneMergePlanner._validate_acceleration(v_mid, 0, v_T_meshgrid[valid_s_T], T_end[valid_s_T])
        s_end[~(valid_s_T & valid_acc)] = np.nan
        T_end[~(valid_s_T & valid_acc)] = np.nan

        T_end = T_end.reshape(v_T.shape[0], w_J.shape[0])
        s_end = s_end.reshape(v_T.shape[0], w_J.shape[0])
        return T_init, s_init, T_end, s_end

    @staticmethod
    def _specify(v_0: float, a_0: float, v_T: np.array, w_T: np.array, w_J: np.array) -> [np.array, np.array]:
        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuarticPoly1D.time_cost_function_derivative_coefs(w_T=w_T, w_J=w_J, a_0=a_0, v_0=v_0, v_T=v_T)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, BP_ACTION_T_LIMITS)
        T = np.fmin.reduce(roots_s, axis=-1)
        s = QuarticPoly1D.distance_profile_function(a_0=a_0, v_0=v_0, v_T=v_T, T=T)(T)
        return T, s

    @staticmethod
    def _validate_acceleration(v_0: float, a_0: float, v_T: np.array, T: np.array) -> np.array:
        # validate acceleration limits of the initial quartic action
        poly = QuarticPoly1D.s_profile_coefficients(a_0, v_0, v_T, T)
        validT = ~np.isnan(T)
        valid_acc = np.zeros_like(T).astype(bool)
        valid_acc[validT] = QuarticPoly1D.are_derivatives_in_limits_zero_coef(2, poly[validT], T[validT], LON_ACC_LIMITS)
        return valid_acc

    @staticmethod
    def _create_safety_matrix(actors: List[ActorState], T: np.array, v_T: np.array, params: ScenarioParams) -> np.array:
        """
        Create safety boolean matrix of actions to the merge point that are longitudinally safe (RSS) at the merge
        point w.r.t. all actors.
        :param actors: list of actors, whose fstate[FS_SX] is relative to the merge point
        :param T: array of possible planning times
        :param v_T: array of possible target velocities
        :return: boolean matrix of size len(v_T) x len(T) of safe actions relatively to all actors
        """
        safety_matrix = np.ones((len(v_T), len(T))).astype(bool)
        for actor in actors:
            cars_margin = (EGO_LENGTH + actor.size.length) / 2 + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
            car_safe = RuleBasedLaneMergePlanner._create_safety_matrix_for_car(actor.fstate[FS_SX], actor.fstate[FS_SV],
                                                                               T, v_T, cars_margin, params)
            safety_matrix = np.logical_and(safety_matrix, car_safe)
        return safety_matrix

    @staticmethod
    def _create_safety_matrix_for_car(actor_s: float, actor_v: float, T: np.array, v_T: np.array, cars_margin: float,
                                      params: ScenarioParams) -> np.array:
        """
        Given an actor on the main road and ranges for planning times and target velocities, create boolean matrix
        of all possible actions that are longitudinally safe (RSS) at actions' endpoint (the merge point).
        :param actor_s: current s of actor relatively to the merge point (negative or positive)
        :param actor_v: current actor's velocity
        :param T: array of possible planning times
        :param v_T: array of possible target velocities
        :param cars_margin: half sum of cars' lengths + safety margin
        :param params: scenario params, e.g. cars maximal accelerations in the worst case
        :return: boolean matrix of size len(v_T) x len(T) of safe actions relatively to the given actor
        """
        td = SAFETY_HEADWAY  # reaction delay of ego
        tda = SPECIFICATION_HEADWAY  # reaction delay of actor
        a = -LON_ACC_LIMITS[0]  # maximal braking deceleration of ego & actor
        front_decel = params.worst_case_front_car_decel  # maximal braking deceleration of front actor during the merge
        back_accel = params.worst_case_back_car_accel  # maximal acceleration of back actor during the merge

        front_v = np.maximum(0, actor_v - T * front_decel)  # target velocity of the front actor
        front_T = T[actor_v > T * front_decel]
        front_T = np.concatenate((front_T, [actor_v/front_decel] * (T.shape[0] - front_T.shape[0])))
        back_v = actor_v + T * back_accel  # target velocity of the front actor
        front_s = actor_s + front_T * actor_v - 0.5 * front_decel * front_T * front_T - cars_margin  # s of front actor at time t
        back_s = actor_s + T * actor_v + 0.5 * back_accel * T * T + cars_margin  # s of back actor at time t
        front_disc = a * a * td * td + 2 * a * front_s + front_v * front_v  # discriminant for front actor
        back_disc = back_v * back_v + 2 * a * (tda * back_v + back_s)  # discriminant for back actor
        # safe_v = (v < sqrt(front_disc) - a*td and front_s > 0) or (v > sqrt(max(0, back_disc)) and back_s < 0)
        max_v_arr = (np.sqrt(np.maximum(0, front_disc)) - a * td) * (front_s > 0)  # if front_s<=0, no safe velocities
        min_v_arr = np.sqrt(np.maximum(0, back_disc)) + (v_T[-1] + RuleBasedLaneMergePlanner.VEL_GRID_RESOLUTION) * (back_s >= 0)  # if back_s>=0, no safe velocities

        max_vi = np.floor(max_v_arr / RuleBasedLaneMergePlanner.VEL_GRID_RESOLUTION).astype(int)
        min_vi = np.ceil(min_v_arr / RuleBasedLaneMergePlanner.VEL_GRID_RESOLUTION).astype(int)
        car_safe = np.zeros((len(v_T), len(T))).astype(bool)
        for ti in range(len(T)):
            car_safe[min_vi[ti]:, ti] = True
            car_safe[:max_vi[ti], ti] = True
        return car_safe

    def _create_actions(self, behavioral_state: BehavioralGridState) -> np.array:
        return self.actions

    def _filter_actions(self, behavioral_state: BehavioralGridState, actions: np.array) -> np.array:
        first_action_specs = [action[0] for action in actions]  # pick the first spec from each LaneMergeSequence
        action_specs_mask = self.action_spec_validator.filter_action_specs(first_action_specs, behavioral_state)
        filtered_action_specs = np.full(len(first_action_specs), None)
        filtered_action_specs[action_specs_mask] = first_action_specs[action_specs_mask]
        return filtered_action_specs

    def _evaluate(self, behavioral_state: BehavioralGridState, actions: np.array) -> np.array:
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
