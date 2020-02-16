from logging import Logger
from typing import List, Optional
import numpy as np
from decision_making.src.global_constants import SAFETY_HEADWAY, LON_ACC_LIMITS, EPS, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, BP_JERK_S_JERK_D_TIME_WEIGHTS, LANE_MERGE_ACTION_T_LIMITS, \
    LANE_MERGE_ACTORS_MAX_VELOCITY, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, LANE_MERGE_YIELD_BACK_ACTOR_RSS_DECEL, \
    SPEEDING_SPEED_TH, LANE_CHANGE_TIME_COMPLETION_TARGET, BP_ACTION_T_LIMITS, LONGITUDINAL_SAFETY_MARGIN_HYSTERESIS, \
    PLANNING_LOOKAHEAD_DIST, MAX_BACKWARD_HORIZON, SPECIFICATION_HEADWAY, LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT, \
    PREDICTED_ACCELERATION_TIME
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, ActionSpec, RelativeLane, \
    StaticActionRecipe, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.default_config import DEFAULT_ACTION_SPEC_FILTERING
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeState, LaneChangeStatus
from decision_making.src.planning.behavioral.state.lane_merge_actor_state import LaneMergeActorState
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.src.planning.types import BoolArray, FS_SX, FS_SA, FS_SV, C_A
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D, QuinticPoly1D
from decision_making.src.state.state import State

OUTPUT_TRAJECTORY_LENGTH = 10


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

    def to_spec(self, s0: float):
        return ActionSpec(self.t, self.t, self.v_T, s0 + self.ds, 0,
                          StaticActionRecipe(RelativeLane.SAME_LANE, self.v_T, AggressivenessLevel.CALM))

    def __str__(self):
        return 't: ' + str(self.t) + ', v_0: ' + str(self.v_0) + ', a_0: ' + str(self.a_0) + \
               ', v_T: ' + str(self.v_T) + ', ds: ' + str(self.ds) + ', coefs: ' + str(self.poly_coefs_num)


class LaneMergeSequence:
    def __init__(self, action_specs: List[LaneMergeSpec], target_rel_lane: RelativeLane):
        self.action_specs = action_specs
        self.target_rel_lane = target_rel_lane

    @property
    def t(self):
        return sum([spec.t for spec in self.action_specs])

    @property
    def v_T(self):
        return self.action_specs[-1].v_T


class LaneChangePlanner(BasePlanner):
    TIME_GRID_RESOLUTION = 1
    VEL_GRID_RESOLUTION = 5

    def __init__(self, state: State, logger: Logger):
        super().__init__(logger)

    def _create_behavioral_state(self, state: State, route_plan: RoutePlan, lane_change_state: LaneChangeState) -> LaneMergeState:
        """
        Create LaneMergeState (which inherits from BehavioralGridState) from the given state
        :param state: state from scene dynamic
        :param route_plan: the route plan
        :param lane_change_state: not in use in this class
        :return: LaneMergeState
        """
        return LaneMergeState.create_from_state(state, route_plan, lane_change_state, self.logger)

    def _create_action_specs(self, lane_merge_state: LaneMergeState, route_plan: RoutePlan) -> np.array:
        """
        Create all possible actions to the merge point, filter unsafe actions, filter actions exceeding vel-acc limits,
        calculate time-jerk cost for the remaining actions.
        :param lane_merge_state: LaneMergeState containing distance to merge and actors
        :return: list of initial action specs, array of jerks until the merge, array of times until the merge
        """
        import time
        st_tot = time.time()

        t_res = LaneChangePlanner.TIME_GRID_RESOLUTION
        ego_time = lane_merge_state.ego_state.timestamp_in_sec
        time_fraction = (np.floor(ego_time/t_res) + 1) * t_res - ego_time
        t_grid = np.arange(time_fraction, LANE_MERGE_ACTION_T_LIMITS[1] + EPS, t_res)
        v_res = LaneChangePlanner.VEL_GRID_RESOLUTION
        v_grid = np.arange(0, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY + EPS, v_res)
        # fine_v_res = v_res / 8
        # ego_v = np.floor(lane_merge_state.ego_state.velocity / v_res) * v_res
        # fine_v_grid = np.arange(ego_v + fine_v_res, ego_v + v_res - EPS, fine_v_res)
        # v_grid = np.append(v_grid, fine_v_grid)
        # v_grid = np.sort(v_grid)

        target_v, target_t = np.meshgrid(v_grid, t_grid)
        target_v, target_t = target_v.ravel(), target_t.ravel()

        safety_bounds = LaneChangePlanner._calculate_safety_bounds(lane_merge_state, target_v, target_t)

        # create single-spec quintic safe actions
        single_actions = LaneChangePlanner._create_single_actions(lane_merge_state, target_v, target_t, safety_bounds)

        # create composite "max_velocity" actions, such that each sequence includes quartic + const_max_vel + quartic
        #max_vel_actions = RuleBasedLaneMergePlanner._create_max_vel_actions(lane_merge_state, target_v, target_t, safety_bounds)

        # create composite "stop" actions, such that each sequence includes quartic + zero_vel + quartic
        # create stop actions only if there are no single-spec actions, since quintic actions have lower time
        # and lower jerk than composite stop action
        # brake_actions = []
        # if len(single_actions) == 0:
        #     brake_actions = LaneChangePlanner._create_braking_actions(lane_merge_state, target_v, target_t, safety_bounds)

        lane_merge_actions = single_actions  # + brake_actions + max_vel_actions

        print('# single_actions:', len(single_actions))  # , 'brake actions:', len(brake_actions))

        # print('\ntime = %f: quintic=(%d) quartic=(%d)' % (time.time() - st_tot, len(single_actions), len(max_vel_actions)))
        return np.array(lane_merge_actions)

    def _filter_actions(self, lane_merge_state: LaneMergeState, actions: np.array) -> np.array:
        """
        Currently do nothing. Safety, acc & vel limits are tested inside _create_action_specs.
        :param lane_merge_state: lane merge state
        :param actions: lane merge action sequences
        :return: array of actions of the same size as the input, but filtered actions are None
        """
        s_0 = lane_merge_state.ego_fstate_1d[FS_SX]
        action_specs = np.array([action.action_specs[0].to_spec(s_0) for action in actions])
        action_specs_mask = DEFAULT_ACTION_SPEC_FILTERING.filter_action_specs(action_specs, lane_merge_state)
        filtered_actions = np.full(len(actions), None)
        filtered_actions[action_specs_mask] = actions[action_specs_mask]
        return filtered_actions

    def _evaluate_actions(self, lane_merge_state: LaneMergeState, route_plan: RoutePlan, actions: np.array) -> np.array:
        """
        Evaluates Action-Specifications with the lowest aggressiveness possible
        :param lane_merge_state: lane merge state
        :param route_plan:
        :param actions: specifications of action_recipes.
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        action_mask = actions.astype(bool)
        if not action_mask.any():
            return np.array([])

        # calculate full actions' jerks and times
        actions_jerks = np.zeros(len(actions))
        actions_times = np.zeros(len(actions))
        actions_dists = np.zeros(len(actions))
        accel_times = np.zeros(len(actions))
        accel_dists = np.zeros(len(actions))
        calm_weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]

        for poly1d in [QuinticPoly1D, QuarticPoly1D]:
            specs_list = [[idx, spec.t, spec.v_0, spec.a_0, spec.v_T, spec.ds]
                          for idx, action in enumerate(actions) if action is not None
                          for spec in action.action_specs if spec.poly_coefs_num == poly1d.num_coefs() and spec.t > 0]
            if len(specs_list) == 0:
                continue

            # calculate jerks for the specs
            specs_matrix = np.array(list(filter(None, specs_list)))
            action_idxs, T, v_0, a_0, v_T, ds = specs_matrix.T
            poly_coefs = QuinticPoly1D.position_profile_coefficients(a_0, v_0, v_T, ds, T) \
                if poly1d is QuinticPoly1D else QuarticPoly1D.position_profile_coefficients(a_0, v_0, v_T, T)
            spec_jerks = poly1d.cumulative_jerk(poly_coefs, T)

            # calculate acceleration times from v_T to max velocity
            accel_to_max_vel_s, accel_to_max_vel_T, _ = KinematicUtils.specify_quartic_actions(
                calm_weights[2], calm_weights[0], v_0=v_T, v_T=LANE_MERGE_ACTION_SPACE_MAX_VELOCITY,
                action_horizon_limit=np.inf)

            # collect actions jerks, times and distances
            for action_idx, spec_jerk, spec_t, spec_v, spec_s, acc_t, acc_s in \
                    zip(action_idxs.astype(int), spec_jerks, T, v_T, ds, accel_to_max_vel_T, accel_to_max_vel_s):
                actions_jerks[action_idx] += spec_jerk
                actions_times[action_idx] += spec_t
                actions_dists[action_idx] += spec_s
                accel_times[action_idx] += acc_t
                accel_dists[action_idx] += acc_s

        # calculate actions' costs according to the AGGRESSIVE jerk-time weights
        jerk_w, _, time_w = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.AGGRESSIVE.value]
        # prefer longer actions before passing to the left lane and shorter actions before passing to the right lane
        left_lane_weight = -lane_merge_state.target_rel_lane.value * time_w / 8
        # assume that after acceleration to max_vel it will continue to max_dist with max_vel
        max_dist = np.max(actions_dists)
        full_times = actions_times + accel_times + (max_dist - actions_dists - accel_dists) / LANE_MERGE_ACTION_SPACE_MAX_VELOCITY

        action_costs = jerk_w * actions_jerks + time_w * full_times + left_lane_weight * actions_times
        action_costs[~action_mask] = np.inf

        # times = np.array([ac.t if ac is not None else np.inf for ac in actions])
        # fastest_idx = np.argmin(times)
        # chosen_idx = np.argmin(action_costs)
        # print('RB.evaluator: fastest_idx=', fastest_idx, 'time[fastest] =', times[fastest_idx], 'full_time[fastest]=', full_times[fastest_idx],
        #       'chosen_idx=', chosen_idx, 'time[chosen]=', times[chosen_idx], 'full_time[chosen]=', full_times[chosen_idx])

        return action_costs

    def _choose_action(self, lane_merge_state: LaneMergeState, actions: np.array, costs: np.array) -> ActionSpec:
        """
        Return action_spec having the minimal cost
        :param lane_merge_state: lane merge state
        :param actions: array of ActionSpecs
        :param costs: array of actions' costs
        :return: action specification with minimal cost
        """
        chosen_action = actions[np.argmin(costs)]
        action_time = chosen_action.t
        chosen_spec = chosen_action.action_specs[0].to_spec(lane_merge_state.ego_fstate_1d[FS_SX])

        # if target time is too soon then stop the lane merge by performing lane change
        if lane_merge_state.ego_fstate_1d[FS_SX] > lane_merge_state.merge_from_s_on_ego_gff and \
                action_time <= LaneChangePlanner.TIME_GRID_RESOLUTION:
            chosen_spec.recipe.relative_lane = lane_merge_state.target_rel_lane
            chosen_spec.t = LANE_CHANGE_TIME_COMPLETION_TARGET
            chosen_spec.s = lane_merge_state.projected_ego_fstates[lane_merge_state.target_rel_lane][FS_SX] + \
                            chosen_spec.t * lane_merge_state.ego_state.velocity
            lane_merge_state.lane_change_state.status = LaneChangeStatus.ANALYZING_SAFETY
            lane_merge_state.lane_change_state.lane_change_start_time = lane_merge_state.ego_state.timestamp_in_sec
            lane_merge_state.lane_change_state.autonomous_mode = True

        return chosen_spec

    @staticmethod
    def _create_single_actions(state: LaneMergeState, target_v: np.array, target_t: np.array, safety_bounds: np.array) \
            -> List[LaneMergeSequence]:

        vts, _ = LaneChangePlanner._calculate_safe_target_points(
            state.ego_fstate_1d[FS_SV], state.ego_fstate_1d[FS_SA], target_v, target_t, safety_bounds)
        if vts.shape[0] == 0:
            return []
        v_T, T, ds = vts.T

        # calculate s_profile coefficients for all actions
        ego_fstate = state.ego_fstate_1d
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]

        # filter actions violating velocity or acceleration limits
        valid_idxs = LaneChangePlanner._validate_vel_acc_limits(a_0, v_0, v_T, ds, T)

        # if there are no quintic actions but there are rear safe gaps, then brake (stop)
        if len(valid_idxs) == 0:
            brake_actions = (v_0 * T > ds)
            if brake_actions.any():  # then stop
                T_brake, s_brake = LaneChangePlanner._create_quartic_actions(v_0, a_0, vT=0, A0=LON_ACC_LIMITS[0], J=5)
                return [LaneMergeSequence([LaneMergeSpec(T_brake, v_0, a_0, 0, s_brake, QuarticPoly1D.num_coefs())], state.target_rel_lane)]
            else:
                return []

        actions = [LaneMergeSequence([LaneMergeSpec(t, v_0, a_0, vT, s, QuinticPoly1D.num_coefs())], state.target_rel_lane)
                   for t, vT, s in zip(T[valid_idxs], v_T[valid_idxs], ds[valid_idxs])]
        return actions

    @staticmethod
    def _create_triple_cubic_actions(v0: np.array, a0: float, vT: np.array, a0_limit: float, aT_limits: np.array,
                                     J_max: float) -> [np.array, np.array]:
        """
        Perform 3 cubic actions:
            constant jerk J_max and acceleration a0 -> a0_limit from velocity v0
            constant jerk and acceleration a0_limit -> aT_limit,
            constant jerk -J_max acceleration aT_limit -> 0 (to zero acceleration) and velocity vT
        :param v0: initial velocity
        :param a0: initial acceleration
        :param vT: array of target velocities
        :param a0_limit: signed: negative for deceleration; acceleration limit at v0
        :param aT_limits: signed: negative for deceleration; array of acceleration limits for all vT
        :param J_max: unsigned: maximal jerk (absolute value)
        :return: array of full times of the triple actions and array of full distances
        """
        action_sgn = np.sign(a0_limit)

        # cubic action #1: acceleration from a0 to a0_limit with constant jerk J_max
        T_to_a0_limit = np.full(vT.shape, (a0 - a0_limit) / J_max) if not np.isscalar(vT) else (a0 - a0_limit) / J_max
        s_to_a0_limit = v0 * T_to_a0_limit + 0.5 * a0 * T_to_a0_limit ** 2 - J_max * T_to_a0_limit ** 3 / 6
        v_at_a0_limit = v0 + a0 * T_to_a0_limit - 0.5 * J_max * T_to_a0_limit ** 2
        if not np.isscalar(T_to_a0_limit):
            T_to_a0_limit[(T_to_a0_limit < 0) | (action_sgn * (vT - v_at_a0_limit) < 0)] = np.nan
        elif (T_to_a0_limit < 0) or (action_sgn * (vT - v_at_a0_limit) < 0):
            T_to_a0_limit = np.nan

        # cubic action #3: acceleration from aT_limit to 0 with constant jerk -J_max
        T_to_zero_acc = -aT_limits / J_max
        v_at_aT_limit = vT - 0.5 * aT_limits * T_to_zero_acc
        if not np.isscalar(T_to_zero_acc):
            T_to_zero_acc[(T_to_zero_acc < 0) | (action_sgn * (v_at_aT_limit - v_at_a0_limit) < 0)] = np.nan
        elif (T_to_zero_acc < 0) or (action_sgn * (v_at_aT_limit - v_at_a0_limit) < 0):
            T_to_zero_acc = np.nan
        s_to_zero_acc = v_at_aT_limit * T_to_zero_acc + 0.5 * aT_limits * T_to_zero_acc ** 2 + J_max * T_to_zero_acc ** 3 / 6

        # cubic action #2: acceleration from a0_limit to aT_limit with constant jerk (J_to_aT_limit)
        T_to_aT_limit = 2 * (v_at_aT_limit - v_at_a0_limit) / (a0_limit + aT_limits)
        J_to_aT_limit = 0.5 * (aT_limits ** 2 - a0_limit ** 2) / (v_at_aT_limit - v_at_a0_limit)
        s_to_aT_limit = v_at_a0_limit * T_to_aT_limit + 0.5 * a0_limit * T_to_aT_limit ** 2 + J_to_aT_limit * T_to_aT_limit ** 3 / 6

        # In order to compute polynomials for each cubic action use QuarticPoly1D.solve(A_inv, constraints)

        return T_to_a0_limit + T_to_aT_limit + T_to_zero_acc, s_to_a0_limit + s_to_aT_limit + s_to_zero_acc

    @staticmethod
    def _create_quartic_actions(v0: np.array, a0: float, vT: np.array, A0: float, J: float) -> [np.array, np.array]:
        """
        create the shortest quartic actions limited by acceleration and jerk
        :param v0:
        :param a0:
        :param vT: array of target velocities
        :param A0: signed; longitudinal acceleration limit
        :param J: positive; jerk limit
        :return:
        """
        # limit action time by acceleration            
        if a0 != 0:
            T_acc = 3 * (vT - v0) * (a0 + A0 - np.sign(A0) * np.sqrt(A0 * (A0 - a0))) / (a0 * (a0 + 3 * A0))
        else:
            T_acc = 1.5 * (vT - v0) / A0

        # limit action time by jerk            
        JdV = 6 * J * (v0 - vT)
        t0_neg_J = (2 * a0 + np.sqrt(4 * a0 * a0 + JdV)) / J
        t0_pos_J = (-2 * a0 + np.sqrt(4 * a0 * a0 - JdV)) / J
        tT_neg_J = (-a0 + np.sqrt(a0 * a0 - JdV)) / J
        tT_pos_J = (a0 + np.sqrt(a0 * a0 + JdV)) / J

        # pick the maximal time        
        t0 = np.maximum(np.nan_to_num(t0_neg_J), np.nan_to_num(t0_pos_J))
        tT = np.maximum(np.nan_to_num(tT_neg_J), np.nan_to_num(tT_pos_J))
        T = np.maximum(np.maximum(tT, t0), np.nan_to_num(T_acc))
        
        s = QuarticPoly1D.distance_profile_function(a0, v0, vT, T)(T)
        return T, s

    # @staticmethod
    # def _create_max_vel_actions(state: LaneMergeState, target_v: np.array, target_t: np.array, safety_bounds: np.array) \
    #         -> List[LaneMergeSequence]:
    #     """
    #     Given array of final velocities, create composite safe actions:
    #         1. quartic CALM acceleration to v_max,
    #         2. constant velocity with v_max,
    #         3. quartic STANDARD deceleration to v_T.
    #     :param state: lane merge state
    #     :param target_v: array of final velocities
    #     :param target_t: array of planning times
    #     :return: list of safe composite actions
    #     """
    #     v_max = LANE_MERGE_ACTION_SPACE_MAX_VELOCITY
    #     ego_fstate = state.ego_fstate_1d
    #     v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
    #
    #     reaction_delay = 0.5
    #     v_ext = v_0 + a_0 * reaction_delay
    #     s_ext = v_0 * reaction_delay + 0.5 * a_0 * reaction_delay**2
    #
    #     # quartic action with acceleration peak = LON_ACC_LIMITS[1]
    #     a_max = LON_ACC_LIMITS[1] * 0.9  # decrease because of TP
    #     a0 = EPS if abs(a_0) < EPS else min(a_0, a_max)
    #     t1 = 3*(v_max - v_ext) * (a0 + a_max - np.sqrt(a_max*(a_max - a0))) / (a0*(a0 + 3*a_max))
    #     # t1 = 3*(v_max - v_0_ext) / (2*amax)  # for a_0 = 0
    #     s_profile_coefs = QuarticPoly1D.position_profile_coefficients(a_0, v_ext, v_max, np.array([t1]))
    #     s1 = Math.zip_polyval2d(s_profile_coefs, np.array([t1])[:, np.newaxis])[0, 0]
    #
    #     # create future state when ego will accelerate to the maximal velocity
    #     future_target_v = target_v[target_t > t1 + reaction_delay]
    #     future_target_t = target_t[target_t > t1 + reaction_delay] - t1 - reaction_delay
    #     future_bounds = safety_bounds[target_t > t1 + reaction_delay] - s1 - s_ext
    #
    #     # calculate candidate safe end-points of quintic actions starting from max_vel_state
    #     vts, _ = LaneChangePlanner._calculate_safe_target_points(v_max, 0, future_target_v, future_target_t, future_bounds)
    #     if vts.shape[0] == 0:
    #         return []
    #     v_T, T, ds = vts.T
    #
    #     # filter actions violating velocity or acceleration limits
    #     valid_idxs = LaneChangePlanner._validate_vel_acc_limits(0, v_max, v_T, ds, T)
    #
    #     actions = []
    #     for t2, s2, v_t in zip(T[valid_idxs], ds[valid_idxs], v_T[valid_idxs]):
    #         action1 = LaneMergeSpec(t1 + reaction_delay, v_0, a_0, v_max, s1 + s_ext, QuinticPoly1D.num_coefs())
    #         action2 = LaneMergeSpec(t2, v_max, 0, v_t, s2, QuinticPoly1D.num_coefs())
    #         actions.append(LaneMergeSequence([action1, action2], state.target_rel_lane))
    #
    #     return actions

    # @staticmethod
    # def _create_braking_actions(state: LaneMergeState, target_v: np.array, target_t: np.array, safety_bounds: np.array) \
    #         -> List[LaneMergeSequence]:
    #
    #     ego_fstate = state.ego_fstate_1d
    #     v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
    #
    #     vts, target_idxs = LaneChangePlanner._calculate_safe_target_points(
    #         state.ego_fstate_1d[FS_SV], state.ego_fstate_1d[FS_SA], target_v, target_t, safety_bounds)
    #     if vts.shape[0] == 0:
    #         return []
    #     v_T, T, ds = vts.T
    #     brake_actions = (v_0 * T > ds)
    #     v_T, T, ds = v_T[brake_actions], T[brake_actions], ds[brake_actions]
    #     safety_bounds = safety_bounds[target_idxs[brake_actions]]
    #
    #     vS = np.flip(np.arange(0, v_0, 0.2))
    #     vS_mesh, vT_mesh = np.meshgrid(vS, v_T)
    #     _, T_mesh = np.meshgrid(vS, T)
    #     _, s_mesh = np.meshgrid(vS, ds)
    #
    #     # T_dec, s_dec = LaneChangePlanner._create_triple_cubic_actions(v_0, a_0, vT=vS, A0=-4, AT=np.array([-5]), J=5)
    #     # T_acc, s_acc = LaneChangePlanner._create_triple_cubic_actions(vS, 0, v_T, A0=3, AT=np.array([2]), J=5)
    #     # quartic_dec = np.isnan(T_dec)
    #     # quartic_acc = np.isnan(T_acc)
    #     # T_dec[quartic_dec], s_dec[quartic_dec] = LaneChangePlanner._create_quartic_actions(v_0, a_0, vT=vS[quartic_dec], A0=-4, J=5)
    #     # T_acc[quartic_acc], s_acc[quartic_acc] = LaneChangePlanner._create_quartic_actions(vS[quartic_acc], 0, v_T[quartic_acc], A0=3, J=5)
    #
    #     T_dec, s_dec = LaneChangePlanner._create_quartic_actions(v_0, a_0, vT=vS, A0=-4, J=5)
    #     T_dec_mesh = np.full(vT_mesh.shape, T_dec)
    #     s_dec_mesh = np.full(vT_mesh.shape, s_dec)
    #     T_acc_mesh, s_acc_mesh = LaneChangePlanner._create_quartic_actions(vS_mesh, 0, vT_mesh, A0=3, J=5)
    #     long_enough_actions = (T_dec_mesh + T_acc_mesh > T_mesh) & (s_dec_mesh + s_acc_mesh < s_mesh)
    #     vS_idx = np.argmax(long_enough_actions, axis=1)
    #     s_action = s_dec[vS_idx] + s_acc_mesh[range(s_acc_mesh.shape[0]), vS_idx]
    #
    #     # s_action is safe if number of important bounds under optimal_s is even
    #     action_is_safe = (np.sum(safety_bounds < s_action[:, np.newaxis], axis=-1) & 1) == 0
    #     v_T, T, ds = v_T[action_is_safe], T[action_is_safe], ds[action_is_safe]

    # @staticmethod
    # def _create_braking_actions(state: LaneMergeState, target_v: np.array, target_t: np.array, bounds: np.array) -> \
    #         List[LaneMergeSequence]:
    # 
    #     v_0, a_0 = state.ego_fstate_1d[FS_SV], state.ego_fstate_1d[FS_SA]
    # 
    #     cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
    #     front_car = state.road_occupancy_grid[cell][0] if cell in state.road_occupancy_grid and len(state.road_occupancy_grid[cell]) > 0 else None
    #     if front_car is None and ~np.isfinite(state.red_line_s_on_ego_gff):  # no need to brake
    #         return []
    # 
    #     v_slow = front_car.dynamic_object.velocity if front_car is not None else 0
    #     bounds = bounds[target_v > v_slow]
    #     target_t = target_t[target_v > v_slow]
    #     target_v = target_v[target_v > v_slow]
    # 
    #     # quartic action with acceleration peak = LON_ACC_LIMITS[1]
    #     a_max = LON_ACC_LIMITS[1] * 0.9  # decrease because of TP
    #     t_acc = 3*(target_v - v_slow) / (2*a_max)  # quartic acceleration for a_0 = 0
    #     s_profile_coefs = QuarticPoly1D.position_profile_coefficients(0, v_slow, target_v, t_acc)
    #     s_acc = Math.zip_polyval2d(s_profile_coefs, t_acc[:, np.newaxis])[:, 0]
    # 
    #     # specify aggressive braking (used if there is no front car but red line)
    #     w_J_agg, _, w_T_agg = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.AGGRESSIVE.value]
    #     s_aggr_brake, t_aggr_brake = KinematicUtils.specify_quartic_action(w_T_agg, w_J_agg, v_0, v_slow, a_0)
    # 
    #     if front_car is None:
    #         t_slow = target_t - t_aggr_brake - t_acc
    #         s_slow = t_slow * v_slow
    #         safe_target_s = s_aggr_brake + s_slow + s_acc
    #     else:
    #         margin = 0.5 * (state.ego_length + front_car.dynamic_object.size.length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
    #         # safe_dist = RSS distance + half lane change
    #         safe_dist = margin + SAFETY_HEADWAY * target_v + (target_v*target_v - v_slow*v_slow)/(-2*LON_ACC_LIMITS[0]) + \
    #                     0.5 * LANE_CHANGE_TIME_COMPLETION_TARGET * (target_v - v_slow)
    #         # for each target t,v find the largest s safe w.r.t. the front car
    #         safe_target_s = front_car.longitudinal_distance + target_t * v_slow - safe_dist
    # 
    #     actions = []
    #     for bnd in range(0, bounds.shape[1], 2):
    #         # find safety_bounds that are strictly safe w.r.t. the front car
    #         valid = np.isfinite(bounds[:, bnd]) & (t_acc <= target_t) & \
    #                 (np.abs(bounds[:, bnd] - safe_target_s) < LaneChangePlanner.TIME_GRID_RESOLUTION * target_v)
    #         valid_idxs = np.where(valid)[0]
    #         if len(valid_idxs) == 0:
    #             continue
    # 
    #         if front_car is not None:
    #             # assume that host arrives to the target, when it's safe_dist behind the front car
    #             dx = front_car.longitudinal_distance - safe_dist[valid_idxs] + t_acc[valid_idxs] * v_slow - s_acc[valid_idxs]
    #             weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.STANDARD.value]
    #             w_J, _, w_T = np.full((valid_idxs.shape[0], weights.shape[0]), weights).T
    #             cost_coeffs_s = QuinticPoly1D.time_cost_function_derivative_coefs(w_T, w_J, a_0, v_0, v_T=v_slow, dx=dx, T_m=0)
    #             roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, BP_ACTION_T_LIMITS)
    #             t_brake = np.fmin.reduce(roots_s, axis=-1)
    #             s_brake = dx + t_brake * v_slow
    #         else:
    #             t_brake = np.full(len(valid_idxs), t_aggr_brake)
    #             s_brake = np.full(len(valid_idxs), s_aggr_brake)
    # 
    #         valid = np.isfinite(t_brake) & (s_brake >= 0) & (t_acc[valid_idxs] + t_brake <= target_t[valid_idxs])
    #         orig_valid_idxs = valid_idxs[valid]
    #         valid_of_valid_idxs = np.where(valid)[0]
    # 
    #         # calculate constant velocity actions
    #         t_slow = target_t[valid_idxs] - t_brake - t_acc[valid_idxs]
    #         s_slow = t_slow * v_slow
    # 
    #         for idx, oidx in zip(valid_of_valid_idxs, orig_valid_idxs):
    #             action1 = LaneMergeSpec(t_brake[idx], v_0, a_0, v_slow, s_brake[idx], QuinticPoly1D.num_coefs())
    #             action2 = LaneMergeSpec(t_slow[idx], v_slow, 0, v_slow, s_slow[idx], poly_coefs_num=2)
    #             action3 = LaneMergeSpec(t_acc[oidx], v_slow, 0, target_v[oidx], s_acc[oidx], QuarticPoly1D.num_coefs())
    #             actions.append(LaneMergeSequence([action1, action2, action3], state.target_rel_lane))
    # 
    #     return actions

    @staticmethod
    def _calculate_safety_bounds(state: LaneMergeState, target_v: np.array, target_t: np.array) -> np.array:
        s_min = max(0, state.merge_from_s_on_ego_gff - state.ego_fstate_1d[FS_SX])
        s_max = state.red_line_s_on_ego_gff - state.ego_fstate_1d[FS_SX]

        # add the front actor to actors_states from the target lane
        actors_states = state.actors_states

        # insert dummy back actor at PLANNING_LOOKAHEAD_DIST behind ego
        if len(actors_states) > 0:
            back_v = actors_states[np.argmin(np.array([actor.s_relative_to_ego for actor in actors_states]))].velocity
            actors_states.insert(0, LaneMergeActorState(-PLANNING_LOOKAHEAD_DIST, back_v, acceleration=0, length=0))

        cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        front_car = state.road_occupancy_grid[cell][0] if cell in state.road_occupancy_grid and len(state.road_occupancy_grid[cell]) > 0 else None
        if front_car is not None:
            actors_states.append(LaneMergeActorState(front_car.longitudinal_distance, front_car.dynamic_object.velocity,
                                                     front_car.dynamic_object.cartesian_state[C_A],
                                                     front_car.dynamic_object.size.length))
        if len(actors_states) == 0:
            return None

        # # TODO: delete it !
        # actors_states = []
        # for i in range(6):
        #     actor = LaneMergeActorState(-i * 20 + 5, state.actors_states[0].velocity, 4)
        #     actors_states.append(actor)
        # actors_states.insert(0, LaneMergeActorState(-PLANNING_LOOKAHEAD_DIST, state.actors_states[0].velocity, 0))
        # front_v = 30/3.6
        # actors_states.append(LaneMergeActorState(30, front_v, 4))
        # state.ego_state.cartesian_state[3] = front_v
        # state.projected_ego_fstates[RelativeLane.SAME_LANE][FS_SV] = front_v

        actors_s, actors_v, actors_a, actors_length = \
            np.array([[actor.s_relative_to_ego, actor.velocity, actor.acceleration, actor.length]
                     for actor in actors_states]).T
        # add extra margin for longer actions to enable comfort short actions and another extra margin for smooth
        # switching with the single_step_planner
        margins = 0.5 * (actors_length + state.ego_length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + \
                  LONGITUDINAL_SAFETY_MARGIN_HYSTERESIS + 2  # + 2 * (target_t[:, np.newaxis] > 2)

        # actors_s.sort()
        # actor_i = np.sum(actors_s < 0)
        sorted_s = np.sort(actors_s if front_car is None else actors_s[:-1])
        print('s_min=', s_min, 's_max=', s_max, 'actors_rel_s=', sorted_s, 'gaps=', np.diff(sorted_s))

        # calculate planning time bounds given target_s
        # 2D matrix with 3 columns of safe actions: target velocities, planning times and distances
        front_bounds, back_bounds = LaneChangePlanner._caclulate_RSS_bounds(
            actors_s, actors_v, actors_a, margins, target_v[:, np.newaxis], target_t[:, np.newaxis])

        # ego can't be behind the dummy back actor (first in actors_states)
        front_bounds[:, 0] = -np.inf
        # ego can't be ahead of the front car (last in actors_states)
        if front_car is not None:
            back_bounds[:, -1] = np.inf
            # verify safety wrt front_car until lateral safety
            time_between_target_lane_touch_and_lateral_safety_wrt_front = 1.5
            front_bounds[:, -1] -= np.maximum(0, target_v - actors_v[-1]) * time_between_target_lane_touch_and_lateral_safety_wrt_front

        # concatenate s_min & s_max to front and back bounds
        front_bounds = np.c_[front_bounds, np.full(front_bounds.shape[0], -np.inf), np.full(front_bounds.shape[0], s_max)]
        back_bounds = np.c_[back_bounds, np.full(back_bounds.shape[0], s_min), np.full(back_bounds.shape[0], np.inf)]

        # the safe regions are either behind front_bounds or ahead of back_bounds
        bounds_s = np.concatenate((front_bounds, back_bounds), axis=1)

        # for each target [v,t] sort its bounds by s
        sorted_idxs = bounds_s.argsort(axis=1)
        important_bounds = np.sort(bounds_s, axis=1)
        sorted_bounds = important_bounds[..., np.newaxis]

        # append column of 1 to the front bounds and -1 to the back bounds
        signs = -np.sign(sorted_idxs - front_bounds.shape[1] + 0.5)[..., np.newaxis]
        sorted_bounds = np.concatenate((sorted_bounds, np.ones_like(sorted_bounds) * signs), axis=-1)
        # for each bound calculate how many actors make it unsafe
        safety_layer = np.cumsum(sorted_bounds[..., 1], axis=-1)

        # a bound is defined important if it changes the safety:
        #       front bound is important if safety_layer passes from 0 to 1
        #       back bound is important if safety layer passes from 1 to 0
        are_important_bounds = ((sorted_bounds[..., 1] == 1) & (safety_layer == 1)) | \
                               ((sorted_bounds[..., 1] == -1) & (safety_layer == 0))
        important_bounds = sorted_bounds[..., 0]
        important_bounds[~are_important_bounds] = np.inf
        important_bounds.sort(axis=1)
        return important_bounds

    @staticmethod
    def _calculate_safe_target_points(v_0: float, a_0: float, target_v: np.array, target_t: np.array,
                                      safety_bounds: np.array) -> [np.array, np.array]:
        """
        Create boolean 2D matrix of actions that are longitudinally safe (RSS) between red line & merge point w.r.t.
        all actors.
        :return: two 2D arrays of [v_T, T, s], where ego is safe at target_s relatively to all actors
        """
        # calculate jerk-optimal target s according to the QUARTIC distance formula
        optimal_s = target_t * (target_t * a_0 + 6 * v_0 + 6 * target_v) / 12

        # find actions for which optimal_s is safe
        if safety_bounds is None:
            return np.c_[target_v, target_t, optimal_s], np.arange(target_t.shape[0])

        above_optimal_bound_idxs = np.sum(safety_bounds < optimal_s[:, np.newaxis], axis=-1)
        # optimal_s is safe if number of important bounds under optimal_s is even
        optimal_s_is_safe = (above_optimal_bound_idxs & 1) == 0

        # calculate important bounds under & above optimal_s for actions with unsafe optimal_s
        unsafe_optimal_s_idxs = np.where(~optimal_s_is_safe)[0]
        target_v_unsafe_optimal, target_t_unsafe_optimal = target_v[unsafe_optimal_s_idxs], target_t[unsafe_optimal_s_idxs]
        above_unsafe_optimal_bound_idxs = above_optimal_bound_idxs[unsafe_optimal_s_idxs]

        # calculate all safe target points for optimal_s (quartic) in 3D space (v_T, T, ds)
        vts_optimal = np.c_[target_v[optimal_s_is_safe], target_t[optimal_s_is_safe], optimal_s[optimal_s_is_safe]]
        # calculate all safe points for a bound under optimal_s
        vts_under_optimal, under_idxs = LaneChangePlanner._calculate_points_for_safe_bounds(
            safety_bounds[unsafe_optimal_s_idxs, above_unsafe_optimal_bound_idxs-1],
            target_v_unsafe_optimal, target_t_unsafe_optimal)
        # calculate all safe points for a bound above optimal_s
        vts_above_optimal, above_idxs = LaneChangePlanner._calculate_points_for_safe_bounds(
            safety_bounds[unsafe_optimal_s_idxs, above_unsafe_optimal_bound_idxs],
            target_v_unsafe_optimal, target_t_unsafe_optimal)
        vts = np.concatenate((vts_optimal, vts_under_optimal, vts_above_optimal), axis=0)
        idxs = np.concatenate((np.where(optimal_s_is_safe)[0], unsafe_optimal_s_idxs[under_idxs], unsafe_optimal_s_idxs[above_idxs]))
        return vts, idxs

    @staticmethod
    def _calculate_points_for_safe_bounds(bounds: np.array, target_v: np.array, target_t: np.array):
        """
        given target times and velocities and safety bounds for these actions, create the appropriate actions (v_T, T, ds)
        :param bounds: array of indices of bounds for each action
        :param target_v: target velocities
        :param target_t: target times
        :return: 2D matrix of actions: (v_T, T, ds)
        """
        actions_with_valid_bounds = np.where(np.isfinite(bounds))[0]
        return np.c_[target_v[actions_with_valid_bounds], target_t[actions_with_valid_bounds], bounds[actions_with_valid_bounds]], \
               actions_with_valid_bounds

    @staticmethod
    def _validate_vel_acc_limits(a_0: float, v_0: float, v_T: np.array, ds: np.array, T: np.array) -> np.array:
        poly_coefs = QuinticPoly1D.position_profile_coefficients(a_0, v_0, v_T, ds, T)
        # the fast analytic kinematic filter reduce the load on the regular kinematic filter
        # calculate actions that don't violate acceleration limits
        valid_acc = QuinticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS * 0.9)
        if not valid_acc.any():
            return np.array([]).astype(int)
        velocity_limits = np.array([0, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY + SPEEDING_SPEED_TH])
        valid_vel = QuinticPoly1D.are_velocities_in_limits(poly_coefs[valid_acc], T[valid_acc], velocity_limits)
        return np.where(valid_acc)[0][valid_vel]

    @staticmethod
    def _caclulate_RSS_bounds(actors_s: np.array, actors_v: np.array, actors_a: np.array, margins: np.array,
                              target_v: np.array, target_t: np.array) -> [np.array, np.array]:
        """
        Given actors on the main road and actions (planning times and target velocities), create two 2D matrices
        of front and back bounds per actor and per action (target_v & target_t).
        A front bound means that some target s is safe w.r.t. the actor if it's behind the bound for the given actor
        and the given action.
        A back bound means that some target s is safe w.r.t. the actor if it's ahead the bound for the given actor
        and the given action.
        :param actors_s: current s of actor relatively to the merge point (negative or positive)
        :param actors_v: current actor's velocity
        :param actors_a: current actor's acceleration
        :param margins: half sum of cars' lengths + safety margin
        :param target_t: array of planning times
        :param target_v: array of target velocities, of the same shape as target_t
        :return: two 2D matrices of shape: actions_num x actors_num.
        """
        front_acc_time = np.minimum(target_t, PREDICTED_ACCELERATION_TIME)
        front_acc_time[:, actors_a < 0] = np.minimum(actors_v[actors_a < 0] / -actors_a[actors_a < 0],
                                                     np.minimum(target_t, PREDICTED_ACCELERATION_TIME))
        front_v = actors_v + front_acc_time * actors_a  # target velocity of the front actor
        front_s = actors_s + 0.5 * (actors_v + front_v) * front_acc_time + (target_t - front_acc_time) * front_v

        back_acc_time = np.empty((target_t.shape[0], actors_v.shape[0]))
        back_acc_time[:, actors_a > 0] = np.minimum((LANE_MERGE_ACTORS_MAX_VELOCITY - actors_v[actors_a > 0]) / actors_a[actors_a > 0],
                                                    np.minimum(target_t, PREDICTED_ACCELERATION_TIME))
        back_v = actors_v + back_acc_time * actors_a  # target velocity of the back actor
        back_s = actors_s + 0.5 * (actors_v + back_v) * back_acc_time + (target_t - back_acc_time) * back_v

        # calculate target_s bounds according to the longitudinal RSS formula
        front_bounds = front_s - margins - np.maximum(
            0, (target_v * target_v - front_v * front_v) / (-2 * LON_ACC_LIMITS[0]) + SAFETY_HEADWAY * target_v)
        back_bounds = back_s + margins + np.maximum(
            0, (back_v * back_v - target_v * target_v) / (2 * LANE_MERGE_YIELD_BACK_ACTOR_RSS_DECEL) + SAFETY_HEADWAY * back_v)
        return front_bounds, back_bounds

    @staticmethod
    def calculate_margin_to_keep_from_front(state: LaneMergeState) -> Optional[float]:
        cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        front_car = state.road_occupancy_grid[cell][0] if cell in state.road_occupancy_grid and len(state.road_occupancy_grid[cell]) > 0 else None
        rear_actors = [actor for actor in state.actors_states if actor.s_relative_to_ego < 0]
        if front_car is None or len(rear_actors) == 0:
            return None

        # get average rear actors' velocity
        target_v = np.mean(np.array([actor.velocity for actor in rear_actors]))
        front_v = front_car.dynamic_object.velocity
        if front_v >= target_v:
            return None

        return LaneChangePlanner.calc_headway_margin(target_v, front_v)

    @staticmethod
    def calc_headway_margin(target_v: float, front_v: float, v_S: float=0):
        """
        calculate margin from a slow front car in addition to the regular margin that enables deceleration + acceleration
        :param target_v: target velocity = convoy velocity
        :param front_v: front car velocity
        :param v_S: minimal permitted velocity after deceleration
        :return: margin from F
        """
        # we are allowed to decelerate to vS
        T_dec, s_dec = LaneChangePlanner._create_quartic_actions(front_v, 0, vT=v_S, A0=-4, J=5)
        T_acc, s_acc = LaneChangePlanner._create_quartic_actions(v_S, 0, target_v, A0=2, J=5)

        half_min_gap = target_v * SAFETY_HEADWAY + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + 2 * LONGITUDINAL_SAFETY_MARGIN_HYSTERESIS
        # calculate full action time T
        T = (s_dec + s_acc + MAX_BACKWARD_HORIZON - half_min_gap - (T_dec + T_acc)*v_S) / (target_v - v_S)
        if T > T_dec + T_acc:  # then use constant velocity v_S between dec & acc
            s = s_dec + s_acc + (T - T_dec - T_acc) * v_S
        else:  # find a higher v_S, without constant velocity
            v_S_arr = np.flip(np.arange(v_S, front_v, 0.1))
            T_dec, s_dec = LaneChangePlanner._create_quartic_actions(front_v, 0, vT=v_S_arr, A0=-4, J=5)
            T_acc, s_acc = LaneChangePlanner._create_quartic_actions(v_S_arr, 0, target_v, A0=2, J=5)
            
            v_S_idx = np.argmin(np.abs(s_dec + s_acc + MAX_BACKWARD_HORIZON - half_min_gap - (T_dec + T_acc) * target_v))
            T, s = T_dec[v_S_idx] + T_acc[v_S_idx], s_dec[v_S_idx] + s_acc[v_S_idx]
    
        # the margin should not include cars size since it's added in specify
        safe_dist = LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + SAFETY_HEADWAY * target_v + \
                    (target_v * target_v - front_v * front_v) / (-2 * LON_ACC_LIMITS[0])
        # here we distract specification headway from the margin since it's added in specify
        margin = s - T * front_v + safe_dist - SPECIFICATION_HEADWAY * front_v  
        return float(np.clip(margin, LONGITUDINAL_SPECIFY_MARGIN_FROM_OBJECT, PLANNING_LOOKAHEAD_DIST))
