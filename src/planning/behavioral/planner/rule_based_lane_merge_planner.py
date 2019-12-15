from logging import Logger
from typing import List
import numpy as np
from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.global_constants import SAFETY_HEADWAY, LON_ACC_LIMITS, EPS, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, BP_JERK_S_JERK_D_TIME_WEIGHTS, LANE_MERGE_ACTION_T_LIMITS, \
    LANE_MERGE_ACTORS_MAX_VELOCITY, LANE_MERGE_WORST_CASE_FRONT_ACTOR_DECEL, LANE_MERGE_WORST_CASE_BACK_ACTOR_ACCEL, \
    LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, LANE_MERGE_YIELD_BACK_ACTOR_RSS_DECEL, SPEEDING_SPEED_TH, \
    LANE_CHANGE_TIME_COMPLETION_TARGET, VELOCITY_LIMITS
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, ActionSpec, RelativeLane, \
    StaticActionRecipe
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeState, LaneChangeStatus
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.src.planning.types import BoolArray, FS_SX, FrenetState1D, FS_SA, FS_SV
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D, QuinticPoly1D
from decision_making.src.state.state import State
from sympy.matrices import *

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
        return ActionSpec(self.t, self.v_T, s0 + self.ds, 0, StaticActionRecipe(RelativeLane.SAME_LANE, self.v_T, AggressivenessLevel.CALM))

    def __str__(self):
        return 't: ' + str(self.t) + ', v_0: ' + str(self.v_0) + ', a_0: ' + str(self.a_0) + \
               ', v_T: ' + str(self.v_T) + ', ds: ' + str(self.ds) + ', coefs: ' + str(self.poly_coefs_num)


class LaneMergeSequence:
    def __init__(self, action_specs: List[LaneMergeSpec]):
        self.action_specs = action_specs

    @property
    def t(self):
        return sum([spec.t for spec in self.action_specs])


class RuleBasedLaneMergePlanner(BasePlanner):
    TIME_GRID_RESOLUTION = 0.5
    VEL_GRID_RESOLUTION = 2

    def __init__(self, logger: Logger):
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

    def _create_action_specs(self, lane_merge_state: LaneMergeState) -> np.array:
        """
        Create all possible actions to the merge point, filter unsafe actions, filter actions exceeding vel-acc limits,
        calculate time-jerk cost for the remaining actions.
        :param lane_merge_state: LaneMergeState containing distance to merge and actors
        :return: list of initial action specs, array of jerks until the merge, array of times until the merge
        """
        import time
        st_tot = time.time()

        t_res = RuleBasedLaneMergePlanner.TIME_GRID_RESOLUTION
        t_grid = np.arange(t_res, LANE_MERGE_ACTION_T_LIMITS[1] + EPS, t_res)
        v_grid = np.arange(0, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY + EPS, RuleBasedLaneMergePlanner.VEL_GRID_RESOLUTION)

        target_v, target_t = np.meshgrid(v_grid, t_grid)
        target_v, target_t = target_v.ravel(), target_t.ravel()
        vts, vts_last = RuleBasedLaneMergePlanner._calculate_safe_target_points(lane_merge_state, target_v, target_t)
        if vts.shape[0] == 0:
            return []

        # create single-spec quintic safe actions
        single_actions = RuleBasedLaneMergePlanner._create_quintic_actions(lane_merge_state, vts)

        # create composite "max_velocity" actions, such that each sequence includes quartic + const_max_vel + quartic
        max_vel_actions = RuleBasedLaneMergePlanner._create_max_vel_actions(lane_merge_state, v_grid, t_grid)

        # create composite "stop" actions, such that each sequence includes quartic + zero_vel + quartic
        # create stop actions only if there are no single-spec actions, since quintic actions have lower time
        # and lower jerk than composite stop action
        stop_actions = RuleBasedLaneMergePlanner._create_braking_actions(lane_merge_state, vts_last)

        lane_merge_actions = single_actions + max_vel_actions + stop_actions

        # print('\ntime = %f: quintic=(%d) quartic=(%d)' % (time.time() - st_tot, len(single_actions), len(max_vel_actions)))
        return np.array(lane_merge_actions)

    def _filter_actions(self, lane_merge_state: LaneMergeState, actions: np.array) -> np.array:
        """
        Currently do nothing. Safety, acc & vel limits are tested inside _create_action_specs.
        :param lane_merge_state: lane merge state
        :param actions: lane merge action sequences
        :return: array of actions of the same size as the input, but filtered actions are None
        """
        return actions

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
            raise NoActionsLeftForBPError("RB.evaluate_actions: No actions to evaluate. timestamp_in_sec: %f" %
                                          lane_merge_state.ego_state.timestamp_in_sec)
        # calculate full actions' jerks and times
        actions_jerks = np.zeros(len(actions))
        actions_times = np.zeros(len(actions))
        actions_dists = np.zeros(len(actions))
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
            accel_to_max_vel_s, accel_to_max_vel_T = KinematicUtils.specify_quartic_actions(
                calm_weights[2], calm_weights[0], v_T, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, action_horizon_limit=np.inf)

            # collect actions jerks, times and distances
            for action_idx, spec_jerk, spec_t, spec_v, spec_s, acc_t, acc_s in \
                    zip(action_idxs.astype(int), spec_jerks, T, v_T, ds, accel_to_max_vel_T, accel_to_max_vel_s):
                actions_jerks[action_idx] += spec_jerk
                actions_times[action_idx] += spec_t + acc_t  # add acceleration time from v_T to max_vel
                actions_dists[action_idx] += spec_s + acc_s

        # calculate actions' costs according to the AGGRESSIVE jerk-time weights
        time_jerk_weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.AGGRESSIVE.value]
        # assume that after acceleration to max_vel it will continue to max_dist with max_vel
        max_dist = np.max(actions_dists)
        full_times = actions_times + (max_dist - actions_dists) / LANE_MERGE_ACTION_SPACE_MAX_VELOCITY
        action_costs = time_jerk_weights[0] * actions_jerks + time_jerk_weights[2] * full_times
        action_costs[actions == None] = np.inf
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
                action_time <= RuleBasedLaneMergePlanner.TIME_GRID_RESOLUTION:
            chosen_spec.recipe.relative_lane = lane_merge_state.target_rel_lane
            chosen_spec.t = LANE_CHANGE_TIME_COMPLETION_TARGET
            chosen_spec.s = lane_merge_state.projected_ego_fstates[lane_merge_state.target_rel_lane][FS_SX] + \
                            chosen_spec.t * lane_merge_state.ego_state.velocity
            lane_merge_state.lane_change_state.status = LaneChangeStatus.AnalyzingSafety
            lane_merge_state.lane_change_state.lane_change_start_time = lane_merge_state.ego_state.timestamp_in_sec
            lane_merge_state.lane_change_state.autonomous_mode = True

        return chosen_spec

    @staticmethod
    def _create_quintic_actions(state: LaneMergeState, vts: np.array) -> List[LaneMergeSequence]:
        v_T, T, ds = vts.T

        # calculate s_profile coefficients for all actions
        ego_fstate = state.ego_fstate_1d
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
        poly_coefs = QuinticPoly1D.position_profile_coefficients(a_0, v_0, v_T, ds, T)

        # the fast analytic kinematic filter reduce the load on the regular kinematic filter
        # calculate actions that don't violate acceleration limits
        valid_acc = QuinticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)
        if not valid_acc.any():
            return []
        valid_vel = QuinticPoly1D.are_velocities_in_limits(poly_coefs[valid_acc], T[valid_acc], VELOCITY_LIMITS)
        valid_idxs = np.where(valid_acc)[0][valid_vel]

        actions = [LaneMergeSequence([LaneMergeSpec(t, v_0, a_0, vT, s, QuinticPoly1D.num_coefs())])
                   for t, vT, s in zip(T[valid_idxs], v_T[valid_idxs], ds[valid_idxs])]
        return actions

    @staticmethod
    def _create_max_vel_actions(state: LaneMergeState, v_grid: np.array, t_grid: np.array) -> List[LaneMergeSequence]:
        """
        Given array of final velocities, create composite safe actions:
            1. quartic CALM acceleration to v_max,
            2. constant velocity with v_max,
            3. quartic STANDARD deceleration to v_T.
        :param state: lane merge state
        :param v_grid: array of final velocities
        :param t_grid: array of planning times
        :return: list of safe composite actions
        """
        s_min = max(0, state.merge_from_s_on_ego_gff - state.ego_fstate_1d[FS_SX])
        s_max = state.red_line_s_on_ego_gff - state.ego_fstate_1d[FS_SX]

        v_max = LANE_MERGE_ACTION_SPACE_MAX_VELOCITY
        ego_fstate = state.ego_fstate_1d
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]

        w_J_calm, _, w_T_calm = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]
        w_J_stand, _, w_T_stand = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.STANDARD.value]
        s1, t1 = KinematicUtils.specify_quartic_action(w_T_calm, w_J_calm, v_0, v_max, a_0)
        S3_grid, T3_grid = KinematicUtils.specify_quartic_actions(w_T_stand, w_J_stand, v_max, v_grid)

        target_v, target_t = np.meshgrid(v_grid, t_grid)
        target_v, target_t = target_v.ravel(), target_t.ravel()
        mesh_to_v_grid = np.tile(np.arange(len(v_grid)), len(t_grid))
        S3 = S3_grid[mesh_to_v_grid]
        T3 = T3_grid[mesh_to_v_grid]

        T2 = target_t - t1 - T3
        S2 = T2 * v_max
        ds = s1 + S2 + S3
        valid = (s_min < ds) & (ds < s_max) & (T2 > 0)
        if not valid.any():
            return []
        T2, S2, T3, S3, v_end, t_end, ds = T2[valid], S2[valid], T3[valid], S3[valid], target_v[valid], target_t[valid], ds[valid]

        actors_s, actors_v, actors_length = np.array([[actor.s_relative_to_ego, actor.velocity, actor.length]
                                                      for actor in state.actors_states]).T
        margins = 0.5 * (actors_length + state.ego_length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
        safety_dist = RuleBasedLaneMergePlanner._caclulate_RSS_distances(
            actors_s[:, np.newaxis], actors_v[:, np.newaxis], margins[:, np.newaxis], v_end, t_end, ds)
        is_safe = safety_dist > 0

        actions = []
        for t2, s2, t3, s3, v_t in zip(T2[is_safe], S2[is_safe], T3[is_safe], S3[is_safe], v_end[is_safe]):
            action1 = LaneMergeSpec(t1, v_0, a_0, v_max, s1, QuarticPoly1D.num_coefs())
            action2 = LaneMergeSpec(t2, v_max, 0, v_max, s2, poly_coefs_num=2)
            action3 = LaneMergeSpec(t3, v_max, 0, v_t, s3, QuarticPoly1D.num_coefs())
            actions.append(LaneMergeSequence([action1, action2, action3]))

        return actions

    @staticmethod
    def _create_braking_actions(state: LaneMergeState, vts_last: np.array) -> List[LaneMergeSequence]:

        ego_fstate = state.ego_fstate_1d
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
        terminal_v, terminal_t, terminal_s = vts_last.T

        # specify aggressive stop
        w_J_agg, _, w_T_agg = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.AGGRESSIVE.value]
        brake_distance, brake_time = KinematicUtils.specify_quartic_action(w_T_agg, w_J_agg, v_0, 0., a_0)

        # specify quartic accelerations to the terminal states
        acc_dist = terminal_s - brake_distance
        acc_time = 2 * acc_dist / terminal_v
        acc_poly_s = QuarticPoly1D.position_profile_coefficients(a_0=0, v_0=0, v_T=terminal_v, T=acc_time)
        valid_acc = QuarticPoly1D.are_accelerations_in_limits(acc_poly_s, acc_time, LON_ACC_LIMITS)
        valid_idxs = np.where((acc_dist > 0) & (brake_time + acc_time <= terminal_t) & valid_acc)[0]

        # create actions with full stop
        actions = []
        for t_acc, s_acc, v_term, t_term in zip(acc_time[valid_idxs], acc_dist[valid_idxs], terminal_v[valid_idxs],
                                                terminal_t[valid_idxs]):
            action1 = LaneMergeSpec(brake_time, v_0, a_0, 0, brake_distance, QuarticPoly1D.num_coefs())
            action2 = LaneMergeSpec(t_term - (brake_time + t_acc), 0, 0, 0, 0, poly_coefs_num=2)
            action3 = LaneMergeSpec(t_acc, 0, 0, v_term, s_acc, QuarticPoly1D.num_coefs())
            actions.append(LaneMergeSequence([action1, action2, action3]))

        # create actions that start during the braking (before the full stop)
        brake_poly_s = QuarticPoly1D.position_profile_coefficients(a_0=a_0, v_0=v_0, v_T=0, T=np.array([brake_time]))
        brake_t = np.arange(start=2, stop=brake_time, step=1)
        brake_s, brake_v, brake_a = QuarticPoly1D.polyval_with_derivatives(brake_poly_s, brake_t)[0].T

        return actions

    @staticmethod
    def _calculate_safe_target_points(state: LaneMergeState, target_v: np.array, target_t: np.array) -> \
            [np.array, np.array]:
        """
        Create boolean 2D matrix of actions that are longitudinally safe (RSS) between red line & merge point w.r.t.
        all actors.
        :param state: lane merge state
        :return: two 2D arrays of [v_T, T, s], where ego is safe at target_s relatively to all actors
        """
        s_min = max(0, state.merge_from_s_on_ego_gff - state.ego_fstate_1d[FS_SX])
        s_max = state.red_line_s_on_ego_gff - state.ego_fstate_1d[FS_SX]

        ego_length = state.ego_length
        ego_fstate = state.ego_fstate_1d
        actors_states = state.actors_states
        actors_s, actors_v, actors_length = np.array([[actor.s_relative_to_ego, actor.velocity, actor.length]
                                                      for actor in actors_states]).T
        # add extra margin for longer actions to enable comfort short actions and another extra margin for smooth
        # switching with the single_step_planner
        margins = 0.5 * (actors_length + ego_length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + \
                  1 + 1 * (target_t[:, np.newaxis] > 2)

        actors_s.sort()
        actor_i = np.sum(actors_s < 0)
        print('s_min=', s_min, 's_max=', s_max, 'actors_rel_s=', actors_s[actor_i-1:actor_i+1])

        # calculate planning time bounds given target_s
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
        if len(actors_states) == 0:
            return target_v, target_t

        # 2D matrix with 3 columns of safe actions: target velocities, planning times and distances
        front_bounds, back_bounds = RuleBasedLaneMergePlanner._caclulate_RSS_bounds(
            actors_s, actors_v, margins, target_v[:, np.newaxis], target_t[:, np.newaxis])

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

        # calculate jerk-optimal target s according to the QUARTIC distance formula
        optimal_s = target_t * (target_t * a_0 + 6 * v_0 + 6 * target_v) / 12

        # a bound is defined important if it changes the safety:
        #       front bound is important if safety_layer passes from 0 to 1
        #       back bound is important if safety layer passes from 1 to 0
        are_important_bounds = ((sorted_bounds[..., 1] == 1) & (safety_layer == 1)) | \
                               ((sorted_bounds[..., 1] == -1) & (safety_layer == 0))
        important_bounds = sorted_bounds[..., 0]
        important_bounds[~are_important_bounds] = np.inf
        important_bounds.sort(axis=1)

        # find actions for which optimal_s is safe
        above_optimal_bound_idxs = np.sum(important_bounds < optimal_s[:, np.newaxis], axis=-1)
        # optimal_s is safe if number of important bounds under optimal_s is even
        optimal_s_is_safe = (above_optimal_bound_idxs & 1) == 0

        # calculate important bounds under & above optimal_s for actions with unsafe optimal_s
        unsafe_optimal_s_idxs = np.where(~optimal_s_is_safe)[0]
        target_v_unsafe_optimal, target_t_unsafe_optimal = target_v[unsafe_optimal_s_idxs], target_t[unsafe_optimal_s_idxs]
        above_unsafe_optimal_bound_idxs = above_optimal_bound_idxs[unsafe_optimal_s_idxs]

        # calculate all safe target points for optimal_s (quartic) in 3D space (v_T, T, ds)
        vts_optimal = np.c_[target_v[optimal_s_is_safe], target_t[optimal_s_is_safe], optimal_s[optimal_s_is_safe]]
        # calculate all safe points for a bound under optimal_s
        vts_under_optimal = RuleBasedLaneMergePlanner._calculate_points_for_safe_bounds(
            important_bounds[unsafe_optimal_s_idxs, above_unsafe_optimal_bound_idxs-1],
            target_v_unsafe_optimal, target_t_unsafe_optimal)
        # calculate all safe points for a bound above optimal_s
        vts_above_optimal = RuleBasedLaneMergePlanner._calculate_points_for_safe_bounds(
            important_bounds[unsafe_optimal_s_idxs, above_unsafe_optimal_bound_idxs],
            target_v_unsafe_optimal, target_t_unsafe_optimal)
        vts = np.concatenate((vts_optimal, vts_under_optimal, vts_above_optimal), axis=0)

        # for each [target_v, target_t], find s of the last safe bound
        last_bound_idxs = np.argmax(important_bounds == np.inf, axis=-1) - 1
        vts_last = RuleBasedLaneMergePlanner._calculate_points_for_safe_bounds(
            important_bounds[range(important_bounds.shape[0]), last_bound_idxs], target_v, target_t)

        return vts, vts_last

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
        return np.c_[target_v[actions_with_valid_bounds], target_t[actions_with_valid_bounds], bounds[actions_with_valid_bounds]]

    @staticmethod
    def _caclulate_RSS_distances(actors_s: np.array, actors_v: np.array, margins: np.array,
                                 target_v: np.array, target_t: np.array, target_s: np.array) -> np.array:
        """
        Given actors on the main road and actions (planning times, target velocities and target distances),
        create matrix of differences between predicted distances from the actors and minimal safe distances at
        actions' end-points target_s.
        :param actors_s: current s of actor relatively to the merge point (negative or positive)
        :param actors_v: current actor's velocity
        :param margins: half sum of cars' lengths + safety margin
        :param target_t: array of planning times
        :param target_v: array of target velocities
        :param target_s: array of target s
        :return: shape like of target_(v/s/t): difference between predicted distances and safe distances
        """
        front_bounds, back_bounds = RuleBasedLaneMergePlanner._caclulate_RSS_bounds(actors_s, actors_v, margins, target_v, target_t)

        # calculate if ego is safe according to the longitudinal RSS formula
        front_safety_dist = front_bounds - target_s
        back_safety_dist = target_s - back_bounds
        safety_dist = np.maximum(front_safety_dist, back_safety_dist)
        return np.min(safety_dist, axis=0)  # AND on actors

    @staticmethod
    def _caclulate_RSS_bounds(actors_s: np.array, actors_v: np.array, margins: np.array,
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
        :param margins: half sum of cars' lengths + safety margin
        :param target_t: array of planning times
        :param target_v: array of target velocities, of the same shape as target_t
        :return: two 2D matrices of shape: actions_num x actors_num.
        """
        front_braking_time = np.minimum(actors_v / LANE_MERGE_WORST_CASE_FRONT_ACTOR_DECEL, target_t) \
            if LANE_MERGE_WORST_CASE_FRONT_ACTOR_DECEL > 0 else target_t
        front_v = actors_v - front_braking_time * LANE_MERGE_WORST_CASE_FRONT_ACTOR_DECEL  # target velocity of the front actor
        front_s = actors_s + 0.5 * (actors_v + front_v) * front_braking_time

        back_accel_time = np.minimum((LANE_MERGE_ACTORS_MAX_VELOCITY - actors_v) / LANE_MERGE_WORST_CASE_BACK_ACTOR_ACCEL, target_t) \
            if LANE_MERGE_WORST_CASE_BACK_ACTOR_ACCEL > 0 else target_t
        back_max_vel_time = target_t - back_accel_time  # time of moving with maximal velocity of the back actor
        back_v = actors_v + back_accel_time * LANE_MERGE_WORST_CASE_BACK_ACTOR_ACCEL  # target velocity of the back actor
        back_s = actors_s + 0.5 * (actors_v + back_v) * back_accel_time + LANE_MERGE_ACTORS_MAX_VELOCITY * back_max_vel_time

        # calculate target_s bounds according to the longitudinal RSS formula
        front_bounds = front_s - margins - (np.maximum(0, target_v * target_v - front_v * front_v) /
                                            (-2 * LON_ACC_LIMITS[0]) + SAFETY_HEADWAY * target_v)
        back_bounds = back_s + margins + (np.maximum(0, back_v * back_v - target_v * target_v) /
                                          (2 * LANE_MERGE_YIELD_BACK_ACTOR_RSS_DECEL) + SAFETY_HEADWAY * back_v)
        return front_bounds, back_bounds

    @staticmethod
    def _sympy():
        import sympy as sp
        from sympy import symbols

        v_0, a_0, s, v_T = symbols('v_0 a_0 s v_T')

        T = symbols('T')
        t = symbols('t')
        A = Matrix([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
            [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
            [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0]]
        )
        [c5, c4, c3, c2, c1, c0] = A.inv() * Matrix([0, v_0, a_0, s, v_T, 0])
        x_t = (c5 * t ** 5 + c4 * t ** 4 + c3 * t ** 3 + c2 * t ** 2 + c1 * t + c0).simplify()
        v_t = sp.diff(x_t, t).simplify()
        a_t = sp.diff(v_t, t).simplify()
        j_t = sp.diff(a_t, t).simplify()
        J = sp.integrate(j_t ** 2, (t, 0, T)).simplify()  # integrate by t
        w_J, w_T = symbols('w_J w_T')
        cost = (w_J * J + w_T * T).simplify()
        cost_diff = sp.diff(cost, s).simplify()  # differentiate by s
        s_opt = sp.solve(cost_diff, s)[0]
        s_opt = T * (T*a_0 + 6*v_0 + 6*v_T) / 12
        cost_opt = cost.subs(s, s_opt).simplify()
        cost_opt = (T**4*w_T + 4*T**2*a_0**2*w_J + 12*T*a_0*v_0*w_J - 12*T*a_0*v_T*w_J + 12*v_0**2*w_J - 24*v_0*v_T*w_J + 12*v_T**2*w_J)/T**3

    @staticmethod
    def get_cost(w_J: float, w_T: float, v_0: float, a_0: float, v_T: float, T: float, s: float):
        return (T**6*w_T + 3*w_J*(3*T**4*a_0**2 + 24*T**3*a_0*v_0 + 16*T**3*a_0*v_T - 40*T**2*a_0*s + 64*T**2*v_0**2 +
                                  112*T**2*v_0*v_T + 64*T**2*v_T**2 - 240*T*s*v_0 - 240*T*s*v_T + 240*s**2))/T**5
