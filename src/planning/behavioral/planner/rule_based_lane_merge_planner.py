from logging import Logger
from typing import List
import numpy as np
from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.global_constants import SAFETY_HEADWAY, LON_ACC_LIMITS, EPS, TRAJECTORY_TIME_RESOLUTION, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, BP_JERK_S_JERK_D_TIME_WEIGHTS, MAX_BACKWARD_HORIZON, \
    LANE_MERGE_ACTION_T_LIMITS, LANE_MERGE_ACTORS_MAX_VELOCITY, LANE_MERGE_WORST_CASE_FRONT_ACTOR_DECEL, \
    LANE_MERGE_WORST_CASE_BACK_ACTOR_ACCEL, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY, LANE_MERGE_YIELD_BACK_ACTOR_RSS_DECEL
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, ActionSpec, StaticActionRecipe, \
    RelativeLane
from decision_making.src.planning.behavioral.default_config import DEFAULT_ACTION_SPEC_FILTERING
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState, LaneMergeActorState
from decision_making.src.planning.types import ActionSpecArray, BoolArray, FS_SX, FrenetState1D, FS_SA, FS_SV
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D, QuinticPoly1D
from decision_making.src.state.state import State
from rte.python.logger.AV_logger import AV_Logger
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

    def __str__(self):
        return 't: ' + str(self.t) + ', v_0: ' + str(self.v_0) + ', a_0: ' + str(self.a_0) + \
               ', v_T: ' + str(self.v_T) + ', ds: ' + str(self.ds) + ', coefs: ' + str(self.poly_coefs_num)


class LaneMergeSequence:
    def __init__(self, action_specs: List[LaneMergeSpec]):
        self.action_specs = action_specs


class RuleBasedLaneMergePlanner(BasePlanner):
    TIME_GRID_RESOLUTION = 0.5
    VEL_GRID_RESOLUTION = 5

    def __init__(self, logger: Logger):
        super().__init__(logger)

    def _create_behavioral_state(self, state: State, route_plan: RoutePlan) -> LaneMergeState:
        """
        Create LaneMergeState (which inherits from BehavioralGridState) from the given state
        :param state: state from scene dynamic
        :param route_plan: the route plan
        :return: LaneMergeState
        """
        return LaneMergeState.create_from_state(state, route_plan, self.logger)

    def _create_action_specs(self, lane_merge_state: LaneMergeState) -> ActionSpecArray:
        """
        The action space consists of 3 quartic actions accelerating to the maximal velocity with 3 aggressiveness
        levels. Specify these actions, validate their longitudinal acceleration and check their longitudinal RSS safety.
        :param lane_merge_state:
        :return: array of valid action specs, including unsafe actions
        """
        # specify and check safety for 3 quartic actions to maximal velocity
        s_0, v_0, a_0 = lane_merge_state.ego_fstate_1d
        v_T = LANE_MERGE_ACTION_SPACE_MAX_VELOCITY
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS.T

        # specify 3 static actions with 3 aggressiveness levels
        specs_t, specs_s = RuleBasedLaneMergePlanner._specify_quartic(v_0, a_0, v_T, w_T, w_J)

        # create action specs based on the above output
        return np.array([ActionSpec(t=T, v=v_T, s=s_0 + ds, d=0,
                                    recipe=StaticActionRecipe(RelativeLane.SAME_LANE, v_T, aggr_level))
                         for T, ds, aggr_level in zip(specs_t, specs_s, AggressivenessLevel)])

    def _filter_actions(self, lane_merge_state: LaneMergeState, action_specs: ActionSpecArray) -> ActionSpecArray:
        """
        In addition to all standard ActionSpec filters apply also RSS safety filter w.r.t. the main road actors
        :param lane_merge_state: lane merge state
        :param action_specs: action specifications
        :return: array of action_specs of the same size as the input, but filtered actions are None
        """
        action_specs_mask = DEFAULT_ACTION_SPEC_FILTERING.filter_action_specs(action_specs, lane_merge_state)
        filtered_action_specs = np.full(len(action_specs), None)
        filtered_action_specs[action_specs_mask] = action_specs[action_specs_mask]
        if not filtered_action_specs.any():
            raise NoActionsLeftForBPError("RB.filter_actions: All actions were filtered. timestamp_in_sec: %f" %
                                          lane_merge_state.ego_state.timestamp_in_sec)
        safe_actions = RuleBasedLaneMergePlanner._safety_filter(lane_merge_state, filtered_action_specs)
        if not safe_actions.any():
            raise NoActionsLeftForBPError("RB.filter_actions: No safe actions. timestamp_in_sec: %f" %
                                          lane_merge_state.ego_state.timestamp_in_sec)
        return safe_actions

    def _evaluate_actions(self, lane_merge_state: LaneMergeState, route_plan: RoutePlan,
                          action_specs: ActionSpecArray) -> np.array:
        """
        Evaluates Action-Specifications with the lowest aggressiveness possible
        :param lane_merge_state: lane merge state
        :param route_plan:
        :param action_specs: specifications of action_recipes.
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        action_mask = action_specs.astype(bool)
        if not action_mask.any():
            raise NoActionsLeftForBPError("RB.evaluate_actions: No actions to evaluate. timestamp_in_sec: %f" %
                                          lane_merge_state.ego_state.timestamp_in_sec)
        action_costs = np.full(len(action_specs), 1.)
        most_calm_action_idx = np.argmax(action_mask)
        action_costs[most_calm_action_idx] = 0
        return action_costs

    def _choose_action(self, lane_merge_state: LaneMergeState, action_specs: ActionSpecArray, costs: np.array) -> \
            ActionSpec:
        """
        Return action_spec having the minimal cost
        :param lane_merge_state: lane merge state
        :param action_specs: array of ActionSpecs
        :param costs: array of actions' costs
        :return: action specification with minimal cost
        """
        return action_specs[np.argmin(costs)]

    @staticmethod
    def _specify_quartic(v_0: float, a_0: float, v_T: np.array, w_T: np.array, w_J: np.array) -> [np.array, np.array]:
        # T_s <- find minimal non-complex local optima within the ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuarticPoly1D.time_cost_function_derivative_coefs(w_T=w_T, w_J=w_J, a_0=a_0, v_0=v_0, v_T=v_T)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, LANE_MERGE_ACTION_T_LIMITS)
        T = np.fmin.reduce(roots_s, axis=-1)
        s = QuarticPoly1D.distance_profile_function(a_0=a_0, v_0=v_0, v_T=v_T, T=T)(T)
        s[np.isclose(T, 0)] = 0
        return T, s

    @staticmethod
    def _safety_filter(lane_merge_state: LaneMergeState, action_specs: ActionSpecArray) -> ActionSpecArray:
        """
        Test RSS safety on the red line, assuming the worst case scenario of the main road actors.
        For each action_spec check if it ends before the red line. If yes, assume constant maximal velocity
        until the red line. If no, find the target point, where the action crosses the red line.
        Then check safety on this target point for all actions.
        :param lane_merge_state: lane merge state, containing data about host and the main road vehicles
        :param action_specs: array of action specifications (some of them may be None)
        :return: array (of the same size as input) of action specifications, where unsafe actions are None
        """
        s_0, v_0, a_0 = lane_merge_state.ego_fstate_1d
        v_T = LANE_MERGE_ACTION_SPACE_MAX_VELOCITY
        rel_red_line_s = lane_merge_state.red_line_s_on_ego_gff - s_0
        specs_t, specs_v, specs_s = np.array([[spec.t, spec.v, spec.s] for spec in action_specs if spec is not None]).T

        # initialize target points for the actions that end before the red line (specs_s < rel_red_line_s)
        target_t = specs_t + (rel_red_line_s - specs_s) / v_T
        target_v = np.full(specs_t.shape[0], v_T, dtype=float)
        target_s = np.full(specs_t.shape[0], rel_red_line_s, dtype=float)

        # calculate position profile coefficients
        poly_s = np.zeros((specs_t.shape[0], QuarticPoly1D.num_coefs()))
        positiveT = np.zeros_like(specs_t).astype(bool)
        positiveT[~np.isnan(specs_t)] = (specs_t[~np.isnan(specs_t)] > 0)
        poly_s[positiveT] = QuarticPoly1D.position_profile_coefficients(a_0, v_0, v_T, specs_t[positiveT])

        # overwrite target points for the crossing red line actions by calculating the crossing point
        crossing_red_line = (specs_s > rel_red_line_s)
        if crossing_red_line.any():
            times = np.arange(0, np.max(specs_t[crossing_red_line]) + EPS, TRAJECTORY_TIME_RESOLUTION)
            s_values = Math.polyval2d(poly_s[crossing_red_line], times)
            poly_vel = Math.polyder2d(poly_s[crossing_red_line], m=1)
            red_line_idxs = np.argmin(np.abs(s_values - rel_red_line_s), axis=1)
            target_t[crossing_red_line] = red_line_idxs * TRAJECTORY_TIME_RESOLUTION
            target_v[crossing_red_line] = Math.zip_polyval2d(poly_vel, target_t[crossing_red_line, np.newaxis])[:, 0]
            target_s[crossing_red_line] = s_values[np.arange(s_values.shape[0]), red_line_idxs]

        # add a dummy actor on the main road at MAX_BACKWARD_HORIZON behind ego with maximal velocity
        actors_states = lane_merge_state.actors_states.copy()
        actors_states.append(LaneMergeActorState(-MAX_BACKWARD_HORIZON, LANE_MERGE_ACTORS_MAX_VELOCITY, 0))

        # check RSS safety at the red line, assuming the worst case scenario
        actors_data = np.array([[actor.s_relative_to_ego, actor.velocity, actor.length] for actor in actors_states])
        actors_s = actors_data[:, 0, np.newaxis]
        actors_v = actors_data[:, 1, np.newaxis]
        margins = 0.5 * (actors_data[:, 2, np.newaxis] + lane_merge_state.ego_length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
        safety_dist = RuleBasedLaneMergePlanner._caclulate_RSS_distances(actors_s, actors_v, margins, target_v, target_t, target_s)

        filtered_specs = action_specs[action_specs != None]
        filtered_specs[safety_dist <= 0] = None
        return filtered_specs

    @staticmethod
    def _caclulate_RSS_distances(actors_s: np.array, actors_v: np.array, margins: np.array,
                                 target_v: np.array, target_t: np.array, target_s: np.array) -> BoolArray:
        """
        Given an actor on the main road and two grids of planning times and target velocities, create boolean matrix
        of all possible actions that are longitudinally safe (RSS) at actions' end-points target_s.
        :param actors_s: current s of actor relatively to the merge point (negative or positive)
        :param actors_v: current actor's velocity
        :param margins: half sum of cars' lengths + safety margin
        :param target_t: array of planning times
        :param target_v: array of target velocities
        :param target_s: array of target s
        :return: shape like of target_(v/s/t): difference between actual distances and safe distances
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

        # calculate if ego is safe according to the longitudinal RSS formula
        front_safety_dist = front_s - target_s - margins - (np.maximum(0, target_v * target_v - front_v * front_v) /
                                                            (-2 * LON_ACC_LIMITS[0]) + SAFETY_HEADWAY * target_v)
        back_safety_dist = target_s - back_s - margins - (np.maximum(0, back_v * back_v - target_v * target_v) /
                                                (2 * LANE_MERGE_YIELD_BACK_ACTOR_RSS_DECEL) + SAFETY_HEADWAY * back_v)
        safety_dist = np.maximum(front_safety_dist, back_safety_dist)
        return np.min(safety_dist, axis=0)  # AND on actors

    @staticmethod
    def choose_max_vel_quartic_trajectory(state: LaneMergeState) -> np.array:
        """
        Used by planning_research for SUMO.
        Check existence of rule-based solution that can merge safely, assuming the worst case scenario of
        main road actors. The function tests a single static action toward maximal velocity (ScenarioParams.ego_max_velocity).
        If the action is safe (longitudinal RSS) during crossing red line w.r.t. all main road actors, return True.
        :param state: lane merge state, containing data about host and the main road vehicles
        :return: accelerations array or empty array if there is no safe action
        """
        logger = AV_Logger.get_logger()
        planner = RuleBasedLaneMergePlanner(logger)
        actions = planner._create_action_specs(state)
        s_0, v_0, a_0 = state.ego_fstate_1d
        specs_v, specs_t, specs_s = np.array([[spec.v, spec.t, spec.s] for spec in actions]).T

        # validate accelerations
        valid_acc, poly_s = RuleBasedLaneMergePlanner._validate_acceleration(v_0, a_0, specs_v, specs_t)
        valid_actions = actions[valid_acc]
        valid_poly_s = poly_s[valid_acc]

        # validate safety
        safe_actions = RuleBasedLaneMergePlanner._safety_filter(state, valid_actions)
        if not safe_actions.astype(bool).any():
            return np.array([])

        chosen_action_idx = np.argmax(safe_actions.astype(bool))
        chosen_action = valid_actions[chosen_action_idx]
        chosen_poly = valid_poly_s[chosen_action_idx]
        poly_acc = np.polyder(chosen_poly, m=2)
        times = np.arange(0.5, min(OUTPUT_TRAJECTORY_LENGTH, chosen_action.t/TRAJECTORY_TIME_RESOLUTION)) * TRAJECTORY_TIME_RESOLUTION
        accelerations = np.zeros(OUTPUT_TRAJECTORY_LENGTH)
        accelerations[:times.shape[0]] = np.polyval(poly_acc, times)
        return accelerations

    @staticmethod
    def _validate_acceleration(v_0: float, a_0: float, v_T: float, T: np.array) -> [np.array, np.array]:
        """
        Check acceleration in limits for quartic polynomials.
        :param v_0: initial velocity
        :param a_0: initial acceleration
        :param v_T: target velocity(es): either scalar or array of size len(T)
        :param T: array of planning times
        :return: boolean array of size len(T) of valid actions and matrix Nx5: s_profile polynomials for all T
        """
        # validate acceleration limits of the initial quartic action
        poly_s = np.zeros((T.shape[0], QuarticPoly1D.num_coefs()))
        nonnan = ~np.isnan(T)
        positiveT = np.copy(nonnan)
        valid_acc = np.copy(nonnan)
        positiveT[nonnan] = np.greater(T[nonnan], 0)
        valid_acc[nonnan] = np.equal(T[nonnan], 0)
        poly_s[positiveT] = QuarticPoly1D.position_profile_coefficients(a_0, v_0, v_T if np.isscalar(v_T) else v_T[positiveT], T[positiveT])
        valid_acc[positiveT] = QuarticPoly1D.are_accelerations_in_limits(poly_s[positiveT], T[positiveT], LON_ACC_LIMITS)
        return valid_acc, poly_s

    @staticmethod
    def create_safe_actions(state: LaneMergeState) -> List[LaneMergeSequence]:
        """
        Create all possible actions to the merge point, filter unsafe actions, filter actions exceeding vel-acc limits,
        calculate time-jerk cost for the remaining actions.
        :param state: LaneMergeState containing distance to merge and actors
        :return: list of initial action specs, array of jerks until the merge, array of times until the merge
        """
        import time
        st_tot = time.time()

        st = time.time()
        ego_fstate = state.ego_fstate_1d
        # calculate the actions' distance (the same for all actions)
        v_T, T, ds = RuleBasedLaneMergePlanner._calculate_safe_target_points(state)
        if len(T) == 0:
            return []

        time_safety = time.time() - st

        st = time.time()
        # create single-spec quintic safe actions
        single_actions = RuleBasedLaneMergePlanner._create_quintic_actions(ego_fstate, v_T, T, ds)
        time_quintic = time.time() - st

        st = time.time()
        # create composite "max_velocity" actions, such that each sequence includes quartic + const_max_vel + quartic
        max_vel_actions = RuleBasedLaneMergePlanner._create_max_vel_actions(state, v_T, T, ds)

        # create composite "stop" actions, such that each sequence includes quartic + zero_vel + quartic
        # create stop actions only if there are no single-spec actions, since quintic actions have lower time
        # and lower jerk than composite stop action
        # if len(single_actions) == 0:
        #     stop_actions = RuleBasedLaneMergePlanner._create_composite_actions(ego_fstate, T, v_T, 0, ds)
        time_quartic = time.time() - st

        lane_merge_actions = single_actions + max_vel_actions

        print('\ntime = %f: safety=%f quintic=%f(%d) quartic=%f(%d)' %
              (time.time() - st_tot, time_safety, time_quintic, len(single_actions), time_quartic, len(max_vel_actions)))
        return lane_merge_actions

    @staticmethod
    def _create_quintic_actions(ego_fstate: FrenetState1D, v_T: np.array, T: np.array, ds: np.array) -> \
            List[LaneMergeSequence]:
        # calculate s_profile coefficients for all actions
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
        poly_coefs = QuinticPoly1D.position_profile_coefficients(a_0, v_0, v_T, ds, T)

        # the fast analytic kinematic filter reduce the load on the regular kinematic filter
        # calculate actions that don't violate acceleration limits
        valid_acc = QuinticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)
        if not valid_acc.any():
            return []
        v_max = LANE_MERGE_ACTION_SPACE_MAX_VELOCITY
        valid_vel = QuinticPoly1D.are_velocities_in_limits(poly_coefs[valid_acc], T[valid_acc], np.array([0, v_max]))
        valid_idxs = np.where(valid_acc)[0][valid_vel]

        actions = [LaneMergeSequence([LaneMergeSpec(t, v_0, a_0, vT, ds, QuinticPoly1D.num_coefs())])
                   for t, vT in zip(T[valid_idxs], v_T[valid_idxs])]

        return actions

    @staticmethod
    def _create_max_vel_actions(state: LaneMergeState, v_T: np.array, T: np.array) -> List[LaneMergeSequence]:
        """
        Given array of final velocities, create composite safe actions:
            1. quartic CALM acceleration to v_max,
            2. constant velocity with v_max,
            3. quartic STANDARD deceleration to v_T.
        :param state: lane merge state
        :param v_T: array of final velocities in safe points
        :return: list of safe composite actions
        """
        v_max = LANE_MERGE_ACTION_SPACE_MAX_VELOCITY
        ego_fstate = state.ego_fstate_1d
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
        w_J_calm, _, w_T_calm = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]
        w_J_stand, _, w_T_stand = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.STANDARD.value]
        s1, t1 = KinematicUtils.specify_quartic_actions(w_T_calm, w_J_calm, v_0, v_max, a_0)
        S3, T3 = KinematicUtils.specify_quartic_actions(w_T_stand, w_J_stand, v_max, v_T)
        S2 = ds - s1 - S3
        T2 = T - t1 - T3
        valid = (S2 > 0) & (T2 > 0) & (np.abs(T2 - S2 / v_max) < RuleBasedLaneMergePlanner.TIME_GRID_RESOLUTION)
        if not valid.any():
            return []
        S2 = S2[valid]
        T2 = T2[valid]
        S3 = S3[valid]
        T3 = T3[valid]
        T2 = S2 / v_max
        T_tot = t1 + T2 + T3

        actors_s, actors_v, actors_length = np.array([[actor.s_relative_to_ego, actor.velocity, actor.length]
                                                      for actor in state.actors_states]).T
        margins = 0.5 * (actors_length + state.ego_length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
        safety_dist = RuleBasedLaneMergePlanner._caclulate_RSS_distances(
            actors_s[:, np.newaxis], actors_v[:, np.newaxis], margins[:, np.newaxis], v_T[valid], T_tot, ds)
        is_safe = safety_dist > 0

        actions = []
        for t2, s2, t3, s3 in zip(T2[is_safe], S2[is_safe], T3[is_safe], S3[is_safe]):
            action1 = LaneMergeSpec(t1, v_0, a_0, v_max, s1, QuarticPoly1D.num_coefs())
            action2 = LaneMergeSpec(t2, v_max, 0, v_max, s2, poly_coefs_num=2)
            action3 = LaneMergeSpec(t3, v_max, 0, v_T, s3, QuarticPoly1D.num_coefs())
            actions.append(LaneMergeSequence([action1, action2, action3]))

        return actions

    @staticmethod
    def _calculate_safe_target_points(state: LaneMergeState) -> [np.array, np.array, np.array]:
        """
        Create boolean 2D matrix of actions that are longitudinally safe (RSS) between red line & merge point w.r.t.
        all actors.
        :param state: lane merge state
        :return: two 1D arrays of v_T & T, where ego is safe at target_s relatively to all actors
        """
        s_max = state.red_line_s_on_ego_gff - state.ego_fstate_1d[FS_SX]
        ego_length = state.ego_length
        ego_fstate = state.ego_fstate_1d
        actors_states = state.actors_states
        actors_s, actors_v, actors_length = np.array([[actor.s_relative_to_ego, actor.velocity, actor.length]
                                                      for actor in actors_states]).T
        margins = 0.5 * (actors_length + ego_length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT

        # calculate planning time bounds given target_s
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
        t_res = RuleBasedLaneMergePlanner.TIME_GRID_RESOLUTION
        t_grid = np.arange(t_res, LANE_MERGE_ACTION_T_LIMITS[1] + EPS, t_res)
        v_grid = np.arange(0, LANE_MERGE_ACTION_SPACE_MAX_VELOCITY + EPS, RuleBasedLaneMergePlanner.VEL_GRID_RESOLUTION)
        target_v, target_t = np.meshgrid(v_grid, t_grid)
        target_v, target_t = target_v.ravel(), target_t.ravel()
        if len(actors_states) == 0:
            return target_v, target_t

        # 2D matrix with 3 columns of safe actions: target velocities, planning times and distances
        front_bounds, back_bounds = RuleBasedLaneMergePlanner._caclulate_safety_bounds(
            actors_s, actors_v, margins, target_v[:, np.newaxis], target_t[:, np.newaxis])

        # the safe regions are either behind front_bounds or ahead of back_bounds
        bounds = np.concatenate((front_bounds, back_bounds), axis=1)

        # for each target [v,t] sort its bounds by s
        sorted_idxs = bounds.argsort(axis=1)
        sorted_bounds = np.sort(bounds, axis=1)
        sorted_bounds = sorted_bounds[..., np.newaxis]

        # append column of 1 to the front bounds and -1 to the back bounds
        signs = -np.sign(sorted_idxs - len(actors_s) + 0.5)[..., np.newaxis]
        sorted_bounds = np.concatenate((sorted_bounds, np.ones_like(sorted_bounds) * signs), axis=-1)
        # for each bound calculate how many actors make it unsafe
        safety_layer = np.concatenate((np.zeros_like(target_t)[:, np.newaxis], np.cumsum(sorted_bounds[..., 1], axis=-1)), axis=-1)

        # calculate jerk-optimal target s according to the quartic distance formula
        optimal_s = target_t * (target_t * a_0 + 6 * v_0 + 6 * target_v) / 12
        optimal_bound_idx = np.sum(sorted_bounds[..., 0] < optimal_s[:, np.newaxis], axis=-1)

        # collect actions for which optimal_s is safe and for the rest actions collect 2 safe bound closest to optimal_s
        optimal_s_is_safe = safety_layer[range(len(target_t)), optimal_bound_idx] == 0
        optimal_s_is_unsafe = ~optimal_s_is_safe
        unsafe_bound_idx = optimal_bound_idx[optimal_s_is_unsafe]
        non_optimal_under_s = sorted_bounds[np.where(optimal_s_is_unsafe)[0], unsafe_bound_idx-1, 0]
        non_optimal_above_s = sorted_bounds[np.where(optimal_s_is_unsafe)[0], unsafe_bound_idx, 0]

        # calculate all safe target points in the VTS 3D space (v_T, T, ds)
        vts1 = np.c_[target_v[optimal_s_is_safe], target_t[optimal_s_is_safe], optimal_s[optimal_s_is_safe]]
        vts2 = np.c_[target_v[optimal_s_is_unsafe], target_t[optimal_s_is_unsafe], non_optimal_under_s]
        vts3 = np.c_[target_v[optimal_s_is_unsafe], target_t[optimal_s_is_unsafe], non_optimal_above_s]
        vts = np.concatenate((vts1, vts2, vts3), axis=0)
        vts = vts[(vts[:, 2] >= 0) & (vts[:, 2] <= s_max)]
        return vts.T

    @staticmethod
    def _caclulate_safety_bounds(actors_s: np.array, actors_v: np.array, margins: np.array,
                                 target_v: np.array, target_t: np.array) -> [np.array, np.array]:
        """
        Given an actor on the main road and two grids of planning times and target velocities, create boolean matrix
        of all possible actions that are longitudinally safe (RSS) at actions' end-points target_s.
        :param actors_s: current s of actor relatively to the merge point (negative or positive)
        :param actors_v: current actor's velocity
        :param margins: half sum of cars' lengths + safety margin
        :param target_t: array of planning times
        :param target_v: array of target velocities
        :return: tensor of shape: target_v x target_t x bounds_num.
            In even indices are bounds behind an actor, in odd indices are bounds ahead of an actor
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
