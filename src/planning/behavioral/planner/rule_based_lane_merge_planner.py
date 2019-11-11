from logging import Logger
from typing import List
import numpy as np
from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.global_constants import SAFETY_HEADWAY, LON_ACC_LIMITS, EPS, TRAJECTORY_TIME_RESOLUTION, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, ActionType, ActionSpec, \
    StaticActionRecipe
from decision_making.src.planning.behavioral.default_config import DEFAULT_ACTION_SPEC_FILTERING
from decision_making.src.planning.behavioral.evaluators.augmented_lane_action_spec_evaluator import \
    AugmentedLaneActionSpecEvaluator
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.src.planning.types import LIMIT_MIN, FS_SX, FS_SV, FS_SA, ActionSpecArray
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D
from decision_making.src.state.state import State

WORST_CASE_FRONT_CAR_DECEL = 3.  # [m/sec^2]
WORST_CASE_BACK_CAR_ACCEL = 1.  # [m/sec^2]
MAX_VELOCITY = 25.
OUTPUT_TRAJECTORY_LENGTH = 10
ACTION_T_LIMITS = np.array([0, 30])


class ScenarioParams:
    def __init__(self, worst_case_back_actor_accel: float = WORST_CASE_BACK_CAR_ACCEL,
                 worst_case_front_actor_decel: float = WORST_CASE_FRONT_CAR_DECEL,
                 ego_reaction_time: float = SAFETY_HEADWAY, back_actor_reaction_time: float = SAFETY_HEADWAY,
                 front_rss_decel: float = -LON_ACC_LIMITS[LIMIT_MIN], back_rss_decel: float = 3.8,
                 ego_max_velocity: float = MAX_VELOCITY, actors_max_velocity: float = MAX_VELOCITY):
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


class RuleBasedLaneMergePlanner(BasePlanner):

    def __init__(self, logger: Logger):
        super().__init__(logger)

    @staticmethod
    def _create_safe_distances_for_max_vel_quartic_actions(state: LaneMergeState, params: ScenarioParams = ScenarioParams()) -> \
            [np.array, np.array]:
        """
        Check existence of rule-based solution that can merge safely, assuming the worst case scenario of
        main road actors. The function tests a single static action toward maximal velocity (ScenarioParams.ego_max_velocity).
        If the action is safe (longitudinal RSS) during crossing red line w.r.t. all main road actors, return True.
        :param state: lane merge state, containing data about host and the main road vehicles
        :param params: scenario parameters, describing the worst case actors' behavior and RSS safety parameters
        :return: accelerations array or None if there is no safe action
        """
        s_0, v_0, a_0 = state.ego_fstate_1d
        rel_red_line_s = state.red_line_s_on_ego_gff - s_0
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS.T
        specs_t, specs_s = RuleBasedLaneMergePlanner._specify_quartic(v_0, a_0, params.ego_max_velocity, w_T, w_J)

        # validate lon. acceleration limits and choose the most calm valid action
        valid_acc, poly_s = RuleBasedLaneMergePlanner._validate_acceleration(v_0, a_0, params.ego_max_velocity, specs_t)
        poly_s = poly_s[valid_acc]
        specs_s = specs_s[valid_acc]
        specs_t = specs_t[valid_acc]

        # initialize target points for the actions that end before the red line (specs_s < rel_red_line_s)
        target_t = specs_t + (rel_red_line_s - specs_s) / params.ego_max_velocity
        target_v = np.full(specs_t.shape[0], params.ego_max_velocity, dtype=float)
        target_s = np.full(specs_t.shape[0], rel_red_line_s, dtype=float)

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

        actors = state.actors_states
        if len(actors) > 0:
            # get the actors' data
            actors_data = np.array([[actor.s_relative_to_ego, actor.velocity, actor.length] for actor in actors])
            actors_s = actors_data[:, 0, np.newaxis]
            actors_v = actors_data[:, 1, np.newaxis]
            margins = 0.5 * (actors_data[:, 2, np.newaxis] + state.ego_length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT

            # check RSS safety at red line, assuming the worst case scenario; the output is 1D of size len(specs_t)
            # the output for each action is the difference (minimum over all actors) between actual distance from actor
            # and safe distance
            safety_dist = RuleBasedLaneMergePlanner._create_RSS_matrix(
                actors_s, actors_v, margins, target_v, target_t, target_s,
                params.worst_case_front_actor_decel, params.worst_case_back_actor_accel,
                params.ego_reaction_time, params.back_actor_reaction_time, params.front_rss_decel, params.back_rss_decel,
                params.actors_max_velocity)  # enable back actor to exceed max_vel to remain consistent with control errors
        else:
            safety_dist = np.ones(len(specs_t))

        return poly_s, target_t, target_v, safety_dist

    @staticmethod
    def _specify_quartic(v_0: float, a_0: float, v_T: np.array, w_T: np.array, w_J: np.array) -> [np.array, np.array]:
        # T_s <- find minimal non-complex local optima within the ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuarticPoly1D.time_cost_function_derivative_coefs(w_T=w_T, w_J=w_J, a_0=a_0, v_0=v_0, v_T=v_T)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, ACTION_T_LIMITS)
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
        nonnan = ~np.isnan(T)
        positiveT = np.copy(nonnan)
        valid_acc = np.copy(nonnan)
        positiveT[nonnan] = np.greater(T[nonnan], 0)
        valid_acc[nonnan] = np.equal(T[nonnan], 0)
        poly_s[positiveT] = QuarticPoly1D.position_profile_coefficients(a_0, v_0, v_T if np.isscalar(v_T) else v_T[positiveT], T[positiveT])
        valid_acc[positiveT] = QuarticPoly1D.are_accelerations_in_limits(poly_s[positiveT], T[positiveT], LON_ACC_LIMITS)
        return valid_acc, poly_s

    @staticmethod
    def _create_RSS_matrix(actors_s: np.array, actors_v: np.array, margins: np.array,
                           target_v: np.array, target_t: np.array, target_s: np.array,
                           wc_front_decel: float, wc_back_accel: float, ego_hw: float, actor_hw: float,
                           rss_front_decel: float, rss_back_decel: float, actors_max_vel: float) -> BoolArray:
        """
        Given an actor on the main road and two grids of planning times and target velocities, create boolean matrix
        of all possible actions that are longitudinally safe (RSS) at actions' end-points target_s.
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
        front_braking_time = np.minimum(actors_v / wc_front_decel, target_t) if wc_front_decel > 0 else target_t
        front_v = actors_v - front_braking_time * wc_front_decel  # target velocity of the front actor
        front_s = actors_s + 0.5 * (actors_v + front_v) * front_braking_time

        back_accel_time = np.minimum((actors_max_vel - actors_v) / wc_back_accel, target_t) if wc_back_accel > 0 else target_t
        back_max_vel_time = target_t - back_accel_time  # time of moving with maximal velocity of the back actor
        back_v = actors_v + back_accel_time * wc_back_accel  # target velocity of the back actor
        back_s = actors_s + 0.5 * (actors_v + back_v) * back_accel_time + actors_max_vel * back_max_vel_time

        # calculate if ego is safe according to the longitudinal RSS formula
        front_safety_dist = front_s - target_s - margins - \
                            (np.maximum(0, target_v * target_v - front_v * front_v) / (2 * rss_front_decel) +
                             ego_hw * target_v)
        back_safety_dist = target_s - back_s - margins - \
                           (np.maximum(0, back_v * back_v - target_v * target_v) / (2 * rss_back_decel) +
                            actor_hw * back_v)
        safety_dist = np.maximum(front_safety_dist, back_safety_dist)
        return np.min(safety_dist, axis=0)

    def _create_behavioral_state(self, state: State, route_plan: RoutePlan) -> LaneMergeState:
        return LaneMergeState.create_from_state(state, route_plan, self.logger)

    def _create_action_specs(self, lane_merge_state: LaneMergeState) -> ActionSpecArray:
        v_0, a_0 = lane_merge_state.ego_fstate_1d[FS_SV], lane_merge_state.ego_fstate_1d[FS_SA]
        ds = lane_merge_state.red_line_s_on_ego_gff - lane_merge_state.ego_fstate_1d[FS_SX]
        poly_coefs_s, specs_t, specs_v, safety_dist = \
            RuleBasedLaneMergePlanner._create_safe_distances_for_max_vel_quartic_actions(lane_merge_state)
        action_specs = [ActionSpec(t, v_T, ds, d=0, recipe=StaticActionRecipe(relative_lane=, v_T,))
                        for t, v_T, coefs_s in zip(specs_t[safety_dist > 0], specs_v[safety_dist > 0], poly_coefs_s[safety_dist > 0])]
        return action_specs

    def _filter_actions(self, lane_merge_state: LaneMergeState, action_specs: ActionSpecArray) -> ActionSpecArray:
        action_specs_mask = DEFAULT_ACTION_SPEC_FILTERING.filter_action_specs(action_specs, lane_merge_state)
        filtered_action_specs = np.full(len(action_specs), None)
        filtered_action_specs[action_specs_mask] = action_specs[action_specs_mask]
        return filtered_action_specs

    def _evaluate_actions(self, lane_merge_state: LaneMergeState, route_plan: RoutePlan,
                          action_specs: ActionSpecArray) -> np.array:
        """
        Evaluates Action-Specifications based on the following logic:
        * lowest aggressiveness possible
        :param lane_merge_state: lane merge state
        :param route_plan:
        :param action_specs: specifications of action_recipes.
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        action_specs_exist = action_specs.astype(bool)
        if not action_specs_exist.any():
            raise NoActionsLeftForBPError("All actions were filtered in BP. timestamp_in_sec: %f" %
                                          lane_merge_state.ego_state.timestamp_in_sec)
        action_costs = np.full(len(action_specs), 1.)
        most_calm_action_idx = np.argmax(action_specs_exist)
        action_costs[most_calm_action_idx] = 0
        return action_costs

    def _choose_action(self, lane_merge_state: LaneMergeState, actions: np.array, costs: np.array) -> \
            [ActionRecipe, ActionSpec]:
        """
        pick the first action_spec from the best LaneMergeSequence having the minimal cost
        :param actions: array of LaneMergeSequence
        :param costs: array of actions' costs
        :return: [ActionSpec] the first action_spec in the best LaneMergeSequence
        """
        # choose the first spec of the best action having the minimal cost
        best_lane_merge_spec: LaneMergeSpec = actions[np.argmin(costs)].action_specs[0]
        # convert spec.s from LaneMergeState to be relative to GFF
        recipe = ActionRecipe(RelativeLane.SAME_LANE, ActionType.FOLLOW_LANE, AggressivenessLevel.CALM)
        ego_s = lane_merge_state.projected_ego_fstates[RelativeLane.SAME_LANE][FS_SX]

        # the first spec of the chosen action may be too short to complete the full lateral movement,
        # therefore we choose non-zero lateral state for the target spec
        ego_fstate = lane_merge_state.ego_state.map_state.lane_fstate
        w_J = BP_JERK_S_JERK_D_TIME_WEIGHTS[0:1, 1]  # lateral jerk weight
        w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[0:1, 2]
        T_d = RuleBasedLaneMergePlanner._specify_quintic(ego_fstate[FS_DV], ego_fstate[FS_DA], 0, -ego_fstate[FS_DX], w_T, w_J)[0]
        dx = dv = da = 0
        if T_d > best_lane_merge_spec.t:
            poly_coefs_d = QuinticPoly1D.solve_1d_bvp(np.concatenate((ego_fstate[FS_DX:], np.zeros(FS_1D_LEN)))[np.newaxis], T_d)
            dx, dv, da = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, np.array([best_lane_merge_spec.t]))[0][0]

        chosen_spec = ActionSpec(best_lane_merge_spec.t, best_lane_merge_spec.v_T, best_lane_merge_spec.ds + ego_s, 0, recipe)
#                                 dx, dv, da, recipe)
        print('chosen_spec=', best_lane_merge_spec.__str__())
        return [recipe, chosen_spec]
