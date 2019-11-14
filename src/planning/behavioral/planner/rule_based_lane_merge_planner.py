from logging import Logger
from typing import List
import numpy as np
from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.global_constants import SAFETY_HEADWAY, LON_ACC_LIMITS, EPS, TRAJECTORY_TIME_RESOLUTION, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, BP_JERK_S_JERK_D_TIME_WEIGHTS, MAX_BACKWARD_HORIZON
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, ActionSpec, StaticActionRecipe, \
    RelativeLane, ActionRecipe
from decision_making.src.planning.behavioral.default_config import DEFAULT_ACTION_SPEC_FILTERING
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState, LaneMergeActorState
from decision_making.src.planning.types import ActionSpecArray, BoolArray
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D
from decision_making.src.state.state import State

WORST_CASE_FRONT_CAR_DECEL = 3.     # [m/sec^2]
WORST_CASE_BACK_CAR_ACCEL = 1.      # [m/sec^2]
BACK_CAR_RSS_DECELERATION = 3.8     # [m/sec^2]
LANE_MERGE_HOST_MAX_VELOCITY = 25.  # [m/sec]
LANE_MERGE_ACTORS_MAX_VELOCITY = 25.  # [m/sec]
OUTPUT_TRAJECTORY_LENGTH = 10
ACTION_T_LIMITS = np.array([0, 30])   # [sec]


class RuleBasedLaneMergePlanner(BasePlanner):

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
        v_T = LANE_MERGE_HOST_MAX_VELOCITY
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS.T

        # specify 3 static actions with 3 aggressiveness levels
        specs_t, specs_s = RuleBasedLaneMergePlanner._specify_quartic(v_0, a_0, v_T, w_T, w_J)

        # create action specs based on the above output
        return [ActionSpec(t=T, v=v_T, s=lane_merge_state.red_line_s_on_ego_gff, d=0,
                           recipe=StaticActionRecipe(RelativeLane.SAME_LANE, v_T, aggr_level))
                for T, aggr_level in zip(specs_t, AggressivenessLevel)]

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
        return RuleBasedLaneMergePlanner._safety_filter(lane_merge_state, filtered_action_specs)

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
            raise NoActionsLeftForBPError("RB evaluate_actions: All actions were filtered in BP. timestamp_in_sec: %f" %
                                          lane_merge_state.ego_state.timestamp_in_sec)
        action_costs = np.full(len(action_specs), 1.)
        most_calm_action_idx = np.argmax(action_specs_exist)
        action_costs[most_calm_action_idx] = 0
        return action_costs

    def _choose_action(self, lane_merge_state: LaneMergeState, action_specs: ActionSpecArray, costs: np.array) -> \
            [ActionRecipe, ActionSpec]:
        """
        pick the first action_spec from the best LaneMergeSequence having the minimal cost
        :param lane_merge_state: lane merge state
        :param action_specs: array of ActionSpecs
        :param costs: array of actions' costs
        :return: [ActionSpec] the first action_spec in the best LaneMergeSequence
        """
        selected_action_index = int(np.argmin(costs))
        return action_specs[selected_action_index].recipe, action_specs[selected_action_index]

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
        v_T = LANE_MERGE_HOST_MAX_VELOCITY
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

        action_specs[action_specs != None][safety_dist <= 0] = None
        return action_specs

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
        front_braking_time = np.minimum(actors_v / WORST_CASE_FRONT_CAR_DECEL, target_t) \
            if WORST_CASE_FRONT_CAR_DECEL > 0 else target_t
        front_v = actors_v - front_braking_time * WORST_CASE_FRONT_CAR_DECEL  # target velocity of the front actor
        front_s = actors_s + 0.5 * (actors_v + front_v) * front_braking_time

        back_accel_time = np.minimum((LANE_MERGE_ACTORS_MAX_VELOCITY - actors_v) / WORST_CASE_BACK_CAR_ACCEL, target_t) \
            if WORST_CASE_BACK_CAR_ACCEL > 0 else target_t
        back_max_vel_time = target_t - back_accel_time  # time of moving with maximal velocity of the back actor
        back_v = actors_v + back_accel_time * WORST_CASE_BACK_CAR_ACCEL  # target velocity of the back actor
        back_s = actors_s + 0.5 * (actors_v + back_v) * back_accel_time + LANE_MERGE_ACTORS_MAX_VELOCITY * back_max_vel_time

        # calculate if ego is safe according to the longitudinal RSS formula
        front_safety_dist = front_s - target_s - margins - (np.maximum(0, target_v * target_v - front_v * front_v) /
                                                            (-2 * LON_ACC_LIMITS[0]) + SAFETY_HEADWAY * target_v)
        back_safety_dist = target_s - back_s - margins - (np.maximum(0, back_v * back_v - target_v * target_v) /
                                                          (2 * LON_ACC_LIMITS[1]) + SAFETY_HEADWAY * back_v)
        safety_dist = np.maximum(front_safety_dist, back_safety_dist)
        return np.min(safety_dist, axis=0)  # AND on actors
