from decision_making.src.exceptions import NoActionsLeftForBPError
from logging import Logger
from typing import List
import numpy as np

from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, RelativeLane
from decision_making.src.planning.behavioral.evaluators.lane_based_action_spec_evaluator import LaneBasedActionSpecEvaluator
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.messages.turn_signal_message import TurnSignalState


class AugmentedLaneActionSpecEvaluator(LaneBasedActionSpecEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool], route_plan: RoutePlan) -> np.ndarray:
        """
        Evaluates Action-Specifications based on the following logic:
        * First chooses the relative with the lowest cost based on route_plan
        * Only looks at actions from this minimal cost lane
        * If there's a leading vehicle, try following it (ActionType.FOLLOW_LANE, lowest aggressiveness possible)
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_LANE action with maximal
        allowed velocity and lowest aggressiveness possible.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :param route_plan: the route plan which contains lane costs
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """

        # Choose the minimum cost lane based on route plan costs.
        # The minimum cost lane is defined as the lane who has the minimum cost at the first point where
        # it diverges from the SAME_LANE.
        minimum_cost_lane = self._find_min_cost_augmented_lane(behavioral_state, route_plan)

        costs = np.full(len(action_recipes), 1)

        # if an augmented lane is chosen to be the minimum_cost_lane or a lane change is desired, also allow the possibility of choosing an
        # action on the straight lane if no actions are available on the augmented lane or the desired lane for a lane change. Prioritize
        # the lane change lane over the minimum_cost_lane.
        lanes_to_try = []

        # If a turn signal is on, prioritize actions towards that lanes by placing the respective relative lane first in the list.
        if behavioral_state.ego_state.turn_signal.s_Data.e_e_turn_signal_state == TurnSignalState.CeSYS_e_LeftTurnSignalOn:
            lanes_to_try = [RelativeLane.LEFT_LANE]
        elif behavioral_state.ego_state.turn_signal.s_Data.e_e_turn_signal_state == TurnSignalState.CeSYS_e_RightTurnSignalOn:
            lanes_to_try = [RelativeLane.RIGHT_LANE]

        # Append SAME_LANE and minimum_cost_lane accordingly
        if minimum_cost_lane != RelativeLane.SAME_LANE:
            lanes_to_try.append(minimum_cost_lane)
            lanes_to_try.append(RelativeLane.SAME_LANE)
        else:
            lanes_to_try.append(RelativeLane.SAME_LANE)

        # If we're currently performing a lane change and in the target lane, override the previous lanes_to_try and prioritize actions
        # in the same lane in order to complete the maneuver.
        if behavioral_state.ego_state.lane_change_info.lane_change_active:
            is_host_in_target_lane = behavioral_state.ego_state.lane_change_info.are_target_lane_ids_in_gff(
                behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE])

            if is_host_in_target_lane:
                lanes_to_try = [RelativeLane.SAME_LANE]

        for target_lane in lanes_to_try:
            # first try to find a valid dynamic action (FOLLOW_VEHICLE) for SAME_LANE
            selected_follow_vehicle_idx = self._get_follow_vehicle_valid_action_idx(behavioral_state, action_recipes,
                                                                                    action_specs, action_specs_mask,
                                                                                    target_lane)
            if selected_follow_vehicle_idx >= 0:
                costs[selected_follow_vehicle_idx] = 0  # choose the found dynamic action, which is least aggressive
                return costs

            # next try to find a valid road sign action (FOLLOW_ROAD_SIGN)
            # Tentative decision is kept in selected_road_sign_idx, to be compared against STATIC actions
            selected_road_sign_idx = self._get_follow_road_sign_valid_action_idx(action_recipes, action_specs_mask,
                                                                                 target_lane)

            # last, look for valid static action
            selected_follow_lane_idx = self._get_follow_lane_valid_action_idx(action_recipes, action_specs_mask,
                                                                              target_lane)

            # finally decide between the road sign and the static action
            if selected_road_sign_idx < 0 and selected_follow_lane_idx < 0:
                # if no action of either type was found, skip checking for actions on the lane
                continue
            elif selected_road_sign_idx < 0:
                # if no road sign action is found, select the static action
                costs[selected_follow_lane_idx] = 0
                return costs
            elif selected_follow_lane_idx < 0:
                # if no static action is found, select the road sign action
                costs[selected_road_sign_idx] = 0
                return costs
            else:
                # if both road sign and static actions are valid - choose
                if self._is_static_action_preferred(action_recipes, selected_road_sign_idx, selected_follow_lane_idx):
                    costs[selected_follow_lane_idx] = 0
                    return costs
                else:
                    costs[selected_road_sign_idx] = 0
                    return costs

        # if the for loop is exited without having returned anything, no valid actions were found in both
        # minimum_cost_lane as well as the SAME_LANE
        raise NoActionsLeftForBPError("All actions were filtered in BP. timestamp_in_sec: %f" %
                                      behavioral_state.ego_state.timestamp_in_sec)

