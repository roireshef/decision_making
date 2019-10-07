from decision_making.src.exceptions import NoActionsLeftForBPError
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.planning.behavioral.evaluators.lane_based_action_spec_evaluator import LaneBasedActionSpecEvaluator
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe, AggressivenessLevel
from decision_making.src.planning.behavioral.evaluators.action_evaluator import \
    ActionSpecEvaluator
from decision_making.src.messages.route_plan_message import RoutePlan


class SingleLaneActionSpecEvaluator(LaneBasedActionSpecEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool], route_plan: RoutePlan) -> np.ndarray:
        """
        Evaluates Action-Specifications based on the following logic:
        * Only takes into account actions on RelativeLane.SAME_LANE
        * If there's a leading vehicle, try following it (ActionType.FOLLOW_VEHICLE, lowest aggressiveness possible)
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_ROAD_SIGN action with lowest
        * aggressiveness, and save it.
        * Find the ActionType.FOLLOW_LANE action with maximal allowed velocity and lowest aggressiveness possible,
        * and save it.
        * Compare the saved FOLLOW_ROAD_SIGN and FOLLOW_LANE actions, and choose between them.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :param route_plan: the route plan which contains lane costs
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        costs = np.full(len(action_recipes), 1)


        # first try to find a valid dynamic action (FOLLOW_VEHICLE) for SAME_LANE
        selected_follow_vehicle_idx = self._get_follow_vehicle_valid_action_idx(behavioral_state, action_recipes,
                                                                                action_specs, action_specs_mask,
                                                                                RelativeLane.SAME_LANE)
        if selected_follow_vehicle_idx >= 0:
            costs[selected_follow_vehicle_idx] = 0  # choose the found dynamic action, which is least aggressive
            return costs

        # next try to find a valid road sign action (FOLLOW_ROAD_SIGN) for SAME_LANE.
        # Tentative decision is kept in selected_road_sign_idx, to be compared against STATIC actions
        selected_road_sign_idx = self._get_follow_road_sign_valid_action_idx(action_recipes, action_specs_mask, RelativeLane.SAME_LANE)

        # last, look for valid static action
        selected_follow_lane_idx = self._get_follow_lane_valid_action_idx(action_recipes, action_specs_mask, RelativeLane.SAME_LANE)

        # finally decide between the road sign and the static action
        if selected_road_sign_idx < 0 and selected_follow_lane_idx < 0:
            # if no action of either type was found, raise an error
            raise NoActionsLeftForBPError("All actions were filtered in BP. timestamp_in_sec: %f" %
                                          behavioral_state.ego_state.timestamp_in_sec)
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

