from decision_making.src.exceptions import NoActionsLeftForBPError
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe, AggressivenessLevel
from decision_making.src.planning.behavioral.evaluators.action_evaluator import \
    ActionSpecEvaluator


class SingleLaneActionSpecEvaluator(ActionSpecEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool]) -> np.ndarray:
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
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        costs = np.full(len(action_recipes), 1)

        # first try to find a valid dynamic action (FOLLOW_VEHICLE) for SAME_LANE
        follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                            if action_specs_mask[i]
                                            and recipe.relative_lane == RelativeLane.SAME_LANE
                                            and recipe.action_type == ActionType.FOLLOW_VEHICLE]
        # The selection is only by aggressiveness, since it relies on the fact that we only follow a vehicle on the
        # SAME lane, which means there is only 1 possible vehicle to follow, so there is only 1 target vehicle speed.
        if len(follow_vehicle_valid_action_idxs) > 0:
            costs[follow_vehicle_valid_action_idxs[0]] = 0  # choose the found dynamic action, which is least aggressive
            return costs

        # next try to find a valid road sign action (FOLLOW_ROAD_SIGN) for SAME_LANE.
        # Selection only needs to consider aggressiveness level, as all the target speeds are ZERO_SPEED.
        # Tentative decision is kept in selected_road_sign_idx, to be compared against STATIC actions
        follow_road_sign_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                              if action_specs_mask[i]
                                              and recipe.relative_lane == RelativeLane.SAME_LANE
                                              and recipe.action_type == ActionType.FOLLOW_ROAD_SIGN]
        if len(follow_road_sign_valid_action_idxs) > 0:
            # choose the found action, which is least aggressive.
            selected_road_sign_idx = follow_road_sign_valid_action_idxs[0]
        else:
            selected_road_sign_idx = -1

        # last, look for valid static action
        filtered_follow_lane_idxs = [i for i, recipe in enumerate(action_recipes)
                            if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                            and recipe.relative_lane == RelativeLane.SAME_LANE]
        if len(filtered_follow_lane_idxs) > 0:
            # find the minimal aggressiveness level among valid static recipes
            min_aggr_level = min([action_recipes[idx].aggressiveness.value for idx in filtered_follow_lane_idxs])

            # among the minimal aggressiveness level, find the fastest action
            follow_lane_valid_action_idxs = [idx for idx in filtered_follow_lane_idxs
                                             if action_recipes[idx].aggressiveness.value == min_aggr_level]

            selected_follow_lane_idx = follow_lane_valid_action_idxs[-1]
        else:
            selected_follow_lane_idx = -1

        # finally decide between the road sign and the static action
        if selected_road_sign_idx < 0 and selected_follow_lane_idx < 0:
            # if no action of either type was found, raise an error
            raise NoActionsLeftForBPError()
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

    def _is_static_action_preferred(self, action_recipes: List[ActionRecipe], road_sign_idx: int, follow_lane_idx: int):
        """
        Selects if a STATIC or ROAD_SIGN action is preferred.
        :param action_recipes: of all possible actions
        :param road_sign_idx: of calmest road sign action
        :param follow_lane_idx: of calmest fastest static action
        :return: True if static action is preferred, False otherwise
        """
        road_sign_action = action_recipes[road_sign_idx]
        # Avoid AGGRESSIVE stop. TODO relax the restriction of not selective an aggressive road sign
        return road_sign_action.aggressiveness == AggressivenessLevel.AGGRESSIVE
