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
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_ROAD_SGN action with lowest
        * aggressiveness.
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_LANE action with maximal
        allowed velocity and lowest aggressiveness possible.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        costs = np.full(len(action_recipes), 1)

        # first try to find a valid dynamic action of type FOLLOW_VEHICLE for SAME_LANE
        follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                            if action_specs_mask[i]
                                            and recipe.relative_lane == RelativeLane.SAME_LANE
                                            and recipe.action_type == ActionType.FOLLOW_VEHICLE]
        # The selection is only by aggressiveness, since it relies on the fact that we only follow a vehicle on the
        # SAME lane, which means there is only 1 possible vehicle to follow, so there is only 1 target vehicle speed.
        if len(follow_vehicle_valid_action_idxs) > 0:
            costs[follow_vehicle_valid_action_idxs[0]] = 0  # choose the found dynamic action, which is least aggressive
            return costs

        # next try to find a valid dynamic action of type FOLLOW_ROAD_SIGN for SAME_LANE. Only need to consider
        # aggressiveness level, as all the target speeds are ZERO_SPEED.
        follow_road_sign_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                              if action_specs_mask[i]
                                              and recipe.relative_lane == RelativeLane.SAME_LANE
                                              and recipe.action_type == ActionType.FOLLOW_ROAD_SIGN]
        if len(follow_road_sign_valid_action_idxs) > 0:
            # TODO DEBUG REMOVE
            print('\x1b[5;30;43m', "available STOP aggressiveness", [action.aggressiveness for action, mask in zip(action_recipes, action_specs_mask)
                                                                     if action.action_type == ActionType.FOLLOW_ROAD_SIGN and mask], '\x1b[0m')
            # TODO DEBUG REMOVE
            # choose the found action, which is least aggressive.
            # Will be used if no proper static action is found
            tentative_road_sign_idx = follow_road_sign_valid_action_idxs[0]
        else:
            tentative_road_sign_idx = -1

        # last, look for valid static action
        filtered_indices = [i for i, recipe in enumerate(action_recipes)
                            if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                            and recipe.relative_lane == RelativeLane.SAME_LANE]
        if len(filtered_indices) > 0:
            # find the minimal aggressiveness level among valid static recipes
            min_aggr_level = min([action_recipes[idx].aggressiveness.value for idx in filtered_indices])

            # among the minimal aggressiveness level, find the fastest action
            follow_lane_valid_action_idxs = [idx for idx in filtered_indices
                                             if action_recipes[idx].aggressiveness.value == min_aggr_level]

            selected_follow_lane_idx = follow_lane_valid_action_idxs[-1]
        else:
            selected_follow_lane_idx = -1

        # now decide between the road sign and the static action
        if tentative_road_sign_idx < 0 and selected_follow_lane_idx < 0:
            raise NoActionsLeftForBPError()
        elif tentative_road_sign_idx < 0:
            print('\x1b[5;30;43m', "available LANE velocities",
                  [action.velocity for i, action in enumerate(action_recipes)
                   if i in follow_lane_valid_action_idxs], "aggr", min_aggr_level, '\x1b[0m')
            print('\x1b[5;30;43m', "selected action FOLLOW_LANE velocity",
                  action_recipes[selected_follow_lane_idx].velocity, "aggr",
                  action_recipes[selected_follow_lane_idx].aggressiveness, "tentative",
                  tentative_road_sign_idx, '\x1b[0m')
            costs[selected_follow_lane_idx] = 0
            return costs
        elif selected_follow_lane_idx < 0:
            costs[tentative_road_sign_idx] = 0
            print('\x1b[5;30;43m', "selected action STOP aggressiveness",
                  action_recipes[tentative_road_sign_idx].aggressiveness, '\x1b[0m')
            return costs
        else:  # both road sign and static actions are valid - choose
            if self._is_static_action_preferred(action_recipes, tentative_road_sign_idx, selected_follow_lane_idx):
                print('\x1b[5;30;43m', "available LANE velocities",
                      [action.velocity for i, action in enumerate(action_recipes)
                       if i in follow_lane_valid_action_idxs], "aggr", min_aggr_level, '\x1b[0m')
                print('\x1b[5;30;43m', "selected action FOLLOW_LANE velocity",
                      action_recipes[selected_follow_lane_idx].velocity, "aggr",
                      action_recipes[selected_follow_lane_idx].aggressiveness, "tentative",
                      tentative_road_sign_idx, '\x1b[0m')
                costs[selected_follow_lane_idx] = 0
                return costs
            else:
                print('\x1b[5;30;43m', "selected action STOP aggressiveness",
                      action_recipes[tentative_road_sign_idx].aggressiveness, '\x1b[0m')
                costs[tentative_road_sign_idx] = 0
                return costs

    def _is_static_action_preferred(self, action_recipes: List[ActionRecipe], road_sign_idx: int, follow_lane_idx: int):
        static_action = action_recipes[follow_lane_idx]
        road_sign_action = action_recipes[road_sign_idx]
        # return static_action.velocity > 10  # Naive

        # This selects many STATIC actions, and is even non-consistent, until finally selecting STOP
        # return static_action.aggressiveness.value < road_sign_action.aggressiveness.value or \
        #     (static_action.aggressiveness.value == road_sign_action.aggressiveness.value and static_action.velocity > 0)

        # This starts from AGGRESSIVE which accelerates, and only then moves to STANDARD / CALM
        # return False

        # Avoid AGGRESSIVE stop
        return road_sign_action.aggressiveness == AggressivenessLevel.AGGRESSIVE
