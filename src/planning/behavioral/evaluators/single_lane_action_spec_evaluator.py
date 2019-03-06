from logging import Logger
from typing import List

import numpy as np

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe
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
        * If there's a leading vehicle, try following it (ActionType.FOLLOW_LANE, lowest aggressiveness possible)
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_LANE action with maximal
        allowed velocity and lowest aggressiveness possible.
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: semantic actions list.
        :param action_specs: specifications of action_recipes.
        :param action_specs_mask: a boolean mask, showing True where actions_spec is valid (and thus will be evaluated).
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        costs = np.full(len(action_recipes), 1)

        # first try to find a valid dynamic action for SAME_LANE
        follow_vehicle_valid_action_idxs = np.array([[i, behavioral_state.marginal_safety_per_action[i], recipe]
                                                     for i, recipe in enumerate(action_recipes)
                                                     if action_specs_mask[i]
                                                     and recipe.relative_lane == RelativeLane.SAME_LANE
                                                     and recipe.action_type == ActionType.FOLLOW_VEHICLE])
        if len(follow_vehicle_valid_action_idxs) > 0:
            best_idx = np.argmax(follow_vehicle_valid_action_idxs[:, 1])
            min_safe_dist = follow_vehicle_valid_action_idxs[best_idx, 1]
            if min_safe_dist > 5:   # safe enough
                chosen_idx = 0      # calm action
            elif len(follow_vehicle_valid_action_idxs) == 3 and min_safe_dist > 3:  # moderate safety
                chosen_idx = 1      # standard action
            else:                   # low safety
                chosen_idx = -1     # aggressive action
            costs[int(follow_vehicle_valid_action_idxs[chosen_idx, 0])] = 0  # choose the found dynamic action
            return costs

        # if a dynamic action not found, calculate maximal valid existing velocity for same-lane static actions
        terminal_velocities = np.unique([recipe.velocity for i, recipe in enumerate(action_recipes)
                                         if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                                         and recipe.relative_lane == RelativeLane.SAME_LANE])
        maximal_allowed_velocity = max(terminal_velocities[terminal_velocities <= BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED])

        # find the most calm same-lane static action with the maximal existing velocity
        follow_lane_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                         if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                                         and recipe.relative_lane == RelativeLane.SAME_LANE
                                         and recipe.velocity == maximal_allowed_velocity]

        costs[follow_lane_valid_action_idxs[0]] = 0  # choose the found static action
        return costs
