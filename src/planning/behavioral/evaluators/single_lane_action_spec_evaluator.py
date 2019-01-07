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

        follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                            if action_specs_mask[i]
                                            and recipe.relative_lane == RelativeLane.SAME_LANE
                                            and recipe.action_type == ActionType.FOLLOW_VEHICLE]

        terminal_velocities = np.unique([recipe.velocity for recipe in action_recipes if isinstance(recipe, StaticActionRecipe)])
        maximal_allowed_velocity = max(terminal_velocities[terminal_velocities <= BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED])

        if len(follow_vehicle_valid_action_idxs) > 0:
            costs[follow_vehicle_valid_action_idxs[0]] = 0
        else:
            follow_lane_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                             if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)
                                             and recipe.relative_lane == RelativeLane.SAME_LANE
                                             and recipe.action_type == ActionType.FOLLOW_LANE
                                             and recipe.velocity <= maximal_allowed_velocity]
            max_valid_velocity = action_recipes[follow_lane_valid_action_idxs[-1]].velocity
            fastest_valid_actions = [i for i in follow_lane_valid_action_idxs
                                     if action_recipes[i].velocity == max_valid_velocity]
            costs[fastest_valid_actions[0]] = 0

        return costs
