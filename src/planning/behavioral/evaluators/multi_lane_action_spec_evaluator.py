from decision_making.src.exceptions import NoActionsLeftForBPError
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionSpec, ActionType, RelativeLane, \
    StaticActionRecipe
from decision_making.src.planning.behavioral.evaluators.action_evaluator import \
    ActionSpecEvaluator
from decision_making.src.global_constants import EPS


class MultiLaneActionSpecEvaluator(ActionSpecEvaluator):
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

        # first try to find actions that follow vehicles
        follow_vehicle_valid_action_idxs = [i for i, recipe in enumerate(action_recipes)
                                            if action_specs_mask[i]
                                            and recipe.action_type == ActionType.FOLLOW_VEHICLE]
        if len(follow_vehicle_valid_action_idxs) > 0:
            costs[follow_vehicle_valid_action_idxs] = 0  # give the follow vehicle actions priority


        filtered_indices = [i for i, recipe in enumerate(action_recipes)
                            if action_specs_mask[i] and isinstance(recipe, StaticActionRecipe)]

        if len(filtered_indices) + len(follow_vehicle_valid_action_idxs) == 0:
            raise NoActionsLeftForBPError()

        # find the minimal aggressiveness level among valid static recipes
        min_aggr_level = min([action_recipes[idx].aggressiveness.value for idx in filtered_indices])

        # find the most fast action with the minimal aggressiveness level
        follow_lane_valid_action_idxs = [idx for idx in filtered_indices
                                         if action_recipes[idx].aggressiveness.value == min_aggr_level]

        # choose the most fast action among the calmest actions;
        # it's last in the recipes list since the recipes are sorted in the increasing order of velocities
        costs[follow_lane_valid_action_idxs[-1]] = 0

        # assign a cost to all the calmest actions based on their velocity, assign EPS to 0 speed to avoid division by 0
        action_velocities = [action_recipes[idx].velocity
                             if action_recipes[idx].velocity > EPS else EPS
                             for idx in follow_lane_valid_action_idxs]

        # normalize velocities to be between 0 and 1
        action_velocities /= np.max(np.abs(action_velocities), axis=0)

        # assign costs to follow lane actions (higher speeds should have lower costs)
        for i, action_idx in enumerate(follow_lane_valid_action_idxs):
            costs[action_idx]  = 1 - action_velocities[i]

        return costs
