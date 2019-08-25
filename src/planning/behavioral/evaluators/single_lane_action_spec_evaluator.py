from logging import Logger
from typing import List

import numpy as np

from decision_making.src.planning.behavioral.data_objects import ActionType, RelativeLane, ActionSpec


class SingleLaneActionsEvaluator:

    @staticmethod
    def evaluate(action_specs: List[ActionSpec]) -> np.ndarray:
        """
        Evaluates Action-Specifications based on the following logic:
        * Only takes into account actions on RelativeLane.SAME_LANE
        * the lowest costs get actions with lower aggressiveness levels, and for the same aggressiveness level
          the lower costs get actions with higher velocities
        :param action_specs: actions specifications list
        :return: numpy array of costs of semantic actions
        """
        costs = np.full(len(action_specs), np.inf)

        # fill the lowest costs for dynamic actions for SAME_LANE
        for i, spec in enumerate(action_specs):
            recipe = spec.recipe
            if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.action_type == ActionType.FOLLOW_VEHICLE:
                costs[i] = recipe.aggressiveness.value

        # find maximal velocity among static actions
        max_velocity = np.max(np.array([spec.v for i, spec in enumerate(action_specs)
                                        if spec.recipe.action_type == ActionType.FOLLOW_LANE]))

        # fill costs for SAME_LANE static actions, such that the lowest costs get lower aggressiveness levels,
        # and for the same aggressiveness level the lower costs get actions with higher velocities
        from_cost = np.max(costs[~np.isinf(costs)]) + 1  # get maximal non-inf cost
        for i, spec in enumerate(action_specs):
            recipe = spec.recipe
            if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.action_type == ActionType.FOLLOW_LANE:
                costs[i] = from_cost + recipe.aggressiveness.value * (max_velocity + 1) + (max_velocity - recipe.velocity)
        return costs
