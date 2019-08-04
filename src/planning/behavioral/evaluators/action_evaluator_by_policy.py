import numpy as np
from logging import Logger

from decision_making.src.exceptions import NoActionsLeftForBPError
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionRecipeEvaluator
from typing import List

from decision_making.src.state.state import State

from ray.rllib.policy.sample_batch import SampleBatch


class LaneMergeRLPolicy(ActionRecipeEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate(self, state: State, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 actions_mask: List[bool], policy) -> np.array:
        """
        Given RL policy, current state and actions mask, return the actions cost by operating policy on the state.
        Filtered actions get the maximal cost 1.
        :param state: current state
        :param behavioral_state: semantic behavioral state, containing the semantic grid.
        :param action_recipes: actions list
        :param actions_mask: actions mask
        :param policy: RL policy
        :return: array of actions costs: the lower the better
        """
        encoded_state = encode_state_for_policy(state)
        logits, _, _, _ = policy.model({SampleBatch.CUR_OBS: np.array([encoded_state])}, [])
        actions_distribution = policy.dist_class(logits[0])
        actions_distribution[~np.array(actions_mask)] = 0  # set zero probability for filtered actions
        prob_sum = np.sum(actions_distribution)
        if prob_sum == 0:
            raise NoActionsLeftForBPError()
        costs = 1 - actions_distribution / prob_sum
        return costs
