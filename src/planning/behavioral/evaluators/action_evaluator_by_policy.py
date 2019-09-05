import numpy as np
from logging import Logger

from decision_making.src.exceptions import NoActionsLeftForBPError
from typing import List

from decision_making.src.state.lane_merge_state import LaneMergeState
from ray.rllib.policy.sample_batch import SampleBatch


class LaneMergeRLPolicy:

    @staticmethod
    def evaluate(state: LaneMergeState) -> np.array:
        """
        Given RL policy, current state and actions mask, return the actions cost by operating policy on the state.
        Filtered actions get the maximal cost 1.
        :param state: LaneMergeState
        :return: array of actions costs: the lower the better
        """
        encoded_state = encode_state_for_policy(state)
        logits, _, _, _ = RL_policy.model({SampleBatch.CUR_OBS: np.array([encoded_state])}, [])
        actions_distribution = RL_policy.dist_class(logits[0])
        # actions_distribution[~np.array(actions_mask)] = 0  # set zero probability for filtered actions
        prob_sum = np.sum(actions_distribution)
        if prob_sum == 0:
            raise NoActionsLeftForBPError()
        costs = 1 - actions_distribution / prob_sum
        return costs
