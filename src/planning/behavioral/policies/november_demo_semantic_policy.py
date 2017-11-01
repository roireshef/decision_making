import numpy as np
from typing import List

from decision_making.src.planning.behavioral.semantic_actions_policy import SemanticActionsPolicy, SemanticAction, \
    SemanticActionSpec, SEMANTIC_CELL_LANE
from decision_making.src.state.state import State

SEMANTIC_GRID_FRONT, SEMANTIC_GRID_ASIDE, SEMANTIC_GRID_BEHIND = 1, 0, -1
GRID_MID = 10

# The margin that we take from the front/read of the vehicle to define the front/rear partitions
SEMANTIC_OCCUPANCY_GRID_PARTITIONS_MARGIN_FROM_EGO = 1


MIN_OVERTAKE_VEL = 2  # [m/s]


class NovDemoPolicy(SemanticActionsPolicy):

    def _eval_actions(self, state: State, semantic_actions: List[SemanticAction],
                      actions_spec: List[SemanticActionSpec]) -> np.ndarray:
        """
        Evaluate the generated actions using the full state.
        Gets a list of actions to evaluate so and returns a vector representing their costs.
        A set of actions is provided, enabling assessing them dependently.
        Note: the semantic actions were generated using the behavioral state which isn't necessarily captures
         all relevant details in the scene. Therefore the evaluation is done using the full state.
        :param state: world state
        :param semantic_actions: semantic actions list
        :param actions_spec: specifications of semantic actions
        :return: numpy array of costs of semantic actions
        """

        straight_lane_ind = self._get_action_ind_by_lane(semantic_actions, 0)
        left_lane_ind = self._get_action_ind_by_lane(semantic_actions, 1)
        right_lane_ind = self._get_action_ind_by_lane(semantic_actions, -1)

        right_is_fast = right_lane_ind is not None and \
                        self._max_velocity - actions_spec[right_lane_ind].v < MIN_OVERTAKE_VEL
        straight_is_fast = straight_lane_ind is not None and \
                           self._max_velocity - actions_spec[straight_lane_ind].v < MIN_OVERTAKE_VEL
        left_is_faster = left_lane_ind is not None and \
                         (straight_lane_ind is None or
                            actions_spec[left_lane_ind].v - actions_spec[straight_lane_ind].v >= MIN_OVERTAKE_VEL)

        costs = np.zeros(len(semantic_actions))
        # move right if both straight and right lanes are not slow
        if right_is_fast and (straight_is_fast or straight_lane_ind is None):
            costs[right_lane_ind] = 1.
            return costs
        # move left if straight is slow and the left is faster than straight
        if not straight_is_fast and (left_is_faster or straight_lane_ind is None):
            costs[left_lane_ind] = 1.
            return costs
        costs[straight_lane_ind] = 1.
        return costs

    @staticmethod
    def _get_action_ind_by_lane(semantic_actions: List[SemanticAction], cell_lane: int):
        action_ind = [i for i, action in enumerate(semantic_actions) if action.cell[SEMANTIC_CELL_LANE] == cell_lane]
        if len(action_ind) > 0:
            return action_ind[0]
        else:
            return None
