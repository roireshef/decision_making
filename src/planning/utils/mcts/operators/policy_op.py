from __future__ import division
import numpy as np

from decision_making.src.planning.utils.mcts.node import ActionTreeNode


class UCB1:
    def __init__(self, c):
        """
        The typical bandit upper confidence bounds algorithm.
        :param c: the exploration-component coefficient
        """
        self.c = c

    def __call__(self, action_node: ActionTreeNode):
        # assert that no nan values are returned
        # for action_node.n = 0
        if self.c == 0:
            return action_node.q

        return action_node.q + self.c * np.sqrt(2 * np.log(action_node.parent.n) / action_node.n)
