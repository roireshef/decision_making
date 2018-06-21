from __future__ import print_function

import random

from decision_making.src.planning.behavioral.action_space.action_space import ActionSpace
from decision_making.src.planning.utils.mcts.tbd import utils


class MCTS:
    def __init__(self, policy_operator, value_operator, backup_operator, action_space: ActionSpace):
        """
        The central MCTS class, which performs the tree search. It gets a
        tree policy, a default policy, and a backup strategy.
        See e.g. Browne et al. (2012) for a survey on monte carlo tree search
        :param policy_operator: operates on ActionTreeNode to compute the values for comparing actions during traversal
        :param value_operator: operates on
        :param backup_operator:
        """
        self.policy_operator = policy_operator
        self.value_operator = value_operator
        self.backup_operator = backup_operator
        self.action_space = action_space

    def __call__(self, root, n=1500):
        """
        Run the monte carlo tree search.

        :param root: The StateNode
        :param n: The number of roll-outs to be performed
        :return:
        """
        if root.parent is not None:
            raise ValueError("Root's parent must be None.")

        for _ in range(n):
            node = MCTS._get_next_node(root, self.policy_operator)
            node.reward = self.value_operator(node)
            self.backup_operator(node)

        return utils.rand_max(root.children.values(), key=lambda x: x.q).action

    @staticmethod
    def _expand(state_node):
        action = random.choice(state_node.untried_actions)
        return state_node.children[action].sample_state()

    @staticmethod
    def _best_child(state_node, tree_policy):
        best_action_node = utils.rand_max(state_node.children.values(), key=tree_policy)
        return best_action_node.sample_state()

    @staticmethod
    def _get_next_node(state_node, tree_policy):
        while not state_node.state.is_terminal():
            if state_node.untried_actions:
                return MCTS._expand(state_node)
            else:
                state_node = MCTS._best_child(state_node, tree_policy)
        return state_node
