from __future__ import division

from abc import abstractclassmethod

from decision_making.src.planning.utils.mcts.node import StateTreeNode, ActionTreeNode, TreeNode


class TreeBackup:
    @abstractclassmethod
    def __call__(self, node: TreeNode):
        pass


class BellmanBackup(TreeBackup):
    def __init__(self, gamma: float):
        """
        A dynamical programming update which resembles the Bellman equation
        of value iteration. See Feldman and Domshlak (2014) for reference.
        :param gamma: reward-decay coefficient
        """
        self.gamma = gamma

    def __call__(self, node: TreeNode):
        """
        :param node: The node to start the backups from
        """
        while node is not None:
            node.n += 1
            if isinstance(node, StateTreeNode):
                node.q = max([x.q for x in node.children.values()])
            elif isinstance(node, ActionTreeNode):
                n = sum([x.n for x in node.children.values()])
                node.q = sum([(self.gamma * x.q + x.reward) * x.n
                              for x in node.children.values()]) / n
            node = node.parent


# TODO: unify those
class MonteCarloBackup(TreeBackup):
    def __call__(self, node: TreeNode):
        monte_carlo(node)


def monte_carlo(node):
    """
    A monte carlo update as in classical UCT.

    See feldman amd Domshlak (2014) for reference.
    :param node: The node to start the backup from
    """
    r = node.reward
    while node is not None:
        node.n += 1
        node.q = ((node.n - 1)/node.n) * node.q + 1/node.n * r
        node = node.parent
