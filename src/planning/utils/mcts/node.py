from typing import List


class TreeNode:
    def __init__(self, parent):
        self.parent = parent
        self.children = {}
        self.q = 0
        self.n = 0
        self.depth = parent.depth + 1 if parent else 0


class ActionTreeNode(TreeNode):
    """
    A node holding an action in the tree.
    """
    def __init__(self, parent, action):
        super(ActionTreeNode, self).__init__(parent)
        self.action = action
        self.n = 0

    def sample_state(self, real_world=False):
        """
        Samples a state from this action and adds it to the tree if the
        state never occurred before.

        :param real_world: If planning in belief states are used, this can
        be set to True if a real world action is taken. The belief is than
        used from the real world action instead from the belief state actions.
        :return: The state node, which was sampled.
        """
        if real_world:
            state = self.parent.state.real_world_perform(self.action)
        else:
            state = self.parent.state.perform(self.action)

        if state not in self.children:
            self.children[state] = StateTreeNode(self, state)

        if real_world:
            self.children[state].state.belief = state.belief

        return self.children[state]

    def __str__(self):
        return "Action: {}".format(self.action)


class StateTreeNode(TreeNode):
    """
    A node holding a state in the tree.
    """
    def __init__(self, parent: TreeNode, state: object, actions: List):
        super(StateTreeNode, self).__init__(parent)
        self.state = state
        self.reward = 0
        self.children = {action: ActionTreeNode(self, action) for action in actions}

    @property
    def untried_actions(self):
        """
        All actions which have never be performed
        :return: A list of the untried actions.
        """
        return [a for a in self.children if self.children[a].n == 0]

    def __str__(self):
        return "State: {}".format(self.state)


