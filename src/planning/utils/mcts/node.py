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

    # TODO: define model class
    def perform(self, model) -> StateTreeNode:
        """
        Performs the action and samples terminal state from a model. If the terminal state is not in the children
        already, add it.
        :param model:
        :return: the terminal state's StateTreeNode
        """
        terminal_state = model.sample(self.parent.state, self.action)

        if terminal_state not in self.children.keys():
            self.children[terminal_state] = StateTreeNode(self, terminal_state)

        return self.children[terminal_state]

    def __str__(self):
        return "Action: %s" % self.__dict__


class StateTreeNode(TreeNode):
    """
    A node holding a state in the tree.
    """
    # TODO: define state class, action class
    def __init__(self, parent: TreeNode, state, actions: List):
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
        return "State: %s" % self.__dict__


