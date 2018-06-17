import random
import numpy as np


def rand_max(iterable, key=None):
    """
    A max function that tie breaks randomly instead of first-wins as in
    built-in max().
    :param iterable: The container to take the max from
    :param key: A function to compute tha max from. E.g.:
      >>> rand_max([-2, 1], key=lambda x:x**2
      -2
      If key is None the identity is used.
    :return: The entry of the iterable which has the maximum value. Tie
    breaks are random.
    """
    if key is None:
        key = lambda x: x

    max_v = -np.inf
    max_l = []

    for item, value in zip(iterable, [key(i) for i in iterable]):
        if value == max_v:
            max_l.append(item)
        elif value > max_v:
            max_l = [item]
            max_v = value

    return random.choice(max_l)


def breadth_first_search(root, fnc=None):
    """
    A breadth first search (BFS) over the subtree starting from root. A
    function can be run on all visited nodes. It gets the current visited
    node and a data object, which it can update and should return it. This
    data is returned by the function but never altered from the BFS itself.
    :param root: The node to start the BFS from
    :param fnc: The function to run on the nodes
    :return: A data object, which can be altered from fnc.
    """
    data = None
    queue = [root]
    while queue:
        node = queue.pop(0)
        data = fnc(node, data)
        for child in node.children.values():
            queue.append(child)
    return data


def depth_first_search(root, fnc=None):
    """
    A depth first search (DFS) over the subtree starting from root. A
    function can be run on all visited nodes. It gets the current visited
    node and a data object, which it can update and should return it. This
    data is returned by the function but never altered from the DFS itself.
    :param root: The node to start the DFS from
    :param fnc: The function to run on the nodes
    :return: A data object, which can be altered from fnc.
    """
    data = None
    stack = [root]
    while stack:
        node = stack.pop()
        data = fnc(node, data)
        for child in node.children.values():
            stack.append(child)
    return data


def get_actions_and_states(node):
    """
    Returns a tuple of two lists containing the action and the state nodes
    under the given node.
    :param node:
    :return: A tuple of two lists
    """
    return depth_first_search(node, _get_actions_and_states)


def _get_actions_and_states(node, data):
    if data is None:
        data = ([], [])

    action_nodes, state_nodes = data

    if isinstance(node, ActionNode):
        action_nodes.append(node)
    elif isinstance(node, StateNode):
        state_nodes.append(node)

    return action_nodes, state_nodes