from abc import ABCMeta, abstractmethod


class TrajectoryPlanner(metaclass=ABCMeta):
    @abstractmethod
    def plan(self, state, reference_route, goal, cost_params):
        """
        plans a trajectory according to the specifications in the arguments
        :param state: environment & ego state object
        :param reference_route: a route given by the previous planning layer, often the center of lane
        :param goal: the goal state to plan toward
        :param cost_params: a dictionary of parameters that specify how to build the planning's cost function
        :return: a numpy tensor of the following dimensions [0-trajectories, 1-trajectory interm.
            states, 2-interm state of the form [x, y, yaw, velocity] in vehicle's coordinate frame]
        """
        pass
