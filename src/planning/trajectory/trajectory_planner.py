from abc import ABCMeta, abstractmethod

import numpy as np

from src.planning.trajectory.cost_function import CostParams


class TrajectoryPlanner(metaclass=ABCMeta):
    @abstractmethod
    # TODO: link state to its type once interface is merged
    def plan(self, state, reference_route: np.ndarray, goal: np.ndarray, cost_params: CostParams):
        """
        plans a trajectory according to the specifications in the arguments
        :param state: environment & ego state object
        :param reference_route: a reference route (often the center of lane). A numpy array of the shape [-1, 2]
        :param goal: A numpy array of the desired ego-state to plan towards, from utils.columns (ego coord-frame)
        :param cost_params: a dictionary of parameters that specify how to build the planning's cost function
        :return: a numpy tensor of the following dimensions [0-trajectories, 1-trajectory interm.
            states, 2-interm state of the form [x, y, yaw, velocity] in vehicle's coordinate frame]
        """
        pass
