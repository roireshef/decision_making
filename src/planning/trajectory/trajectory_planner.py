from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.state.state import State


class TrajectoryPlanner(metaclass=ABCMeta):
    # TODO: object type-hint should be changed to DDSMessage type once commited
    @abstractmethod
    def plan(self, state: State, reference_route: np.ndarray, goal: np.ndarray,
             cost_params: TrajectoryCostParams) -> Tuple[np.ndarray, float, object]:
        """
        Plans a trajectory according to the specifications in the arguments
        :param state: environment & ego state object
        :param reference_route: a reference route (often the center of lane). A numpy array of the shape [-1, 2]
        :param goal: A numpy array of the desired ego-state to plan towards, from utils.columns (ego coord-frame)
        :param cost_params: a dictionary of parameters that specify how to build the planning's cost function
        :return: a tuple of: (numpy array: trajectory - each row is [x, y, yaw, velocity], trajectory cost,
        debug results dictionary)
        """
        pass
