from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.state.state import State
from logging import Logger


class TrajectoryPlanner(metaclass=ABCMeta):
    def __init__(self, logger: Logger):
        self._logger = logger

    @abstractmethod
    def plan(self, state: State, reference_route: np.ndarray, goal: np.ndarray, time: float,
             cost_params: TrajectoryCostParams) -> Tuple[np.ndarray, float, TrajectoryVisualizationMsg]:
        """
        Plans a trajectory according to the specifications in the arguments
        :param time: the time-window to plan for (time to get from initial state to goal state)
        :param state: environment & ego state object
        :param reference_route: a reference route (often the center of lane). A numpy array of the shape [-1, 2]
        :param goal: A 1D numpy array of the desired ego-state to plan towards, represented in current
        ego-coordinate-frame (see EGO_* in planning.utils.columns.py for the fields)
        :param cost_params: a dictionary of parameters that specify how to build the planning's cost function
        :return: a tuple of: (numpy array: trajectory - each row is [x, y, yaw, velocity], trajectory cost,
        debug results)
        """
        pass
