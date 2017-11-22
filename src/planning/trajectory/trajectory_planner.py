from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from decision_making.src.exceptions import raises, NoValidTrajectoriesFound
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State
from logging import Logger


# TODO: Document, fill
class SamplableTrajectory(metaclass=ABCMeta):
    """
    Abstract class that holds all the statistics to sample points on a specific planned trajectory
    """
    @abstractmethod
    def sample(self, time_points: np.ndarray) -> np.ndarray:
        """
        This function takes an array of time stamps and returns an array of points <x, y, theta, v> along the trajectory
        :param time_points: 1D numpy array of time stamps (relative to the time the trajectory was planned)
        :return: 2D numpy array with every row having the format of <x, y, yaw, velocity>
        """
        pass


class TrajectoryPlanner(metaclass=ABCMeta):
    def __init__(self, logger: Logger, predictor: Predictor):
        self._logger = logger
        self._predictor = predictor

    @abstractmethod
    @raises(NoValidTrajectoriesFound)
    def plan(self, state: State, reference_route: np.ndarray, goal: np.ndarray, time: float,
             cost_params: TrajectoryCostParams) -> Tuple[np.ndarray, float, SamplableTrajectory,
                                                         TrajectoryVisualizationMsg]:
        """
        Plans a trajectory according to the specifications in the arguments
        :param time: the time-window to plan for (time to get from initial state to goal state)
        :param state: environment & ego state object
        :param reference_route: a reference route (often the center of lane). A numpy array of the shape [-1, 2] where
        each row is a point (x, y) in world coordinates.
        :param goal: A 1D numpy array of the desired ego-state to plan towards, represented in current
        global-coordinate-frame (see EGO_* in planning.utils.columns.py for the fields)
        :param cost_params: Data object with parameters that specify how to build the planning's cost function
        :return: a tuple of: (numpy array: trajectory - each row is [x, y, yaw, velocity], trajectory cost,
        samplable represantation of the chosen trajecdebug results)
        """
        pass
