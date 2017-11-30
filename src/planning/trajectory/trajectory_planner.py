from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from decision_making.src.exceptions import raises, NoValidTrajectoriesFound
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.types import CartesianPath, CartesianTrajectory, CartesianState
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State
from logging import Logger


# TODO: fill the units in timestamps
class SamplableTrajectory(metaclass=ABCMeta):
    def __init__(self, timestamp: float):
        """
        Abstract class that holds all the statistics to sample points on a specific planned trajectory
        :param timestamp: global timestamp to use as a reference (other timestamps will be given relative to it)
        """
        self.timestamp = timestamp

    @abstractmethod
    def sample(self, time_points: np.ndarray) -> np.ndarray:
        """
        This function takes an array of time stamps and returns an array of points <x, y, theta, v> along the trajectory
        :param time_points: 1D numpy array of time stamps (relative to self.timestamp)
        :return: 2D numpy array with every row having the format of <x, y, yaw, velocity>
        """
        pass


class TrajectoryPlanner(metaclass=ABCMeta):
    def __init__(self, logger: Logger, predictor: Predictor):
        self._logger = logger
        self._predictor = predictor

    @abstractmethod
    @raises(NoValidTrajectoriesFound)
    def plan(self, state: State, reference_route: CartesianPath, goal: CartesianState, goal_time: float,
             cost_params: TrajectoryCostParams) -> Tuple[SamplableTrajectory, float, TrajectoryVisualizationMsg]:
        """
        Plans a trajectory according to the specifications in the arguments
        :param goal_time: defines the global time in [sec] of the goal. Enables the target state and time
            to be determined in the behavioral planner, so that any re-planning iteration is consistent in the TP.
        :param state: environment & ego state object
        :param reference_route: a reference route (often the center of lane).
        :param goal: A 1D numpy array of the desired ego-state to plan towards, represented in current
        global-coordinate-frame (see EGO_* in planning.utils.types.py for the fields)
        :param cost_params: Data object with parameters that specify how to build the planning's cost function
        :return: a tuple of: (samplable represantation of the chosen trajectory, trajectory cost, debug results)
        """
        pass
