from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from decision_making.src.exceptions import raises, NoValidTrajectoriesFound
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.types import CartesianPath, CartesianTrajectory
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State
from logging import Logger


class TrajectoryPlanner(metaclass=ABCMeta):
    def __init__(self, logger: Logger, predictor: Predictor):
        self._logger = logger
        self._predictor = predictor

    @abstractmethod
    @raises(NoValidTrajectoriesFound)
    def plan(self, state: State, reference_route: CartesianPath, goal: np.ndarray, goal_time: float,
             cost_params: TrajectoryCostParams) -> Tuple[CartesianTrajectory, float, TrajectoryVisualizationMsg]:
        """
        Plans a trajectory according to the specifications in the arguments
        :param goal_time: defines the global time in [sec] of the goal. Enables the target state and time
            to be determined in the behavioral planner, so that any re-planning iteration is consistent in the TP.
        :param state: environment & ego state object
        :param reference_route: a reference route (often the center of lane). A numpy array of the shape [, 2] where
        each row is a point (x, y) in world coordinates.
        :param goal: A 1D numpy array of the desired ego-state to plan towards, represented in current
        global-coordinate-frame (see EGO_* in planning.utils.types.py for the fields)
        :param cost_params: Data object with parameters that specify how to build the planning's cost function
        :return: a tuple of: (numpy array: trajectory - each row is [x, y, yaw, velocity], trajectory cost,
        debug results)
        """
        pass
