from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import Tuple

import numpy as np

from decision_making.src.exceptions import raises, NoValidTrajectoriesFound
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.types import CartesianPath2D, CartesianExtendedTrajectory, CartesianTrajectories, \
    CartesianExtendedState
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State


class SamplableTrajectory(metaclass=ABCMeta):
    def __init__(self, timestamp: float, max_sample_time: float):
        """
        Abstract class that holds all the statistics to sample points on a specific planned trajectory
        :param timestamp: [sec] global timestamp *in seconds* to use as a reference
                (other timestamps will be given relative to it)
        :param max_sample_time: [sec] global timestamp which is the end of the planning-horizon to prevent extrapolation
        in sampling from trajectory plan
        """
        self.max_sample_time = max_sample_time
        self.timestamp = timestamp

    @abstractmethod
    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        """
        This function takes an array of time stamps and returns an array of points <x, y, theta, v, acceleration,
        curvature> along the trajectory
        :param time_points: 1D numpy array of time stamps *in seconds* (global self.timestamp)
        :return: 2D numpy array with every row having the format of <x, y, yaw, velocity, a, k>
        """
        pass


class TrajectoryPlanner(metaclass=ABCMeta):
    def __init__(self, logger: Logger, predictor: Predictor):
        self._logger = logger
        self._predictor = predictor

    @property
    def predictor(self):
        return self._predictor

    @abstractmethod
    @raises(NoValidTrajectoriesFound)
    def plan(self, state: State, reference_route: CartesianPath2D, goal: CartesianExtendedState, lon_plan_horizon: float,
             cost_params: TrajectoryCostParams) -> Tuple[SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """
        Plans a trajectory according to the specifications in the arguments
        :param lon_plan_horizon: defines the longitudinal planning horizon in [sec] for reaching the goal. Enables the target
            state and time to be determined in the behavioral planner, so that any re-planning iteration is consistent
            in the TP.
        :param state: environment & ego state object
        :param reference_route: a reference route (often the center of lane).
        :param goal: A 1D numpy array of the desired ego-state to plan towards, represented in current
        global-coordinate-frame (see EGO_* in planning.utils.types.py for the fields)
        :param cost_params: Data object with parameters that specify how to build the planning's cost function
        :return: a tuple of: (samplable represantation of the chosen trajectory, tensor of trajectory alternatives,
         trajectories costs correspond to previous output)
        """
        pass
