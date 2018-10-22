import numpy as np
from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import Tuple

from decision_making.src.exceptions import raises, NoValidTrajectoriesFound, CouldNotGenerateTrajectories
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import CartesianPath2D, CartesianTrajectories, \
    CartesianExtendedState
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State


class TrajectoryPlanner(metaclass=ABCMeta):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor):
        self._logger = logger
        self._predictor = predictor

    @property
    def predictor(self):
        return self._predictor

    @abstractmethod
    @raises(NoValidTrajectoriesFound, CouldNotGenerateTrajectories)
    def plan(self, state: State, reference_route: FrenetSerret2DFrame, goal: CartesianExtendedState, time_horizon: float,
             cost_params: TrajectoryCostParams) -> \
            Tuple[SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """
        Plans a trajectory according to the specifications in the arguments
        :param time_horizon: defines the planning horizon in [sec] for reaching the goal. Enables the target
            state and time to be determined in the behavioral planner, so that any re-planning iteration is consistent
            in the TP.
        :param state: environment & ego state object
        :param reference_route: the frenet frame of the reference route (often the center of lane).
        :param goal: A 1D numpy array of the desired ego-state to plan towards, represented in current
        global-coordinate-frame (see EGO_* in planning.utils.types.py for the fields)
        :param cost_params: Data object with parameters that specify how to build the planning's cost function
        :return: a tuple of: (samplable representation of the chosen trajectory, tensor of trajectory alternatives,
         trajectories costs correspond to previous output)
        """
        pass
