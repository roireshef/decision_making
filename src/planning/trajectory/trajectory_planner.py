import numpy as np
from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import Tuple
from decision_making.src.exceptions import raises, CartesianLimitsViolated, FrenetLimitsViolated
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import CartesianTrajectories, \
    CartesianExtendedState
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.planning.behavioral.state import State


class TrajectoryPlanner(metaclass=ABCMeta):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor):
        self._logger = logger
        self._predictor = predictor

    @property
    def predictor(self):
        return self._predictor

    @abstractmethod
    @raises(CartesianLimitsViolated, FrenetLimitsViolated)
    def plan(self, state: State, reference_route: FrenetSerret2DFrame, goal: CartesianExtendedState,
             T_target_horizon: float, T_trajectory_end_horizon: float, cost_params: TrajectoryCostParams) -> \
            Tuple[SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """
        Plans a trajectory according to the specifications in the arguments
        :param state: environment & ego state object
        :param reference_route: the frenet frame of the reference route (often the center of lane).
        :param goal: A 1D numpy array of the desired ego-state to plan towards, represented in current
        global-coordinate-frame (see EGO_* in planning.utils.types.py for the fields)
        :param T_target_horizon: RELATIVE [sec]. defines the planning horizon in [sec] for reaching the target/goal 
        (terminal boundary condition). Enables the target state and time to be determined in the behavioral planner, 
        so that any re-planning iteration is consistent in the TP.
        :param T_trajectory_end_horizon: RELATIVE [sec]. The time at which the final trajectory will end, including 
        padding (extension beyond the target - the terminal boundary condition). 
        A consequence of a minimal time required by control.
        :param cost_params: Data object with parameters that specify how to build the planning's cost function
        :return: a tuple of: (samplable representation of the chosen trajectory, tensor of trajectory alternatives,
         trajectories costs correspond to previous output)
        """
        pass
