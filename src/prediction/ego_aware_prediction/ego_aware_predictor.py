import numpy as np
from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List, Dict

from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import FrenetTrajectories2D, FrenetStates2D, FrenetTrajectories1D, \
    FrenetStates1D
from decision_making.src.state.state import State, DynamicObject


class EgoAwarePredictor(metaclass=ABCMeta):
    """
    Base class for advanced prediction logic, which is able to account for reactions to ego's action
    """

    def __init__(self, logger: Logger):
        self._logger = logger

    @abstractmethod
    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray,
                        action_trajectory: SamplableTrajectory) -> Dict[int, List[DynamicObject]]:
        """
        Predicts the future of the specified objects, for the specified timestamps
        :param state: the initial state to begin prediction from. Though predicting a single object, the full state
        provided to enable flexibility in prediction given state knowledge
        :param object_ids: a list of ids of the specific objects to predict
        :param prediction_timestamps: np array of timestamps in [sec] to predict the object for. In ascending order.
        Global, not relative
        :param action_trajectory: the ego's planned action trajectory
        :return: a mapping between object id to the list of future dynamic objects of the matching object
        """
        pass

    @abstractmethod
    def predict_1d_frenet_states(self, objects_fstates: FrenetStates1D, horizons: np.ndarray) -> FrenetTrajectories1D:
        """
        Constant velocity prediction for all timestamps and objects in a matrix computation
        :param objects_fstates: numpy 2D array [Nx3] where N is the number of objects, each row is an FSTATE
        :param horizons: numpy 1D array [T] with T horizons (relative time for prediction into the future)
        :return: numpy 3D array [NxTx3]
        """
        pass

    @abstractmethod
    def predict_2d_frenet_states(self, objects_fstates: FrenetStates2D, horizons: np.ndarray) -> FrenetTrajectories2D:
        """
        Constant velocity prediction for all timestamps and objects in a matrix computation
        :param objects_fstates: numpy 2D array [Nx6] where N is the number of objects, each row is an FSTATE
        :param horizons: numpy 1D array [T] with T horizons (relative time for prediction into the future)
        :return: numpy 3D array [NxTx6]
        """
        pass
