from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List, Dict
import numpy as np

from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.state.state import State, DynamicObject


class ActionAwarePredictor(metaclass=ABCMeta):
    """
    Base class for advanced prediction logic, which is able to account for reactions to ego's action
    """

    def __init__(self, logger: Logger):
        self._logger = logger

    @abstractmethod
    def predict_state(self, state: State, prediction_timestamps: np.ndarray, action_trajectory: SamplableTrajectory)\
            -> (List[State]):
        """
        Predicts the future states of the given state, for the specified timestamps
        :param state: the initial state to begin prediction from
        :param prediction_timestamps: np array of timestamps in [sec] to predict states for. In ascending order.
        Global, not relative
        :param action_trajectory: the ego's planned action trajectory
        :return: a list of non markov predicted states for the requested prediction_timestamp, and a full state for the
        terminal predicted state
        """
        pass

    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray,
                        action_trajectory: SamplableTrajectory) -> Dict[int, List[DynamicObject]]:
        """
        Predicte the future of the specified objects, for the specified timestamps
        :param state: the initial state to begin prediction from. Though predicting a single object, the full state
        provided to enable flexibility in prediction given state knowledge
        :param object_ids: a list of ids of the specific objects to predict
        :param prediction_timestamps: np array of timestamps in [sec] to predict the object for. In ascending order.
        Global, not relative
        :param action_trajectory: the ego's planned action trajectory
        :return: a mapping between object id to the list of future dynamic objects of the matching object
        """
        pass
