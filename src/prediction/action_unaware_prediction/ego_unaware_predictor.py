from abc import ABCMeta, abstractmethod
from typing import List, Dict
import numpy as np
from logging import Logger

from decision_making.src.planning.behavioral.state import State, DynamicObject


class EgoUnawarePredictor(metaclass=ABCMeta):
    """
    Base class for prediction which is unaware to ego's actions. 
    """
    def __init__(self, logger: Logger):
        self._logger = logger

    @abstractmethod
    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray) \
            -> Dict[int, List[DynamicObject]]:
        """
        Predicts the future of the specified objects, for the specified timestamps
        :param state: the initial state to begin prediction from. Though predicting a single object, the full state
        provided to enable flexibility in prediction given state knowledge
        :param object_ids: a list of ids of the specific objects to predict
        :param prediction_timestamps: np array of timestamps in [sec] to predict the object for. In ascending order.
        Global, not relative
        :return: a mapping between object id to the list of future dynamic objects of the matching object
        """
        pass

    @abstractmethod
    def predict_state(self, state: State, prediction_timestamps: np.ndarray) -> List[State]:
        """
        Predicts the future states of the given state, for the specified timestamps
        :param state: the initial state to begin prediction from
        :param prediction_timestamps: np array of timestamps in [sec] to predict the object for. In ascending order.
        Global, not relative
        :return: a list of predicted states for the requested prediction_timestamps
        """