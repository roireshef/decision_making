from abc import ABCMeta, abstractmethod
from typing import List, Dict
import numpy as np
from logging import Logger

from decision_making.src.state.state import State, DynamicObject


class ActionUnawarePredictor(metaclass=ABCMeta):
    """
    Base class for simple / naive prediction of states
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
