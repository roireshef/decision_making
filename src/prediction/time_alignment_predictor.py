from typing import List
import numpy as np
from logging import Logger

from decision_making.src.prediction.unaware_predictor import UnawarePredictor
from decision_making.src.state.state import State, DynamicObject


class TimeAlignmentPredictor(UnawarePredictor):
    """
    Performs physical prediction for the purpose of short time alignment between ego and dynamic objects.
    Logic should be re-considered if the time horizon gets too large.
    """
    def __init__(self, logger: Logger):
        super().__init__(logger=logger)

    def predict_state(self, state: State, prediction_timestamps: np.ndarray) -> List[State]:
        """
        Predictes the future states of the given state, for the specified timestamps
        :param state: the initial state to begin prediction from
        :param prediction_timestamps: np array of timestamps in [sec] to predict states for. In ascending order.
        Global, not relative
        :return: a list of predicted states for the requested prediction_timestamps
        """
        pass

    def predict_object(self, state: State, object_id: int, prediction_timestamps: np.ndarray) -> List[DynamicObject]:
        """
        Predictes the future of the specified object, for the specified timestamps
        :param state: the initial state to begin prediction from. Though predicting a single object, the full state
        provided to enable flexibility in prediction given state knowledge
        :param object_id: the specific object to predict
        :param prediction_timestamps: np array of timestamps in [sec] to predict the object for. In ascending order.
        Global, not relative
        :return: a list of future dynamic objects of the specified object
        """
        pass

