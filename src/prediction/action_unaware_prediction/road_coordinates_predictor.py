from typing import List, Dict
import numpy as np
from logging import Logger

from decision_making.src.prediction.action_unaware_prediction.action_unaware_predictor import ActionUnawarePredictor
from decision_making.src.state.state import State, DynamicObject


class RoadCoordinatesPredictor(ActionUnawarePredictor):
    """
    Performs simple / naive prediction in road coordinates (road following prediction, constant velocity)
    and returns objects with calculated and cached road coordinates. This is in order to save coordinate conversion time
    for the predictor's clients.
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

    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray) \
            -> Dict[int, List[DynamicObject]]:
        """
        Predicte the future of the specified objects, for the specified timestamps
        :param state: the initial state to begin prediction from. Though predicting a single object, the full state
        provided to enable flexibility in prediction given state knowledge
        :param object_ids: a list of ids of the specific objects to predict
        :param prediction_timestamps: np array of timestamps in [sec] to predict the object for. In ascending order.
        Global, not relative
        :return: a mapping between object id to the list of future dynamic objects of the matching object
        """
        pass

