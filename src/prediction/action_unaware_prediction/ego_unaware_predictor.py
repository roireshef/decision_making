from abc import ABCMeta, abstractmethod
from typing import List, Dict
import numpy as np
from logging import Logger

from decision_making.src.planning.types import GlobalTimeStampInSec, MinGlobalTimeStampInSec
from decision_making.src.state.state import State, DynamicObject


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

    def align_objects_to_most_recent_timestamp(self, state: State,
                                               current_timestamp: GlobalTimeStampInSec = MinGlobalTimeStampInSec) -> State:
        """
        Returns state with all objects aligned to the most recent timestamp.
        Most recent timestamp is taken as the max between the current_timestamp, and the most recent
        timestamp of all objects in the scene.
        :param current_timestamp: current timestamp in global time in [sec]
        :param state: state containing objects with different timestamps
        :return: new state with all objects aligned
        """
        ego_timestamp_in_sec = state.ego_state.timestamp_in_sec
        objects_timestamp_in_sec = [state.dynamic_objects[x].timestamp_in_sec for x in
                                    range(len(state.dynamic_objects))]
        objects_timestamp_in_sec.append(ego_timestamp_in_sec)
        most_recent_timestamp = np.max(objects_timestamp_in_sec)
        most_recent_timestamp = np.maximum(most_recent_timestamp, current_timestamp)
        self._logger.debug("Prediction of ego by: %s sec", most_recent_timestamp - ego_timestamp_in_sec)
        return self.predict_state(state=state, prediction_timestamps=np.array([most_recent_timestamp]))[0]