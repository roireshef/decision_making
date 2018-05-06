from typing import List, Dict
import numpy as np
from logging import Logger

from decision_making.src.exceptions import TimeAlignmentPredictionHorizonTooLong
from decision_making.src.prediction.action_unaware_predictor import ActionUnawarePredictor
from decision_making.src.state.state import State, DynamicObject
from decision_making_sim.src.global_constants import TIME_ALIGNMENT_PREDICTOR_MAX_HORIZON
from prediction_research.src.utils.state_utils import get_object_from_state


class TimeAlignmentPredictor(ActionUnawarePredictor):
    """
    Performs physical prediction for the purpose of short time alignment between ego and dynamic objects.
    Logic should be re-considered if the time horizon gets too large.
    """

    def __init__(self, logger: Logger):
        super().__init__(logger=logger)

    def predict_state(self, state: State, prediction_timestamps: np.ndarray) -> List[State]:
        """
        Predicts the future states of the given state, for the specified timestamps
        :param state: the initial state to begin prediction from
        :param prediction_timestamps: np array of size 1 of timestamp in [sec] to predict states for. In ascending order.
        Global, not relative
        :return: a list of predicted states for the requested prediction_timestamps
        """

        # Predict to a single horizon
        assert len(prediction_timestamps) == 1

        # Verify that prediction horizon doesn't exceed the maximal horizon defined for this short-time predictor
        object_ids = [obj.obj_id for obj in state.dynamic_objects]
        self._validate_prediction_horizon(object_ids=object_ids, prediction_timestamps=prediction_timestamps,
                                          state=state)

    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray) \
            -> Dict[int, List[DynamicObject]]:
        """
        Predicts the future of the specified objects, for the specified timestamps
        :param state: the initial state to begin prediction from. Though predicting a single object, the full state
        provided to enable flexibility in prediction given state knowledge
        :param object_ids: a list of ids of the specific objects to predict
        :param prediction_timestamps: np array of size 1 of timestamp in [sec] to predict the object for. In ascending order.
        Global, not relative
        :return: a mapping between object id to the list of future dynamic objects of the matching object
        """

        # Predict to a single horizon
        assert len(prediction_timestamps) == 1

        # Verify that prediction horizon doesn't exceed the maximal horizon defined for this short-time predictor
        self._validate_prediction_horizon(object_ids=object_ids, prediction_timestamps=prediction_timestamps,
                                          state=state)

    def _validate_prediction_horizon(self, object_ids: List[int], prediction_timestamps: np.ndarray,
                                     state: State) -> None:
        """
        Raise error is longest horizon exceeds maximal horizon
        :param obj_ids: ids of objects to be predicted
        :param prediction_timestamps: np array of timestamps in [sec] to predict the object for. In ascending order.
        :param state:
        :return:
        """

        # Find object with oldest timestamp
        predicted_objects_timestamps = [get_object_from_state(state=state, target_obj_id=obj_id).timestamp_in_sec for
                                        obj_id in object_ids]
        predicted_objects_timestamps = np.r_[predicted_objects_timestamps, state.ego_state.timestamp_in_sec]
        oldest_timestamp = np.min(predicted_objects_timestamps)
        longest_horizon = prediction_timestamps[-1] - oldest_timestamp

        # Verify horizon length
        if longest_horizon > TIME_ALIGNMENT_PREDICTOR_MAX_HORIZON:
            oldest_object_id = object_ids[int(np.argmin(predicted_objects_timestamps))]
            raise TimeAlignmentPredictionHorizonTooLong(
                'Prediction horizon of object %d is %f. Exceeds maximal horizon of %f. Current ego timestamp: %d' % (
                    oldest_object_id, longest_horizon, TIME_ALIGNMENT_PREDICTOR_MAX_HORIZON, state.ego_state.timestamp))
