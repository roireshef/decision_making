from logging import Logger
from typing import List, Dict

import numpy as np

from decision_making.src.global_constants import DEFAULT_CURVATURE
from decision_making.src.prediction.action_unaware_prediction.ego_unaware_predictor import EgoUnawarePredictor
from decision_making.src.prediction.utils.prediction_utils import PredictionUtils
from decision_making.src.state.state import State, DynamicObject


class PhysicalTimeAlignmentPredictor(EgoUnawarePredictor):
    """
    Performs physical prediction (constant velocity in cartesian frame) for the purpose of short time alignment between ego and
    dynamic objects.
    Logic should be re-considered if the time horizon gets too large.
    """

    def __init__(self, logger: Logger):
        super().__init__(logger)

    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray) \
            -> Dict[int, List[DynamicObject]]:
        """
        Performs physical prediction (constant velocity in cartesian frame) for the purpose of short time alignment
        between ego and dynamic objects
        :param state: the initial state to begin prediction from.
        :param object_ids: a list of ids of the specific objects to predict
        :param prediction_timestamps: np array of size 1 of timestamp in [sec] to predict states for. In ascending order
        Global, not relative
        :return: a mapping between object id to the list of future dynamic objects of the matching object
       """
        # Predict to a single horizon
        assert len(prediction_timestamps) == 1

        dynamic_objects = State.get_objects_from_state(state=state, target_obj_ids=object_ids)

        predicted_dynamic_objects: Dict[int, List[DynamicObject]] = dict()

        for dynamic_object in dynamic_objects:
            predicted_dynamic_object_states = self._predict_object(dynamic_object=dynamic_object,
                                                                   prediction_timestamp=prediction_timestamps[0])
            predicted_dynamic_objects[dynamic_object.obj_id] = predicted_dynamic_object_states

        return predicted_dynamic_objects

    def predict_state(self, state: State, prediction_timestamps: np.ndarray) -> List[State]:
        """
        Predicts the future states of the given state, for the specified timestamps
        :param state: the initial state to begin prediction from
        :param prediction_timestamps: np array of size 1 of timestamp in [sec] to predict states for. In ascending order
        Global, not relative
        :return: a list of predicted states for the requested prediction_timestamps
        """

        # Predict to a single horizon
        assert len(prediction_timestamps) == 1

        # Simple object-wise prediction
        object_ids = [obj.obj_id for obj in state.dynamic_objects]

        predicted_dynamic_objects_dict = self.predict_objects(state=state, object_ids=object_ids,
                                                              prediction_timestamps=prediction_timestamps)

        predicted_dynamic_objects = [future_object_states[0] for future_object_states in
                                     predicted_dynamic_objects_dict.values()]

        predicted_ego_state = self._predict_object(dynamic_object=state.ego_state,
                                                   prediction_timestamp=prediction_timestamps[0])[0]

        predicted_state = state.clone_with(dynamic_objects=predicted_dynamic_objects,
                                           ego_state=predicted_ego_state)

        return [predicted_state]

    def _predict_object(self, dynamic_object: DynamicObject, prediction_timestamp: float) \
            -> List[DynamicObject]:
        """
         Performs physical prediction (constant velocity in cartesian frame) for the purpose of short time alignment
        between ego and dynamic objects, for a single object.
        :param dynamic_object: in map coordinates
        :param prediction_timestamp: a timestamp in [sec] to predict_object_trajectories for. In ascending
        order. Global, not relative
        :return: list of predicted objects of the received dynamic object
        """

        prediction_horizon = prediction_timestamp - dynamic_object.timestamp_in_sec

        predicted_x = dynamic_object.x + dynamic_object.velocity * np.cos(dynamic_object.yaw) * prediction_horizon
        predicted_y = dynamic_object.y + dynamic_object.velocity * np.sin(dynamic_object.yaw) * prediction_horizon

        obj_final_cstate = np.array(
            [predicted_x, predicted_y, dynamic_object.yaw, dynamic_object.velocity, 0, DEFAULT_CURVATURE])

        predicted_object_states = PredictionUtils.convert_ctrajectory_to_dynamic_objects(dynamic_object,
                                                                                         obj_final_cstate[np.newaxis, :],
                                                                                         np.array(
                                                                                             [prediction_timestamp]))
        return predicted_object_states
