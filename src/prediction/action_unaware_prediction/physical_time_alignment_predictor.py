import copy
from logging import Logger
from typing import List, Dict

import numpy as np

from decision_making.src.prediction.action_unaware_prediction.ego_unaware_predictor import EgoUnawarePredictor
from decision_making.src.prediction.utils.prediction_utils import PredictionUtils
from decision_making.src.state.state import State, DynamicObject, EgoState


class PhysicalTimeAlignmentPredictor(EgoUnawarePredictor):
    """
    Performs physical prediction for the purpose of short time alignment between ego and dynamic objects.
    Logic should be re-considered if the time horizon gets too large.
    Dynamic objects are predicted as continuing in the same intra road lat and following the road's curve in constant
    velocity (velocity is assumed to be in the road's direction, meaning no lateral movement)
    Ego is predicted to continue in constant dv and sv.
    """

    def __init__(self, logger: Logger):
        super().__init__(logger)

    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray) \
            -> Dict[int, List[DynamicObject]]:
        dynamic_objects = [State.get_object_from_state(state=state, target_obj_id=obj_id) for obj_id in object_ids]

        predicted_dynamic_objects: Dict[int, List[DynamicObject]] = dict()

        for dynamic_object in dynamic_objects:
            predicted_dynamic_object_states = self._predict_object(dynamic_object=dynamic_object,
                                                                   prediction_timestamps=prediction_timestamps)
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

        # list of predicted states in future times
        predicted_states: List[State] = list()

        # A list of predicted dynamic objects in future times. init with empty lists
        objects_in_predicted_states: List[List[DynamicObject]] = [list() for x in range(len(prediction_timestamps))]

        for dynamic_object in state.dynamic_objects:
            predicted_objects = self._predict_object(dynamic_object=dynamic_object,
                                                     prediction_timestamps=prediction_timestamps)
            for timestamp_ind in range(len(prediction_timestamps)):
                objects_in_predicted_states[timestamp_ind].append(predicted_objects[timestamp_ind])

        predicted_ego_states = self._predict_ego(ego_state=state.ego_state,
                                                 prediction_timestamps=prediction_timestamps)

        for timestamp_ind in range(len(prediction_timestamps)):
            new_state = State(occupancy_state=copy.deepcopy(state.occupancy_state),
                              dynamic_objects=objects_in_predicted_states[timestamp_ind],
                              ego_state=predicted_ego_states[timestamp_ind])
            predicted_states.append(new_state)

        return predicted_states

    def _predict_object(self, dynamic_object: DynamicObject, prediction_timestamps: np.ndarray) \
            -> List[DynamicObject]:

        """
        Method to compute future locations, yaw, and velocities for dynamic objects. Dynamic objects are predicted as
        continuing in the same intra road lat and following the road's curve in constant
        velocity (velocity is assumed to be in the road's direction, meaning no lateral movement)
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps in [sec] to predict_object_trajectories for. In ascending
        order. Global, not relative
        :return: list of predicted objects of the received dynamic object
        """

        route_xy = PredictionUtils.constant_velocity_x_y_prediction(dynamic_object=dynamic_object,
                                                                    prediction_timestamps=prediction_timestamps)
        # add yaw and velocity
        route_len = route_xy.shape[0]

        initial_yaw = dynamic_object.yaw
        yaw_vector = np.ones(shape=[route_len, 1]) * initial_yaw
        # Using v_x to preserve the v_x field of dynamic object
        velocity_column = np.ones(shape=[route_len, 1]) * dynamic_object.v_x

        route_x_y_theta_v = np.concatenate((route_xy, yaw_vector, velocity_column), axis=1)

        predicted_object_states = PredictionUtils.convert_ctrajectory_to_dynamic_objects(dynamic_object=dynamic_object,
                                                                                         predictions=route_x_y_theta_v,
                                                                                         prediction_timestamps=prediction_timestamps)
        return predicted_object_states

    def _predict_ego(self, ego_state: EgoState, prediction_timestamps: np.ndarray) \
            -> List[EgoState]:

        """
        Method to compute future locations, yaw, and velocities for ego. Ego is predicted to continue in constant dv
        and sv.
        :param ego_state: in map coordinates
        :param prediction_timestamps: np array of timestamps in [sec] to predict_object_trajectories for. In ascending
        order. Global, not relative
        :return: a list of predicted objects of the received ego object
        """

        route_xy = PredictionUtils.constant_velocity_x_y_prediction(dynamic_object=ego_state,
                                                                    prediction_timestamps=prediction_timestamps)
        # add yaw and velocity
        route_len = route_xy.shape[0]

        initial_yaw = ego_state.yaw
        yaw_vector = np.ones(shape=[route_len, 1]) * initial_yaw
        # Using v_x to preserve the v_x field of dynamic object
        velocity_column = np.ones(shape=[route_len, 1]) * ego_state.v_x

        route_x_y_theta_v = np.concatenate((route_xy, yaw_vector, velocity_column), axis=1)

        predicted_object_states = PredictionUtils.convert_ctrajectory_to_dynamic_objects(dynamic_object=ego_state,
                                                                                         predictions=route_x_y_theta_v,
                                                                                         prediction_timestamps=prediction_timestamps)
        return predicted_object_states
