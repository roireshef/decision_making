import copy
from typing import List, Type

import numpy as np

from decision_making.src.prediction.columns import PREDICT_X, PREDICT_Y, PREDICT_YAW, PREDICT_VEL
from decision_making.src.state.state import DynamicObject, EgoState, State
from mapping.src.model.map_api import MapAPI


class Predictor:
    """
    Base class for predictors of dynamic objects
    """

    def __init__(self, map_api: MapAPI):
        self._map_api = map_api

    # TODO - check if Type hint works - general for all the code
    def predict_object_trajectories(self, dynamic_object: Type[DynamicObject],
                                    prediction_timestamps: np.ndarray) -> np.ndarray:
        """
        Method to compute future locations, yaw, and velocities for dynamic objects. Returns the np.array used by the
         trajectory planner.
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        Global, not relative
        :return: predicted object's locations in global map coordinates np.array([x, y, theta, vel])
        """
        pass

    def predict_ego_trajectories(self, ego_state: EgoState, prediction_timestamps: np.ndarray) -> np.ndarray:
        """
        Method to compute future locations, yaw, and velocities for ego vehicle. Returns the np.array used by the
         trajectory planner.
        :param ego_state: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :return: ego's predicted locations in global map coordinates np.array([x, y, theta, vel])
        """
        return self.predict_object_trajectories(ego_state, prediction_timestamps)

    def _convert_predictions_to_dynamic_objects(self, dynamic_object: DynamicObject, predictions: np.ndarray,
                                                prediction_timestamps: np.ndarray)->List[Type[DynamicObject]]:
        """
        given original dynamic object, its predictions, and their respective time stamps, creates a list of dynamic
         objects corresponding to the predicted object in those timestamps.
        :param dynamic_object:
        :param predictions:
        :param prediction_timestamps:
        :return:
        """
        # Initiate array of DynamicObject at predicted times
        predicted_object_states: List[Type[DynamicObject]] = [copy.deepcopy(dynamic_object) for x in
                                                        range(len(prediction_timestamps))]
        # Fill with predicted state
        for t_ind, predicted_object_state in enumerate(predicted_object_states):
            predicted_pos = np.array([predictions[t_ind, PREDICT_X], predictions[t_ind, PREDICT_Y], 0.0])
            predicted_yaw = predictions[t_ind, PREDICT_YAW]
            predicted_object_state.timestamp = prediction_timestamps[t_ind]
            predicted_object_state.x = predicted_pos[0]
            predicted_object_state.y = predicted_pos[1]
            predicted_object_state.yaw = predicted_yaw
            # TODO: check consistency between diff of (x,y) and v_x, v_y as calculated below:
            predicted_object_state.v_x = predictions[t_ind, PREDICT_VEL] * np.cos(predicted_yaw)
            predicted_object_state.v_y = predictions[t_ind, PREDICT_VEL] * np.sin(predicted_yaw)
            predicted_object_state.road_localization = DynamicObject.compute_road_localization(predicted_pos,
                                                                                               predicted_yaw,
                                                                                               self._map_api)

        return predicted_object_states

    def _predict_object_state(self, dynamic_object: Type[DynamicObject],
                              prediction_timestamps: np.ndarray) -> List[Type[DynamicObject]]:
        """
        Wrapper method that uses the predict_object_trajectories method, and creates the dynamic object list.
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :return: List of predicted states of the dynamic object where pos/yaw/vel values are predicted using
        predict_object_trajectories. IMPORTANT - returned list must be in the same order as prediction_timestamps.
        """
        predictions = self.predict_object_trajectories(dynamic_object, prediction_timestamps)
        return self._convert_predictions_to_dynamic_objects(dynamic_object, predictions, prediction_timestamps)

    def _predict_ego_state(self, ego_state: EgoState, prediction_timestamps: np.ndarray) -> List[EgoState]:
        """
        Wrapper method that uses the predict_ego_trajectories method, and creates the list of predicted ego states.
        :param ego_state: initial ego state
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :return: List of predicted states of ego where pos/yaw/vel values are predicted using
        predict_ego_trajectories. IMPORTANT - returned list must be in the same order as prediction_timestamps.
        """
        # TODO: update EgoState attributes that are copied and are not part of DynamicObject (as steering_angle)
        predictions = self.predict_ego_trajectories(ego_state, prediction_timestamps)
        return self._convert_predictions_to_dynamic_objects(ego_state, predictions, prediction_timestamps)

    def predict_state(self, state: State, prediction_timestamps: np.ndarray) -> List[State]:
        """
         Wrapper method that uses the _predict_ego_state and _predict_object_state, and creates a list containing the
         complete predicted states.
        :param state: State object
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        Global, not relative
        :return: a list of predicted states.
        """

        # TODO - consider adding reference route so that this method will be able to project the current
        #  state to the reference route, for example to a different lane.
        # TODO - no need to deepcopy states which we clear afterwards. deep copy only what you need.
        initial_state = copy.deepcopy(state)  # protecting the state input from changes
        predicted_states = [copy.deepcopy(state) for x in
                            range(prediction_timestamps.shape[0])]  # creating copies to populate

        ego_state = initial_state.ego_state
        dynamic_objects = initial_state.dynamic_objects
        predicted_ego_states = self._predict_ego_state(ego_state, prediction_timestamps)
        for t_ind in range(len(prediction_timestamps)):
            predicted_states[t_ind].ego_state = predicted_ego_states[t_ind]  # updating ego_state
            predicted_states[t_ind].dynamic_objects.clear()  # clearing dynamic object lists of copied states

        for dynamic_object in dynamic_objects:
            predicted_obj_states = self._predict_object_state(dynamic_object, prediction_timestamps)
            for t_ind in range(len(prediction_timestamps)):
                predicted_states[t_ind].dynamic_objects.append(
                    predicted_obj_states[t_ind])  # adding predicted obj_state

        return predicted_states
