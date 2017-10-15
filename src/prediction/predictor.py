import copy
from typing import List, Type

import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.prediction.columns import PREDICT_X, PREDICT_Y, PREDICT_YAW, PREDICT_VEL
from decision_making.src.state.state import DynamicObject, EgoState, State
from mapping.src.model.map_api import MapAPI


class Predictor:
    """
    Base class for predictors of dynamic objects
    """

    @classmethod
    def predict_object_trajectories(cls, dynamic_object: Type[DynamicObject], prediction_timestamps: np.ndarray,
                                    map_api: MapAPI,
                                    nav_plan: NavigationPlanMsg) -> np.ndarray:
        """
        Method to compute future locations, yaw, and velocities for dynamic objects. Returns the np.array used by the
         trajectory planner.
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :param map_api: used in order to get the predicted trajectory and center lanes in map  coordinates
        :param nav_plan: predicted navigation plan of the object
        :return: predicted object's locations in global map coordinates np.array([x, y, theta, vel])
        """
        pass

    @classmethod
    def predict_ego_trajectories(cls, ego_state: EgoState, prediction_timestamps: np.ndarray,
                                    map_api: MapAPI,
                                    nav_plan: NavigationPlanMsg) -> np.ndarray:
        """
        Method to compute future locations, yaw, and velocities for ego vehicle. Returns the np.array used by the
         trajectory planner.
        :param ego_state: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :param map_api: used in order to get the predicted trajectory and center lanes in map  coordinates
        :param nav_plan: predicted navigation plan of ego
        :return: ego's predicted locations in global map coordinates np.array([x, y, theta, vel])
        """
        return cls.predict_object_trajectories(ego_state, prediction_timestamps, map_api, nav_plan)

    @staticmethod
    def _convert_predictions_to_dynamic_objects(dynamic_object, predictions, prediction_timestamps):
        # Initiate array of DynamicObject at predicted times
        predicted_object_states = [copy.deepcopy(dynamic_object) for x in range(len(prediction_timestamps))]
        # Fill with predicted state
        for t_ind, predicted_object_state in enumerate(predicted_object_states):
            predicted_object_state.timestamp = prediction_timestamps[t_ind]
            predicted_object_state.x = predictions[t_ind, PREDICT_X]
            predicted_object_state.y = predictions[t_ind, PREDICT_Y]
            predicted_object_state.yaw = predictions[t_ind, PREDICT_YAW]
            # TODO: check consistency between diff of (x,y) and v_x, v_y as calculated below:
            predicted_object_state.v_x = predictions[t_ind, PREDICT_VEL] * np.cos(
                predictions[t_ind, PREDICT_YAW])
            predicted_object_state.v_y = predictions[t_ind, PREDICT_VEL] * np.sin(
                predictions[t_ind, PREDICT_YAW])
        return predicted_object_states

    @classmethod
    def _predict_object_state(cls, dynamic_object: Type[DynamicObject], prediction_timestamps: np.ndarray,
                              map_api: MapAPI,
                              nav_plan: NavigationPlanMsg) -> List[Type[DynamicObject]]:
        """
        Wrapper method that uses the predict_object_trajectories, and creates the dynamic object list.
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :param map_api: used in order to get the predicted trajectory and center lanes in map  coordinates
        :param nav_plan: predicted navigation plan of the object
        :return: List of predicted states of the dynamic object where pos/yaw/vel values are predicted using
        predict_object_trajectories.
        """
        predictions = cls.predict_object_trajectories(dynamic_object, prediction_timestamps, map_api, nav_plan)
        return Predictor._convert_predictions_to_dynamic_objects(dynamic_object, predictions, prediction_timestamps)

    @classmethod
    def _predict_ego_state(cls, ego_state: EgoState, prediction_timestamps: np.ndarray, map_api: MapAPI,
                           nav_plan: NavigationPlanMsg) -> List[EgoState]:
        # TODO: update EgoState attributes that are copied and are not part of DynamicObject (as steering_angle)
        predictions = cls.predict_ego_trajectories(ego_state, prediction_timestamps, map_api, nav_plan)
        return Predictor._convert_predictions_to_dynamic_objects(ego_state, predictions, prediction_timestamps)

    @classmethod
    def predict_state(cls, state: State, prediction_timestamps: np.ndarray, map_api: MapAPI,
                      nav_plan: NavigationPlanMsg) -> List[State]:
        """
         Wrapper method that uses the _predict_ego_state and _predict_object_state, and creates a list containing the
         complete predicted states. TODO - consider adding reference route so that this method will be able to project
         the current state to the reference route, for example to a different lane.
        :param state: State object
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :param map_api: used in order to get the predicted trajectory and center lanes in map  coordinates
        :param nav_plan: predicted navigation plan of the object
        :return: a list of predicted states.
        """
        initial_state = copy.deepcopy(state) # protecting the state input from changes
        predicted_states = [copy.deepcopy(state) for x in range(len(prediction_timestamps))] # creating copies to populate

        ego_state = initial_state.ego_state
        dynamic_objects = initial_state.dynamic_objects
        predicted_ego_states = cls._predict_ego_state(ego_state, prediction_timestamps, map_api, nav_plan)
        for t_ind in range(len(prediction_timestamps)):
            predicted_states[t_ind].ego_state = predicted_ego_states[t_ind] # updating ego_state
            predicted_states[t_ind].dynamic_objects.clear() # clearing dynamic object lists of copied states

        for dynamic_object in dynamic_objects:
            predicted_obj_states = cls._predict_object_state(dynamic_object, prediction_timestamps, map_api, nav_plan)
            for t_ind in range(len(prediction_timestamps)):
                predicted_states[t_ind].dynamic_objects.append(predicted_obj_states[t_ind]) # adding predicted obj_state

        return predicted_states

