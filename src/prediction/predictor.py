import traceback
import copy
from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List, Type

import numpy as np

from decision_making.src.prediction.columns import PREDICT_X, PREDICT_Y, PREDICT_YAW, PREDICT_VEL
from decision_making.src.state.state import DynamicObject, EgoState, State
from mapping.src.exceptions import LongitudeOutOfRoad

import six

from mapping.src.model.localization import RoadLocalization


@six.add_metaclass(ABCMeta)
class Predictor:
    """
    Base class for predictors of dynamic objects
    """

    def __init__(self, logger: Logger):
        self._logger = logger

    @abstractmethod
    def predict_object(self, dynamic_object: DynamicObject,
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

    def predict_ego(self, ego_state: EgoState, prediction_timestamps: np.ndarray) -> np.ndarray:
        """
        Method to compute future locations, yaw, and velocities for ego vehicle. Returns the np.array used by the
         trajectory planner.
        :param ego_state: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :return: ego's predicted locations in global map coordinates np.array([x, y, theta, vel])
        """
        return self.predict_object(ego_state, prediction_timestamps)

    @abstractmethod
    def predict_object_on_road(self, road_localization: RoadLocalization, localization_timestamp: float,
                               prediction_timestamps: np.ndarray) -> List[RoadLocalization]:
        """
        computes future locations, yaw and velocities for an object directly in the road coordinates-frame
        :param road_localization: object's road localization
        :param localization_timestamp: the timestamp of road_localization argument (units are the same as DynamicObject.timestamp)
        :param prediction_timestamps: np array of timestamps to predict future localizations for. In ascending order.
        :return: a list of future localizations that correspond to prediction_timestamps
        """

    def _convert_predictions_to_dynamic_objects(self, dynamic_object: DynamicObject, predictions: np.ndarray,
                                                prediction_timestamps: np.ndarray,
                                                project_velocity_to_global: bool) -> List[DynamicObject]:
        """
        given original dynamic object, its predictions, and their respective time stamps, creates a list of dynamic
         objects corresponding to the predicted object in those timestamps.
        :param project_velocity_to_global: whether th predicted velocity should be converted to global frame
        :param dynamic_object:
        :param predictions:
        :param prediction_timestamps:
        :return:
        """
        # Initiate array of DynamicObject at predicted times
        predicted_object_states: List[DynamicObject] = [copy.deepcopy(dynamic_object) for x in
                                                        range(len(prediction_timestamps))]
        # Fill with predicted state
        for t_ind, predicted_object_state in enumerate(predicted_object_states):
            predicted_pos = np.array([predictions[t_ind, PREDICT_X], predictions[t_ind, PREDICT_Y], 0.0])
            predicted_yaw = predictions[t_ind, PREDICT_YAW]
            predicted_object_state.timestamp_in_sec = prediction_timestamps[t_ind]
            predicted_object_state.x = predicted_pos[0]
            predicted_object_state.y = predicted_pos[1]
            predicted_object_state.yaw = predicted_yaw
            # TODO: check consistency between diff of (x,y) and v_x, v_y as calculated below:
            if project_velocity_to_global:
                predicted_object_state.v_x = predictions[t_ind, PREDICT_VEL] * np.cos(predicted_yaw)
                predicted_object_state.v_y = predictions[t_ind, PREDICT_VEL] * np.sin(predicted_yaw)
            else:
                # We currently assume that the velocity vector is towards object's x axis
                # TODO: remove assumption
                predicted_object_state.v_x = predictions[t_ind, PREDICT_VEL]
                predicted_object_state.v_y = 0.0

        return predicted_object_states

    def _predict_object_state(self, dynamic_object: DynamicObject,
                              prediction_timestamps: np.ndarray) -> List[DynamicObject]:
        """
        Wrapper method that uses the predict_object_trajectories method, and creates the dynamic object list.
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :return: List of predicted states of the dynamic object where pos/yaw/vel values are predicted using
        predict_object_trajectories. IMPORTANT - returned list must be in the same order as prediction_timestamps.
        """
        predictions = self.predict_object(dynamic_object, prediction_timestamps)
        return self._convert_predictions_to_dynamic_objects(dynamic_object, predictions, prediction_timestamps, True)

    def _predict_ego_state(self, ego_state: EgoState, prediction_timestamps: np.ndarray) -> List[EgoState]:
        """
        Wrapper method that uses the predict_ego_trajectories method, and creates the list of predicted ego states.
        :param ego_state: initial ego state
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :return: List of predicted states of ego where pos/yaw/vel values are predicted using
        predict_ego_trajectories. IMPORTANT - returned list must be in the same order as prediction_timestamps.
        """
        # TODO: update EgoState attributes that are copied and are not part of DynamicObject (as steering_angle)
        predictions = self.predict_ego(ego_state, prediction_timestamps)
        return self._convert_predictions_to_dynamic_objects(ego_state, predictions, prediction_timestamps, False)

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
            try:
                predicted_obj_states = self._predict_object_state(dynamic_object, prediction_timestamps)
                for t_ind in range(len(prediction_timestamps)):
                    predicted_states[t_ind].dynamic_objects.append(
                        predicted_obj_states[t_ind])  # adding predicted obj_state
            except LongitudeOutOfRoad as e:
                self._logger.warning("Prediction of object id %d is out of road. %s", dynamic_object.obj_id,
                                     dynamic_object.__dict__)
            except Exception as e:
                # TODO: remove and handle
                self._logger.error("Prediction of object failed: %s. Trace: %s", e, traceback.format_exc())

        return predicted_states

    def align_objects_to_most_recent_timestamp(self, state: State) -> State:
        """
        Returnes state with all objects aligned to the most recent timestamp
        :param state: state containing objects with different timestamps
        :return: new state with all objects aligned
        """
        # TODO: we might want to replace it with the current machine timestamp
        ego_timestamp_in_sec = state.ego_state.timestamp_in_sec
        objects_timestamp_in_sec = [state.dynamic_objects[x].timestamp_in_sec for x in
                                    range(len(state.dynamic_objects))]
        objects_timestamp_in_sec.append(ego_timestamp_in_sec)
        most_recent_timestamp = np.max(objects_timestamp_in_sec)

        return self.predict_state(state=state, prediction_timestamps=np.array([most_recent_timestamp]))[0]
