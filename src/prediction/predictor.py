import copy
from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List

import numpy as np
import six

from decision_making.src.planning.types import CartesianTrajectory, GlobalTimeStampInSec, MinGlobalTimeStampInSec
from decision_making.src.state.state import NewDynamicObject, State
from mapping.src.exceptions import LongitudeOutOfRoad


@six.add_metaclass(ABCMeta)
class Predictor:
    """
    Base class for predictors of dynamic objects
    """

    def __init__(self, logger: Logger):
        self._logger = logger

    @abstractmethod
    def predict_object(self, dynamic_object: NewDynamicObject,
                       prediction_timestamps: np.ndarray) -> CartesianTrajectory:
        """
        Method to compute future locations, yaw, and velocities for dynamic objects. Returns the np.array used by the
         trajectory planner.
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps in [sec] to predict_object_trajectories for. In ascending order.
        Global, not relative
        :return: predicted object's locations in global map coordinates np.array([x, y, theta, vel])
        """
        pass

    def predict_state(self, state: State, prediction_timestamps: np.ndarray) -> List[State]:
        """
         Wrapper method that uses the _predict_ego_state and _predict_object_state, and creates a list containing the
         complete predicted states.
        :param state: State object
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        Global, not relative
        :return: a list of predicted states.
        """

        # list of predicted states in future times
        predicted_states: List[State] = list()

        # A list of predited dynamic objects in future times. init with empty lists
        objects_in_predicted_states: List[List[NewDynamicObject]] = [list() for x in range(len(prediction_timestamps))]

        ego_state = state.ego_state
        dynamic_objects = state.dynamic_objects

        # TODO: temporal bug fix - hack: refrain from prediction of ego state if it has the latest timestamp
        # TODO: it solves the bug only for simulation, when prediction_timestamp[0] == ego.timestamp
        if len(prediction_timestamps) == 1 and ego_state.timestamp_in_sec == prediction_timestamps[0]:
            predicted_ego_states = np.array([ego_state])
        else:
            predicted_ego_states = self._predict_object_state(ego_state, prediction_timestamps)

        # populate list of dynamic objects in future times
        for dynamic_object in dynamic_objects:
            try:
                predicted_obj_states = self._predict_object_state(dynamic_object, prediction_timestamps)
                for t_ind in range(len(prediction_timestamps)):
                    objects_in_predicted_states[t_ind].append(predicted_obj_states[t_ind])  # adding predicted obj_state
            except LongitudeOutOfRoad as e:
                self._logger.warning("Prediction of object id %d is out of road. %s", dynamic_object.obj_id,
                                     dynamic_object.__dict__)

        for t_ind in range(len(prediction_timestamps)):
            new_state = State(occupancy_state=copy.deepcopy(state.occupancy_state),
                              dynamic_objects=objects_in_predicted_states[t_ind],
                              ego_state=predicted_ego_states[t_ind])
            predicted_states.append(new_state)

        return predicted_states

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

    @staticmethod
    def _convert_predictions_to_dynamic_objects(dynamic_object: NewDynamicObject, predictions: CartesianTrajectory,
                                                prediction_timestamps: np.ndarray) -> List[NewDynamicObject]:
        """
        given original dynamic object, its predictions, and their respective time stamps, creates a list of dynamic
         objects corresponding to the predicted object in those timestamps.
        :param dynamic_object:
        :param predictions:
        :param prediction_timestamps:
        :return:
        """
        # Initiate array of DynamicObject at predicted times
        predicted_object_states: List[NewDynamicObject] = list()

        # Fill with predicted state
        for t_ind in range(len(prediction_timestamps)):
            predicted_object_states.append(
                dynamic_object.clone_from_cartesian_state(timestamp_in_sec=prediction_timestamps[t_ind],
                                                             cartesian_state=np.concatenate((predictions[t_ind],
                                                                                             np.array([0, 0])))
                                                          )
            )

        return predicted_object_states

    def _predict_object_state(self, dynamic_object: NewDynamicObject,
                              prediction_timestamps: np.ndarray) -> List[NewDynamicObject]:
        """
        Wrapper method that uses the predict_object_trajectories method, and creates the dynamic object list.
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :return: List of predicted states of the dynamic object where pos/yaw/vel values are predicted using
        predict_object_trajectories. IMPORTANT - returned list must be in the same order as prediction_timestamps.
        """
        predictions = self.predict_object(dynamic_object, prediction_timestamps)
        return self._convert_predictions_to_dynamic_objects(dynamic_object, predictions, prediction_timestamps)
