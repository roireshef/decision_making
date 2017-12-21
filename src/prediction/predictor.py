import traceback
import copy
from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List, Type

import numpy as np

from decision_making.src.planning.types import CartesianTrajectory
from decision_making.src.planning.types import C_X, C_Y, C_THETA, C_V
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

    @abstractmethod
    def predict_object_on_road(self, dynamic_object: DynamicObject, prediction_timestamps: np.ndarray) -> List[
        DynamicObject]:
        """
        computes future locations, yaw and velocities for an object directly in the road coordinates-frame
        :param dynamic_object: dynamic object
        :param prediction_timestamps: np array of timestamps in [sec] to predict_object_trajectories for. In ascending order.
        Global, not relative
        :return: a list of dynamic objects with future localizations that correspond to prediction_timestamps
        """

    def _convert_predictions_to_dynamic_objects(self, dynamic_object: DynamicObject, predictions: CartesianTrajectory,
                                                prediction_timestamps: np.ndarray) -> List[DynamicObject]:
        """
        given original dynamic object, its predictions, and their respective time stamps, creates a list of dynamic
         objects corresponding to the predicted object in those timestamps.
        :param dynamic_object:
        :param predictions:
        :param prediction_timestamps:
        :return:
        """
        # Initiate array of DynamicObject at predicted times
        predicted_object_states: List[DynamicObject] = list()

        # Fill with predicted state
        for t_ind in range(len(prediction_timestamps)):
            predicted_object_states.append(
                dynamic_object.set_cartesian_state(timestamp_in_sec=prediction_timestamps[t_ind],
                                                   cartesian_state=predictions[t_ind]))

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
        return self._convert_predictions_to_dynamic_objects(dynamic_object, predictions, prediction_timestamps)

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
        predicted_ego_states = self._predict_object_state(ego_state, prediction_timestamps)
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
