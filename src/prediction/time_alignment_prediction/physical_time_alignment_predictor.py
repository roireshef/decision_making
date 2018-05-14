import copy
from logging import Logger
from typing import List

import numpy as np

from decision_making.src.planning.types import GlobalTimeStampInSec, MinGlobalTimeStampInSec
from decision_making.src.prediction.time_alignment_prediction.time_alignment_predictor import TimeAlignmentPredictor
from decision_making.src.prediction.utils.prediction_utils import PredictionUtils
from decision_making.src.state.state import State, DynamicObject


class PhysicalTimeAlignmentPredictor(TimeAlignmentPredictor):
    """
    Performs physical prediction for the purpose of short time alignment between ego and dynamic objects.
    Logic should be re-considered if the time horizon gets too large.
    """

    def __init__(self, logger: Logger):
        super().__init__(logger)

    def align_objects_to_most_recent_timestamp(self, state: State,
                                               current_timestamp: GlobalTimeStampInSec=MinGlobalTimeStampInSec) -> State:
        """
        Returns state with all objects aligned to the most recent timestamp.
        Most recent timestamp is taken as the max between the current_timestamp, and the most recent
        timestamp of all objects in the scene.
        :param state: state containing objects with different timestamps
        :param current_timestamp: current timestamp in global time in [sec]
        :return: new state with all objects aligned
        """

        ego_timestamp_in_sec = state.ego_state.timestamp_in_sec

        objects_timestamp_in_sec = [state.dynamic_objects[x].timestamp_in_sec for x in
                                    range(len(state.dynamic_objects))]
        objects_timestamp_in_sec.append(ego_timestamp_in_sec)

        most_recent_timestamp = np.max(objects_timestamp_in_sec)
        most_recent_timestamp = np.maximum(most_recent_timestamp, current_timestamp)

        self._logger.debug("Prediction of ego by: %s sec", most_recent_timestamp - ego_timestamp_in_sec)

        return self._predict_state(state=state, prediction_timestamps=np.array([most_recent_timestamp]))[0]

    def _predict_state(self, state: State, prediction_timestamps: np.ndarray) -> List[State]:
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

        # A list of predcited dynamic objects in future times. init with empty lists
        objects_in_predicted_states: List[List[DynamicObject]] = [list() for x in range(len(prediction_timestamps))]

        for dynamic_object in state.dynamic_objects:
            predicted_objects = self._predict_object(dynamic_object=dynamic_object,
                                                     prediction_timestamps=prediction_timestamps)

            for predicted_object_states in predicted_objects:
                for timestamp_ind in range(len(prediction_timestamps)):
                    objects_in_predicted_states[timestamp_ind].append(predicted_object_states[timestamp_ind])

        predicted_ego_states = self._predict_object(dynamic_object=state.ego_state,
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
        Method to compute future locations, yaw, and velocities for dynamic objects. Returns the np.array used by the
        trajectory planner.
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps in [sec] to predict_object_trajectories for. In ascending
        order. Global, not relative
        :return: predicted object's cartesian trajectory in global map coordinates
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
