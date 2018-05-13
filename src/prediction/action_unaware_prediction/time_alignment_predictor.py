from typing import List, Dict
import numpy as np
from logging import Logger
import copy

from decision_making.src.exceptions import PredictObjectInPastTimes
from decision_making.src.global_constants import PREDICTION_LOOKAHEAD_COMPENSATION_RATIO
from decision_making.src.planning.types import CartesianTrajectory
from decision_making.src.prediction.action_unaware_prediction.action_unaware_predictor import ActionUnawarePredictor
from decision_making.src.state.state import State, DynamicObject
from mapping.src.service.map_service import MapService
from mapping.src.transformations.geometry_utils import CartesianFrame


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

        # list of predicted states in future times
        predicted_states: List[State] = list()

        predicted_ego_ctrajectory = self._predict_object_ctrajectory(state.ego_state, prediction_timestamps)
        predicted_ego_states = self._convert_predictions_to_dynamic_objects(dynamic_object=state.ego_state,
                                                                           predictions=predicted_ego_ctrajectory,
                                                                           prediction_timestamps=prediction_timestamps)

        # A list of predcited dynamic objects in future times. init with empty lists
        objects_in_predicted_states: List[List[DynamicObject]] = [list() for x in range(len(prediction_timestamps))]

        for dynamic_object in state.dynamic_objects:
            predicted_ctrajectory = self._predict_object_ctrajectory(dynamic_object=dynamic_object,
                                                                     prediction_timestamps=prediction_timestamps)
            predicted_dynamic_object_states = self._convert_predictions_to_dynamic_objects(dynamic_object=dynamic_object,
                                                                                    predictions=predicted_ctrajectory,
                                                                                    prediction_timestamps=prediction_timestamps)
            for timestamp_ind in range(len(prediction_timestamps)):
                objects_in_predicted_states[timestamp_ind].append(predicted_dynamic_object_states[timestamp_ind])

        for timestamp_ind in range(len(prediction_timestamps)):
            new_state = State(occupancy_state=copy.deepcopy(state.occupancy_state),
                              dynamic_objects=objects_in_predicted_states[timestamp_ind],
                              ego_state=predicted_ego_states[timestamp_ind])
            predicted_states.append(new_state)

        return predicted_states

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

        dynamic_objects = [State.get_object_from_state(state=state, target_obj_id=obj_id) for obj_id in object_ids]

        predicted_dynamic_objects: Dict[int, List[DynamicObject]] = dict()

        for dynamic_object in dynamic_objects:
            predicted_ctrajectory = self._predict_object_ctrajectory(dynamic_object=dynamic_object,
                                                                     prediction_timestamps=prediction_timestamps)
            predicted_dynamic_object_states = self._convert_predictions_to_dynamic_objects(dynamic_object=dynamic_object,
                                                                                    predictions=predicted_ctrajectory,
                                                                                    prediction_timestamps=prediction_timestamps)
            predicted_dynamic_objects[dynamic_object.obj_id] = predicted_dynamic_object_states

        return predicted_dynamic_objects

    def align_objects_to_most_recent_timestamp(self, state: State) -> State:
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

        self._logger.debug("Prediction of ego by: %s sec", most_recent_timestamp - ego_timestamp_in_sec)

        return self.predict_state(state=state, prediction_timestamps=np.array([most_recent_timestamp]))[0]

    @staticmethod
    def _convert_predictions_to_dynamic_objects(dynamic_object: DynamicObject, predictions: CartesianTrajectory,
                                                prediction_timestamps: np.ndarray) -> List[DynamicObject]:
        """
        Given original dynamic object, its predictions, and their respective time stamps, creates a list of dynamic
         objects corresponding to the predicted object in those timestamps.
        :param dynamic_object: the original dynamic object
        :param predictions: the ctrajectory prediction of the dynamic object
        :param prediction_timestamps: the prediction timestamps
        :return:creates a list of dynamic objects corresponding to the predicted object ctrajectory in those timestamps.
        """
        # Initiate array of DynamicObject at predicted times
        predicted_object_states: List[DynamicObject] = list()

        # Fill with predicted state
        for t_ind in range(len(prediction_timestamps)):
            predicted_object_states.append(
                dynamic_object.clone_cartesian_state(timestamp_in_sec=prediction_timestamps[t_ind],
                                                     cartesian_state=predictions[t_ind]))
        return predicted_object_states

    def _predict_object_ctrajectory(self, dynamic_object: DynamicObject, prediction_timestamps: np.ndarray) \
            -> CartesianTrajectory:

        """
        Method to compute future locations, yaw, and velocities for dynamic objects. Returns the np.array used by the
        trajectory planner.
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps in [sec] to predict_object_trajectories for. In ascending order.
        Global, not relative
        :return: predicted object's cartesian trajectory in global map coordinates
        """

        # we assume the object is travelling exactly on a constant latitude. (i.e., lateral speed = 0)
        # TODO: handle objects with negative velocities
        # TODO: If v_y is not small, this computation will be incorrect for ego prediction
        object_velocity = dynamic_object.total_speed

        # we assume the objects is travelling with a constant velocity, therefore the lookahead distance is
        predicted_distances_from_start = object_velocity * (prediction_timestamps - dynamic_object.timestamp_in_sec)
        # raise exception if trying to predict an object in past times
        if not np.all(predicted_distances_from_start >= 0.0):
            raise PredictObjectInPastTimes(
                'Trying to predict object (id=%d) with timestamp %f [sec] to past timestamps: %s' % (
                    dynamic_object.obj_id, dynamic_object.timestamp_in_sec, prediction_timestamps))

        lookahead_distance = predicted_distances_from_start[-1]
        lookahead_distance *= PREDICTION_LOOKAHEAD_COMPENSATION_RATIO

        map_based_nav_plan = \
            MapService.get_instance().get_road_based_navigation_plan(dynamic_object.road_localization.road_id)

        lookahead_route, _ = MapService.get_instance().get_lookahead_points(
            dynamic_object.road_localization.road_id,
            dynamic_object.road_localization.road_lon,
            lookahead_distance,
            dynamic_object.road_localization.intra_road_lat,
            map_based_nav_plan)

        # resample the route to prediction_timestamps, assuming constant velocity

        # If lookahead_route's length == 1, then a single-point route_xy is duplicated.
        if lookahead_route.shape[0] > 1:
            _, route_xy, _ = CartesianFrame.resample_curve(curve=lookahead_route,
                                                           arbitrary_curve_sampling_points=predicted_distances_from_start)
        else:
            raise Exception('Predict object (id=%d) has empty or 1 lookahead_route. Object info: %s' % (
                dynamic_object.obj_id, str(dynamic_object)))

        # add yaw and velocity
        route_len = route_xy.shape[0]

        initial_yaw = dynamic_object.yaw
        yaw_vector = np.ones(shape=[route_len, 1]) * initial_yaw
        # Using v_x to preserve the v_x field of dynamic object
        velocity_column = np.ones(shape=[route_len, 1]) * dynamic_object.v_x

        route_x_y_theta_v = np.concatenate((route_xy, yaw_vector, velocity_column), axis=1)

        return route_x_y_theta_v
