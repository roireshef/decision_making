from logging import Logger
from typing import List, Dict

import copy
import numpy as np

from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.types import C_X, C_V, C_YAW, C_Y
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State, DynamicObject
from mapping.src.service.map_service import MapService
from decision_making.src.prediction.ego_aware_prediction.maneuver_recognition.maneuver_classifier import ManeuverClassifier
from decision_making.src.prediction.ego_aware_prediction.trajectory_generation.trajectory_generator import TrajectoryGenerator


class ManeuverBasedPredictor(EgoAwarePredictor):
    """
    Maneuver-based modular predictor uses a parametrized motion model to predict each object trajectory.
    It introduces the following modules:
    - Maneuver classification module: specifies the maneuver parameters of each objects's trajectory.
    - Trajectory generation module: generates object trajectory according to a specified motion model and the parameters
      provided by the maneuver classification module.
    """

    def __init__(self, logger: Logger, maneuver_classifier: ManeuverClassifier,
                 trajectory_generator: TrajectoryGenerator):
        """
        :param logger: logger
        :param maneuver_classifier: maneuver classifier module
        :param trajectory_generator: trajectory generation module
        """
        super().__init__(logger=logger)
        self._trajectory_generator = trajectory_generator
        self._maneuver_classifier = maneuver_classifier

    def predict_state(self, state: State, prediction_timestamps: np.ndarray, action_trajectory: SamplableTrajectory) \
            -> (List[State]):
        """
        Predicts the future states of the given state, for the specified timestamps
        :param state: the initial state to begin prediction from
        :param prediction_timestamps: np array of timestamps in [sec] to predict states for. In ascending order.
        Global, not relative
        :param action_trajectory: the ego's planned action trajectory
        :return: a list of non markov predicted states for the requested prediction_timestamp, and a full state for the
        terminal predicted state
        """

        # Simple object-wise prediction
        object_ids = [obj.obj_id for obj in state.dynamic_objects]

        action_trajectory_sampled_extended = action_trajectory.sample(time_points=prediction_timestamps)
        action_trajectory_sampled = action_trajectory_sampled_extended[:, [C_X, C_Y, C_YAW, C_V]]

        predicted_objects_states_dict = self.predict_objects(state=state, object_ids=object_ids,
                                                             prediction_timestamps=prediction_timestamps,
                                                             action_trajectory=action_trajectory)

        # Aggregate all object together with ego into list of future states
        future_states: List[State] = list()
        for time_idx in range(len(prediction_timestamps)):
            predicted_dynamic_objects = [future_object_states[time_idx] for future_object_states in
                                         predicted_objects_states_dict.values()]
            predicted_ego_state = state.ego_state.clone_cartesian_state(
                timestamp_in_sec=prediction_timestamps[time_idx],
                cartesian_state=action_trajectory_sampled[time_idx])

            state = State(occupancy_state=copy.deepcopy(state.occupancy_state),
                          ego_state=predicted_ego_state,
                          dynamic_objects=predicted_dynamic_objects)

            future_states.append(state)

        return future_states

    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray,
                        action_trajectory: SamplableTrajectory) -> Dict[int, List[DynamicObject]]:
        """
        Predicte the future of the specified objects, for the specified timestamps
        :param state: the initial state to begin prediction from. Though predicting a single object, the full state
        provided to enable flexibility in prediction given state knowledge
        :param object_ids: a list of ids of the specific objects to predict
        :param prediction_timestamps: np array of timestamps in [sec] to predict the object for. In ascending order.
        Global, not relative
        :param action_trajectory: the ego's planned action trajectory
        :return: a mapping between object id to the list of future dynamic objects of the matching object
        """

        predicted_objects_states_dict: Dict[int, List[DynamicObject]] = dict()

        for obj_id in object_ids:
            dynamic_object = State.get_object_from_state(state=state, target_obj_id=obj_id)
            horizon = prediction_timestamps[-1] - dynamic_object.timestamp_in_sec
            predicted_maneuver_spec = self._maneuver_classifier.classify_maneuver(state=state, object_id=obj_id, T_s=horizon)

            frenet_frame = MapService.get_instance().get_road_center_frenet_frame(
                road_id=dynamic_object.road_localization.road_id)

            init_time = dynamic_object.timestamp_in_sec

            maneuver_samplable_trajectory = self._trajectory_generator.generate_trajectory(
                timestamp_in_sec=init_time,
                frenet_frame=frenet_frame,
                predicted_maneuver_spec=predicted_maneuver_spec)

            maneuver_trajectory_extended = maneuver_samplable_trajectory.sample(time_points=prediction_timestamps)
            maneuver_trajectory = maneuver_trajectory_extended[:, [C_X, C_Y, C_YAW, C_V]]

            future_states = [
                dynamic_object.clone_cartesian_state(timestamp_in_sec=prediction_timestamps[x],
                                                     cartesian_state=maneuver_trajectory[x]) for
                x in range(len(prediction_timestamps))]

            predicted_objects_states_dict[obj_id] = future_states

        return predicted_objects_states_dict
