import numpy as np
from logging import Logger
from typing import List, Dict

from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import FS_SX, FS_SV, FS_DX, FrenetTrajectories2D, \
    FrenetStates2D, FrenetState1D, FrenetTrajectories1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, DynamicObject


# TODO: Consider making predictors static classes
class RoadFollowingPredictor(EgoAwarePredictor):
    """
    Dynamic objects are predicted as continuing in the same intra road lat and following the road's curve in constant
    velocity (velocity is assumed to be in the road's direction, meaning no lateral movement)
    """

    def __init__(self, logger: Logger):
        super().__init__(logger)

    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray,
                        action_trajectory: SamplableTrajectory) \
            -> Dict[int, List[DynamicObject]]:
        """
        Predict the future of the specified objects, for the specified timestamps
        :param state: the initial state to begin prediction from. Though predicting a single object, the full state
        provided to enable flexibility in prediction given state knowledge
        :param object_ids: a list of ids of the specific objects to predict
        :param prediction_timestamps: np array of timestamps in [sec] to predict the object for. In ascending order.
        Global, not relative
        :param action_trajectory: the ego's planned action trajectory.
        :return: a mapping between object id to the list of future dynamic objects of the matching object
        """

        if len(object_ids) == 0:
            return {}

        objects = [State.get_object_from_state(state=state, target_obj_id=obj_id)
                   for obj_id in object_ids]

        objects_fstates = [obj.map_state.lane_fstate for obj in objects]

        first_timestamp = State.get_object_from_state(state=state, target_obj_id=object_ids[0]).timestamp_in_sec
        predictions = self.predict_2d_frenet_states(np.array(objects_fstates), prediction_timestamps - first_timestamp)

        # Create a dictionary from predictions
        predicted_objects_states_dict = {obj.obj_id: [
            objects[obj_idx].clone_from_map_state(MapState(predictions[obj_idx, time_idx], obj.map_state.lane_id),
                                                  timestamp_in_sec=timestamp)
            for time_idx, timestamp in enumerate(prediction_timestamps)]
            for obj_idx, obj in enumerate(objects)}

        return predicted_objects_states_dict

    def predict_state(self, state: State, prediction_timestamps: np.ndarray,
                      action_trajectory: SamplableTrajectory) -> (List[State]):
        """
        Predicts the future states of the given state, for the specified timestamps
        :param state: the initial state to begin prediction from
        :param prediction_timestamps: np array of timestamps in [sec] to predict states for. In ascending order.
        Global, not relative
        :param action_trajectory: the ego's planned action trajectory. If given, ego is predicted according to it,
        otherwise the predicted ego states will be none.
        :return: a list of non markov predicted states for the requested prediction_timestamp, and a full state for the
        terminal predicted state
        """

        # Simple object-wise prediction
        object_ids = [obj.obj_id for obj in state.dynamic_objects]

        if action_trajectory is not None:
            extended_sampled_action_trajectory = action_trajectory.sample(time_points=prediction_timestamps)

        predicted_objects_states_dict = self.predict_objects(state=state, object_ids=object_ids,
                                                             prediction_timestamps=prediction_timestamps,
                                                             action_trajectory=action_trajectory)

        # Aggregate all object together with ego into list of future states
        future_states: List[State] = list()

        for time_idx in range(len(prediction_timestamps)):
            predicted_dynamic_objects = [future_object_states[time_idx] for future_object_states in
                                         predicted_objects_states_dict.values()]

            if action_trajectory is not None:
                predicted_ego_state = state.ego_state.clone_from_cartesian_state(
                    timestamp_in_sec=prediction_timestamps[time_idx],
                    cartesian_state=extended_sampled_action_trajectory[time_idx])
            else:
                # Note! This uses the same ego object without deep copying it. Since EgoState should be immutable this
                # should be OK.
                predicted_ego_state = state.ego_state

            state = state.clone_with(ego_state=predicted_ego_state,
                                     dynamic_objects=predicted_dynamic_objects)

            future_states.append(state)

        return future_states

    def predict_1d_frenet_states(self, objects_fstates: FrenetState1D, horizons: np.ndarray) -> FrenetTrajectories1D:
        """
        See base class
        """
        T = horizons.shape[0]
        N = objects_fstates.shape[0]
        if N == 0:
            return []
        zero_slice = np.zeros([N, T])

        s = objects_fstates[:, FS_SX, np.newaxis] + objects_fstates[:, np.newaxis, FS_SV] * horizons
        v = np.tile(objects_fstates[:, np.newaxis, FS_SV], T)

        return np.dstack((s, v, zero_slice))

    def predict_2d_frenet_states(self, objects_fstates: FrenetStates2D, horizons: np.ndarray) -> FrenetTrajectories2D:
        """
        See base class
        """

        T = horizons.shape[0]
        N = objects_fstates.shape[0]
        if N == 0:
            return []
        zero_slice = np.zeros([N, T])

        s = objects_fstates[:, FS_SX, np.newaxis] + objects_fstates[:, np.newaxis, FS_SV] * horizons
        v = np.tile(objects_fstates[:, np.newaxis, FS_SV], T)
        d = np.tile(objects_fstates[:, np.newaxis, FS_DX], T)

        return np.dstack((s, v, zero_slice, d, zero_slice, zero_slice))
