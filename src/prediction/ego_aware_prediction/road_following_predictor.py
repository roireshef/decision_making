import copy
import numpy as np
from logging import Logger
from typing import List, Dict, Optional

from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.types import FS_SX, FS_SV, FS_DX
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import State, NewDynamicObject


class RoadFollowingPredictor(EgoAwarePredictor):
    """
    Dynamic objects are predicted as continuing in the same intra road lat and following the road's curve in constant
    velocity (velocity is assumed to be in the road's direction, meaning no lateral movement)
    """

    def __init__(self, logger: Logger):
        super().__init__(logger)

    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray,
                        action_trajectory: Optional[SamplableTrajectory]) -> Dict[int, List[NewDynamicObject]]:
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

        predicted_objects_states_dict: Dict[int, List[NewDynamicObject]] = dict()

        for obj_id in object_ids:
            dynamic_object = State.get_object_from_state(state=state, target_obj_id=obj_id)

            future_states = self._predict_object(dynamic_object, prediction_timestamps)

            predicted_objects_states_dict[obj_id] = future_states

        return predicted_objects_states_dict

    def predict_state(self, state: State, prediction_timestamps: np.ndarray,
                      action_trajectory: Optional[SamplableTrajectory]) \
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
                predicted_ego_state = None

            state = State(occupancy_state=copy.deepcopy(state.occupancy_state),
                          ego_state=predicted_ego_state,
                          dynamic_objects=predicted_dynamic_objects)

            future_states.append(state)

        return future_states

    def _predict_object(self, dynamic_object: NewDynamicObject, prediction_timestamps: np.ndarray) \
            -> List[NewDynamicObject]:
        """
        Method to compute future locations, yaw, and velocities for dynamic objects. Dynamic objects are predicted as
        continuing in the same intra road lat and following the road's curve in constant
        velocity (velocity is assumed to be in the road's direction, meaning no lateral movement)
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps in [sec] to predict_object_trajectories for. In ascending
        order. Global, not relative
        :return: list of predicted objects of the received dynamic object
        """

        predicted_object_states = []
        obj_fstate = dynamic_object.map_state.road_fstate
        for timestamp in prediction_timestamps:

            horizon = timestamp - dynamic_object.timestamp_in_sec
            obj_terminal_fstate = np.array([obj_fstate[FS_SX] + obj_fstate[FS_SV] * horizon, obj_fstate[FS_SV], 0,
                                            obj_fstate[FS_DX], 0, 0])

            #TODO: Note!! This works only when the road id doesn't change during prediction
            predicted_object_states.append(dynamic_object.clone_from_map_state(timestamp_in_sec=timestamp,
                                                              map_state=MapState(road_fstate=obj_terminal_fstate,
                                                                                 road_id=dynamic_object.map_state.road_id)))

        return predicted_object_states
