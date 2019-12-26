from logging import Logger
from typing import List, Dict

import numpy as np
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import FrenetTrajectories2D, \
    FrenetStates2D, FrenetState1D, FrenetTrajectories1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.prediction.utils.frenet_prediction_utils import FrenetPredictionUtils
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

    def predict_1d_frenet_states(self, objects_fstates: FrenetState1D, horizons: np.ndarray) -> FrenetTrajectories1D:
        """
        See base class
        """
        return FrenetPredictionUtils.predict_1d_frenet_states(objects_fstates, horizons)

    def predict_2d_frenet_states(self, objects_fstates: FrenetStates2D, horizons: np.ndarray) -> FrenetTrajectories2D:
        """
        See base class
        """

        return FrenetPredictionUtils.predict_2d_frenet_states(objects_fstates, horizons)
