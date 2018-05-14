from typing import List, Dict
import numpy as np
from logging import Logger

from decision_making.src.prediction.action_unaware_prediction.action_unaware_predictor import ActionUnawarePredictor
from decision_making.src.prediction.utils.prediction_utils import PredictionUtils
from decision_making.src.state.state import State, DynamicObject
from mapping.src.transformations.geometry_utils import CartesianFrame


class RoadActionUnawarePredictor(ActionUnawarePredictor):
    """
    Performs simple / naive prediction in road coordinates (road following prediction, constant velocity)
    and returns objects with calculated and cached road coordinates. This is in order to save coordinate conversion time
    for the predictor's clients.
    """

    def __init__(self, logger: Logger):
        super().__init__(logger=logger)

    def predict_objects(self, state: State, object_ids: List[int], prediction_timestamps: np.ndarray) \
            -> Dict[int, List[DynamicObject]]:
        """
        Predicts the future of the specified objects, for the specified timestamps
        :param state: the initial state to begin prediction from. Though predicting a single object, the full state
        provided to enable flexibility in prediction given state knowledge
        :param object_ids: a list of ids of the specific objects to predict
        :param prediction_timestamps: np array of size 1 of timestamp in [sec] to predict the object for. In ascending
        order. Global, not relative
        :return: a mapping between object id to the list of future dynamic objects of the matching object
        """

        # Predict to a single horizon
        assert len(prediction_timestamps) == 1

        dynamic_objects = [State.get_object_from_state(state=state, target_obj_id=obj_id) for obj_id in object_ids]

        predicted_dynamic_objects: Dict[int, List[DynamicObject]] = dict()

        for dynamic_object in dynamic_objects:
            predicted_dynamic_object_states = self._predict_object(dynamic_object=dynamic_object,
                                                                   prediction_timestamps=prediction_timestamps)
            predicted_dynamic_objects[dynamic_object.obj_id] = predicted_dynamic_object_states

        return predicted_dynamic_objects

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

        # Using v_x to preserve the v_x field of dynamic object
        velocity_column = np.ones(shape=[route_len, 1]) * dynamic_object.v_x

        # Adjust yaw to the path's yaw

        # TODO: check if this condition is still relevant
        # If there isn't enough points to determine yaw, leave the initial yaw
        if route_len == 1:
            yaw_vector = np.ones(shape=[route_xy.shape[0], 1]) * dynamic_object.yaw
            route_x_y_theta_v = np.concatenate((route_xy, yaw_vector, velocity_column), axis=1)
        else:
            route_x_y_theta_v = np.c_[CartesianFrame.add_yaw(route_xy), velocity_column]

        predicted_object_states = PredictionUtils.convert_ctrajectory_to_dynamic_objects(dynamic_object=dynamic_object,
                                                                                         predictions=route_x_y_theta_v,
                                                                                         prediction_timestamps=prediction_timestamps)
        return predicted_object_states

