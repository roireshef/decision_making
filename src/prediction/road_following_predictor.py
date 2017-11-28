from typing import List

import numpy as np

from decision_making.src.prediction.constants import PREDICTION_LOOKAHEAD_LINEARIZATION_MARGIN
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import DynamicObject, RoadLocalization
from mapping.src.service.map_service import MapService
from mapping.src.transformations.geometry_utils import CartesianFrame


# TODO: implement all interface's methods!
class RoadFollowingPredictor(Predictor):
    """
    See base class.

    This predictor assumes that each dynamic object is moving in constant
    velocity along the longitudinal axis of of the road. Meaning, there is no acceleration,
    and the speed norm is directed towards advancing in the road while preserving the lateral offset.
    In addition, it uses the navigation plan generated by MapAPI, given the object's current lane
    """

    def predict_object_on_road(self, road_localization: RoadLocalization, localization_timestamp: float,
                               prediction_timestamps: np.ndarray) -> List[RoadLocalization]:
        pass

    def predict_object(self, dynamic_object: DynamicObject,
                       prediction_timestamps: np.ndarray) -> np.ndarray:
        """
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        Global, not relative
        :return: predicted object's locations in global map coordinates np.array([x, y, theta, vel])
        """

        # we assume the object is travelling exactly on a constant latitude. (i.e., lateral speed = 0)
        object_velocity = np.linalg.norm([dynamic_object.v_x, dynamic_object.v_y])

        # we assume the objects is travelling with a constant velocity, therefore the lookahead distance is
        lookahead_distance = (prediction_timestamps[-1] - dynamic_object.timestamp_in_sec) * object_velocity
        lookahead_distance += PREDICTION_LOOKAHEAD_LINEARIZATION_MARGIN

        # TODO: Handle negative prediction times. For now, we take only t >= 0
        lookahead_distance = np.maximum(lookahead_distance, 0.0)

        map_based_nav_plan = MapService.get_instance().get_road_based_navigation_plan(dynamic_object.road_localization.road_id)

        lookahead_route, initial_yaw = MapService.get_instance().get_lookahead_points(dynamic_object.road_localization.road_id,
                                                                          dynamic_object.road_localization.road_lon,
                                                                          lookahead_distance,
                                                                          dynamic_object.road_localization.full_lat,
                                                                          map_based_nav_plan)

        # resample the route to prediction_timestamps
        predicted_distances_from_start = object_velocity * (prediction_timestamps - dynamic_object.timestamp_in_sec) # assuming constant velocity
        # TODO: Handle negative prediction times. For now, we take only t >= 0
        predicted_distances_from_start = np.maximum(predicted_distances_from_start, 0.0)
        route_xy, _ = CartesianFrame.resample_curve(curve=lookahead_route,
                                                 arbitrary_curve_sampling_points=predicted_distances_from_start)
        # add yaw and velocity
        route_len = route_xy.shape[0]
        velocity_column = np.ones(shape=[route_len, 1]) * object_velocity

        if route_len > 1:
            route_x_y_theta_v = np.c_[CartesianFrame.add_yaw(route_xy), velocity_column]
            # route_x_y_theta_v[:,2] += initial_yaw
        else:
            yaw_vector = np.ones(shape=[route_len, 1]) * initial_yaw
            route_x_y_theta_v = np.concatenate((route_xy, yaw_vector, velocity_column), axis=1)

        return route_x_y_theta_v
