import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.prediction.constants import LOOKAHEAD_MARGIN_DUE_TO_ROUTE_LINEARIZATION_APPROXIMATION
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import DynamicObject
from mapping.src.model.map_api import MapAPI
from mapping.src.transformations.geometry_utils import CartesianFrame


class RoadFollowingPredictor(Predictor):
    """
    See base class.

    This predictor assumes that each dynamic object is moving in constant
    velocity along the longitudinal axis of of the road. Meaning, there is no acceleration,
    and the speed norm is directed towards advancing in the road while preserving the lateral offset.
    """


    def predict_object_trajectories(self, dynamic_object: DynamicObject, prediction_timestamps: np.ndarray,
                                    nav_plan: NavigationPlanMsg) -> np.ndarray:
        """
        :param dynamic_object: in map coordinates
        :param prediction_timestamps: np array of timestamps to predict_object_trajectories for. In ascending order.
        :param map_api: used in order to get the predicted trajectory and center lanes in map  coordinates
        :param nav_plan: predicted navigation plan of the object
        :return: predicted object's locations in global map coordinates np.array([x, y, theta, vel])
        """

        # we assume the object is travelling exactly on a constant latitude. (i.e., lateral speed = 0)
        object_velocity = np.linalg.norm([dynamic_object.v_x, dynamic_object.v_y])

        # we assume the objects is travelling with a constant velocity, therefore the lookahead distance is
        lookahead_distance = (prediction_timestamps[-1] - dynamic_object.timestamp) * object_velocity
        lookahead_distance += LOOKAHEAD_MARGIN_DUE_TO_ROUTE_LINEARIZATION_APPROXIMATION

        lookahead_route = self._map_api.get_lookahead_points(dynamic_object.road_localization.road_id,
                                                       dynamic_object.road_localization.road_lon,
                                                       lookahead_distance,
                                                       dynamic_object.road_localization.full_lat,
                                                       nav_plan)

        # resample the route to prediction_timestamps
        predicted_distances_from_start = object_velocity * (prediction_timestamps - dynamic_object.timestamp) # assuming constant velocity
        route_xy, _ = CartesianFrame.resample_curve(curve=lookahead_route,
                                                 arbitrary_curve_sampling_points=predicted_distances_from_start)

        # add yaw and velocity
        velocity_column = np.ones(shape=[route_xy.shape[0], 1]) * object_velocity
        route_x_y_theta_v = np.c_[CartesianFrame.add_yaw(route_xy), velocity_column]

        return route_x_y_theta_v
