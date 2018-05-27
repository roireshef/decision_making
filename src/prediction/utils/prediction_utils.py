from typing import List
import numpy as np

from decision_making.src.global_constants import PREDICTION_LOOKAHEAD_COMPENSATION_RATIO
from decision_making.src.planning.types import CartesianTrajectory, CartesianPath2D
from decision_making.src.state.state import DynamicObject
from mapping.src.service.map_service import MapService
from mapping.src.transformations.geometry_utils import CartesianFrame


class PredictionUtils:
    @staticmethod
    def convert_ctrajectory_to_dynamic_objects(dynamic_object: DynamicObject, predictions: CartesianTrajectory,
                                               prediction_timestamps: np.ndarray) -> List[DynamicObject]:
        """
        Given original dynamic object, its predictions, and their respective time stamps, creates a list of dynamic
         objects corresponding to the predicted object in those timestamps.
        :param dynamic_object: the original dynamic object
        :param predictions: the ctrajectory prediction of the dynamic object
        :param prediction_timestamps: the prediction timestamps
        :return:creates a list of dynamic objects corresponding to the predicted object ctrajectory in those timestamps.
        """

        predicted_object_states = [dynamic_object.clone_cartesian_state(timestamp_in_sec=prediction_timestamps[t_ind],
                                                                        cartesian_state=predictions[t_ind]) for t_ind in
                                   range(len(prediction_timestamps)]

        return predicted_object_states

    @staticmethod
    def constant_velocity_x_y_prediction(dynamic_object: DynamicObject, prediction_timestamps: np.ndarray) \
            -> CartesianPath2D:
        """
        """

        # we assume the object is travelling exactly on a constant latitude. (i.e., lateral speed = 0)
        # TODO: handle objects with negative velocities
        # TODO: If v_y is not small, this computation will be incorrect for ego prediction
        object_velocity = dynamic_object.total_speed

        # we assume the objects is travelling with a constant velocity, therefore the lookahead distance is
        predicted_distances_from_start = object_velocity * (prediction_timestamps - dynamic_object.timestamp_in_sec)

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
        elif lookahead_route.shape[0] == 1:
            route_xy = np.reshape(np.tile(lookahead_route[0], predicted_distances_from_start.shape[0]), (-1, 2))
        else:
            raise Exception('Predict object (id=%d) has empty  lookahead_route. Object info: %s' % (
                dynamic_object.obj_id, str(dynamic_object)))

        return route_xy
