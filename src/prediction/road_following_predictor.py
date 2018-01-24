from typing import List

import numpy as np

from decision_making.src.exceptions import PredictObjectInPastTimes, PredictedObjectHasNegativeVelocity
from decision_making.src.global_constants import PREDICTION_LOOKAHEAD_COMPENSATION_RATIO
from decision_making.src.planning.types import C_X, C_Y, CartesianTrajectory
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import DynamicObject
from mapping.src.service.map_service import MapService
from mapping.src.transformations.geometry_utils import CartesianFrame


class RoadFollowingPredictor(Predictor):
    """
    See base class.

    This predictor assumes that each dynamic object is moving in constant
    velocity and constant acceleration along the longitudinal axis of of the road.
    Meaning, there is no acceleration in the lateral direction, and the speed is directed towards advancing
    in the road while preserving the lateral offset.
    In addition, it uses the navigation plan generated by MapAPI, given the object's current lane
    """

    def predict_object_on_road(self, dynamic_object: DynamicObject, prediction_timestamps: np.ndarray) -> List[
        DynamicObject]:
        """
        See base class
        """
        map_based_nav_plan = MapService.get_instance().get_road_based_navigation_plan(
            dynamic_object.road_localization.road_id)

        predicted_object_states = []
        for prediction_timestamp in prediction_timestamps:
            # Advance on road according to plan in constant speed, constant acceleration
            prediction_time_axis = prediction_timestamp - dynamic_object.timestamp_in_sec
            predicted_advancement_in_longitude = dynamic_object.v_x * prediction_time_axis + 0.5 * dynamic_object.acceleration_lon * np.square(
                prediction_time_axis)
            predicted_road_id, predicted_road_lon = \
                MapService.get_instance().advance_on_plan(initial_road_id=dynamic_object.road_localization.road_id,
                                                          initial_lon=dynamic_object.road_localization.road_lon,
                                                          lookahead_dist=predicted_advancement_in_longitude,
                                                          navigation_plan=map_based_nav_plan)

            # Convert predicted road localization to new cartesian state
            predicted_location, predicted_orientation = MapService.get_instance().convert_road_to_global_coordinates(
                road_id=predicted_road_id,
                lon=predicted_road_lon,
                lat=dynamic_object.road_localization.intra_road_lat)

            predicted_cartesian_state = np.array(
                [predicted_location[C_X], predicted_location[C_Y], predicted_orientation, dynamic_object.v_x])

            # Create new dynamic object
            predicted_object = dynamic_object.clone_cartesian_state(timestamp_in_sec=prediction_timestamp,
                                                                    cartesian_state=predicted_cartesian_state)
            predicted_object_states.append(predicted_object)

        return predicted_object_states

    def predict_object(self, dynamic_object: DynamicObject,
                       prediction_timestamps: np.ndarray) -> CartesianTrajectory:
        """
        See base class
        """

        # we assume the object is travelling exactly on a constant latitude. (i.e., lateral speed = 0)
        # TODO: handle objects with negative velocities
        object_velocity = np.abs(dynamic_object.v_x)
        if object_velocity < 0.0:
            raise PredictedObjectHasNegativeVelocity(
                'Object with id (%d) velocity is %f. Prediction timestamps: %s. Object data: %s' % (
                    dynamic_object.obj_id,
                    object_velocity, prediction_timestamps, dynamic_object))

        # we assume the objects is travelling with a constant velocity, therefore the lookahead distance is
        lookahead_distance = (prediction_timestamps[-1] - dynamic_object.timestamp_in_sec) * object_velocity
        # raise exception if trying to predict an object in past times
        if lookahead_distance < 0.0:
            raise PredictObjectInPastTimes(
                'Trying to predict object (id=%d) with timestamp %f [sec] to past timestamps: %s' % (
                    dynamic_object.obj_id, dynamic_object.timestamp_in_sec, prediction_timestamps))
        lookahead_distance *= PREDICTION_LOOKAHEAD_COMPENSATION_RATIO

        map_based_nav_plan = \
            MapService.get_instance().get_road_based_navigation_plan(dynamic_object.road_localization.road_id)

        lookahead_route, initial_yaw = MapService.get_instance().get_lookahead_points(
            dynamic_object.road_localization.road_id,
            dynamic_object.road_localization.road_lon,
            lookahead_distance,
            dynamic_object.road_localization.intra_road_lat,
            map_based_nav_plan)

        # resample the route to prediction_timestamps, assuming constant velocity
        predicted_distances_from_start = object_velocity * (prediction_timestamps - dynamic_object.timestamp_in_sec)
        # raise exception if trying to predict an object in past times
        if not np.all(predicted_distances_from_start >= 0.0):
            raise PredictObjectInPastTimes(
                'Trying to predict object (id=%d) with timestamp %f [sec] to past timestamps: %s' % (
                    dynamic_object.obj_id, dynamic_object.timestamp_in_sec, prediction_timestamps))

        # If lookahead_route's length == 1, then a single-point route_xy is duplicated. The yaw can not be calculated.
        # If lookahead_route's length > 1 but route_xy length == 1, then again the yaw can not be calculated.
        # In these cases yaw_vector contains a duplicated value of initial_yaw.
        yaw_vector = None
        if lookahead_route.shape[0] > 1:
            _, route_xy, _ = CartesianFrame.resample_curve(curve=lookahead_route,
                                                        arbitrary_curve_sampling_points=predicted_distances_from_start)
            if route_xy.shape[0] == 1:
                yaw_vector = np.ones(shape=[route_xy.shape[0], 1]) * initial_yaw
        elif lookahead_route.shape[0] == 1:
            route_xy = np.reshape(np.tile(lookahead_route[0], predicted_distances_from_start.shape[0]), (-1, 2))
            yaw_vector = np.ones(shape=[route_xy.shape[0], 1]) * initial_yaw
        else:
            raise Exception('Predict object (id=%d) has empty lookahead_route. Object info: %s' % (
                dynamic_object.obj_id, dynamic_object.__dict__))

        # add yaw and velocity
        route_len = route_xy.shape[0]
        velocity_column = np.ones(shape=[route_len, 1]) * object_velocity

        if yaw_vector is None:
            route_x_y_theta_v = np.c_[CartesianFrame.add_yaw(route_xy), velocity_column]
        else:
            route_x_y_theta_v = np.concatenate((route_xy, yaw_vector, velocity_column), axis=1)

        return route_x_y_theta_v
