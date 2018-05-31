from typing import List
import numpy as np

from decision_making.src.global_constants import PREDICTION_LOOKAHEAD_COMPENSATION_RATIO, WERLING_TIME_RESOLUTION
from decision_making.src.planning.types import CartesianTrajectory, CartesianPath2D, FS_SV, FS_SX
from decision_making.src.prediction.ego_aware_prediction.ended_maneuver_params import EndedManeuverParams
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.state.state import DynamicObject
from decision_making.src.state.state_utils import get_object_fstate
from mapping.src.service.map_service import MapService
from mapping.src.transformations.geometry_utils import CartesianFrame


class PredictionUtils:
    @staticmethod
    def convert_to_maneuver_spec(object_state: DynamicObject,
                                 ended_maneuver_params: EndedManeuverParams) -> ManeuverSpec:

        """
        Converts the parameters of the maneuver to a complete maneuver spec
        :param object_state: the dynamic object to predict
        :param ended_maneuver_params: the maneuver parameters to be converted to maneuver spec
        """

        assert -0.5 <= ended_maneuver_params.lat_normalized <= 0.5

        map_api = MapService.get_instance()
        road_id = object_state.road_localization.road_id

        # Get Object's Frenet frame
        road_frenet = map_api.get_road_center_frenet_frame(road_id=road_id)

        # Object's initial state in Frenet frame
        obj_init_fstate = get_object_fstate(object_state=object_state, frenet_frame=road_frenet)

        # Calculate object's initial state in Frenet frame according to model
        road_center_lanes_lat = map_api.get_center_lanes_latitudes(road_id=road_id)
        object_center_lane_latitude = road_center_lanes_lat[object_state.road_localization.lane_num]
        lane_width = map_api.get_road(road_id=road_id).lane_width
        num_lanes = map_api.get_road(road_id=road_id).lanes_num
        road_width = lane_width * num_lanes

        # Motion model (in Frenet frame)
        t_axis = np.arange(0.0, ended_maneuver_params.T_s + 10 * np.finfo(float).eps, WERLING_TIME_RESOLUTION)
        # Calculate velocity according to average acceleration
        s_v_vec = obj_init_fstate[FS_SV] + ended_maneuver_params.avg_s_a * t_axis
        # Clip negative velocities if starting velocity is positive
        if obj_init_fstate[FS_SV] > 0:
            s_v_vec = np.clip(s_v_vec, 0.0, np.inf)
        s_x_vec = obj_init_fstate[FS_SX] + np.cumsum(s_v_vec * WERLING_TIME_RESOLUTION)

        s_x_final = s_x_vec[-1]
        s_v_final = s_v_vec[-1]
        s_a_final = ended_maneuver_params.s_a_final
        d_x_final = (-road_width / 2.0 + object_center_lane_latitude) + lane_width * (
        ended_maneuver_params.relative_lane + ended_maneuver_params.lat_normalized)
        d_v_final = 0.0
        d_a_final = 0.0

        obj_final_fstate = np.array([s_x_final, s_v_final, s_a_final, d_x_final, d_v_final, d_a_final])

        return ManeuverSpec(init_state=obj_init_fstate, final_state=obj_final_fstate, T_s=ended_maneuver_params.T_s,
                            T_d=ended_maneuver_params.T_s)

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
                                   range(len(prediction_timestamps))]

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
