from typing import List
import numpy as np

from decision_making.src.global_constants import PREDICTION_LOOKAHEAD_COMPENSATION_RATIO, WERLING_TIME_RESOLUTION
from decision_making.src.planning.types import CartesianTrajectory, CartesianPath2D, FS_SV, FS_SX, \
    CartesianExtendedTrajectory
from decision_making.src.prediction.ego_aware_prediction.ended_maneuver_params import EndedManeuverParams
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.state.state import NewDynamicObject
from decision_making.src.state.state_utils import get_object_fstate
from mapping.src.service.map_service import MapService
from mapping.src.transformations.geometry_utils import CartesianFrame


class PredictionUtils:
    @staticmethod
    def convert_to_maneuver_spec(object_state: NewDynamicObject,
                                 ended_maneuver_params: EndedManeuverParams) -> ManeuverSpec:

        """
        Converts the parameters of the maneuver to a complete maneuver spec
        :param object_state: the dynamic object to predict
        :param ended_maneuver_params: the maneuver parameters to be converted to maneuver spec
        """

        assert -0.5 <= ended_maneuver_params.lat_normalized <= 0.5

        map_api = MapService.get_instance()
        road_id = object_state.map_state.road_id

        # Get Object's Frenet frame
        road_frenet = map_api.get_road_center_frenet_frame(road_id=road_id)

        # Object's initial state in Frenet frame
        obj_init_fstate = get_object_fstate(object_state=object_state, frenet_frame=road_frenet)

        # Calculate object's initial state in Frenet frame according to model
        road_center_lanes_lat = map_api.get_center_lanes_latitudes(road_id=road_id)
        object_center_lane_latitude = road_center_lanes_lat[object_state.map_state.lane_num]
        lane_width = map_api.get_road(road_id=road_id).lane_width
        num_lanes = map_api.get_road(road_id=road_id).lanes_num
        road_width = lane_width * num_lanes

        s_x_final, s_v_final = PredictionUtils.compute_x_from_average_a(ended_maneuver_params.T_s,
                                                                        ended_maneuver_params.avg_s_a,
                                                                        obj_init_fstate[FS_SX],
                                                                        obj_init_fstate[FS_SV])

        s_a_final = ended_maneuver_params.s_a_final
        d_x_final = (-road_width / 2.0 + object_center_lane_latitude) + lane_width * (
            ended_maneuver_params.relative_lane + ended_maneuver_params.lat_normalized)
        d_v_final = 0.0
        d_a_final = 0.0

        obj_final_fstate = np.array([s_x_final, s_v_final, s_a_final, d_x_final, d_v_final, d_a_final])

        return ManeuverSpec(init_state=obj_init_fstate, final_state=obj_final_fstate, T_s=ended_maneuver_params.T_s,
                            T_d=ended_maneuver_params.T_s)

    @staticmethod
    def compute_x_from_average_a(t: float, avg_a: float, init_x: float, init_v: float) -> (float, float):
        """

        :param t:
        :param avg_a:
        :param init_x:
        :param init_v:
        :return:
        """
        # Motion model (in Frenet frame)
        t_axis = np.arange(0.0, t + 10 * np.finfo(float).eps, WERLING_TIME_RESOLUTION)
        # Calculate velocity according to average acceleration
        s_v_vec = init_v + avg_a * t_axis
        # Clip negative velocities if starting velocity is positive
        if init_v > 0:
            s_v_vec = np.clip(s_v_vec, 0.0, np.inf)
        s_x_vec = init_x + np.cumsum(s_v_vec[1:] * WERLING_TIME_RESOLUTION)
        s_x_vec = np.r_[init_x, s_x_vec]

        return s_x_vec[-1], s_v_vec[-1]

    @staticmethod
    def convert_ctrajectory_to_dynamic_objects(dynamic_object: NewDynamicObject, predictions: CartesianExtendedTrajectory,
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
