from abc import abstractmethod

import numpy as np

from decision_making.src.global_constants import WERLING_TIME_RESOLUTION
from decision_making.src.planning.types import FS_SX, FS_SV
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.state.state import State, DynamicObject
from mapping.src.service.map_service import MapService
from decision_making.src.prediction.ego_aware_prediction.maneuver_recognition.maneuver_classifier import ManeuverClassifier
from prediction_research.src.utils.state_utils import get_object_fstate


class WerlingManeuverClassifier(ManeuverClassifier):
    """
    Iterates over all possible parameters permutations every time the maneuver classifier is called.
    """

    @abstractmethod
    def classify_maneuver(self, state: State, object_id: int) -> ManeuverSpec:
        """
        Predicts the type of maneuver an object will execute
        Assuming zero acceleration in the initial state
        :param state: world state
        :param object_id: of predicted object
        :return: maneuver specification of an object
        """
        pass

    def _generate_maneuver_spec(self, object_state: DynamicObject, T_s: float, T_d: float, avg_s_a: float,
                                s_a_final: float, d_in_lanes: float, lat: float) -> ManeuverSpec:
        """

        :param object_state:
        :param T_s:
        :param T_d:
        :param avg_s_a:
        :param s_a_final:
        :param d_in_lanes:
        :param lat: lateral location relative to center lane. Normalized to [lanes], therefore limits are (-0.5, 0.5)
        :return:
        """

        assert -0.5 <= lat <= 0.5

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
        t_axis = np.arange(0.0, T_s + 10 * np.finfo(float).eps, WERLING_TIME_RESOLUTION)
        # Calculate velocity according to average acceleration
        s_v_vec = obj_init_fstate[FS_SV] + avg_s_a * t_axis
        # Clip negative velocities if starting velocity is positive
        if obj_init_fstate[FS_SV] > 0:
            s_v_vec = np.clip(s_v_vec, 0.0, np.inf)
        s_x_vec = obj_init_fstate[FS_SX] + np.cumsum(s_v_vec * WERLING_TIME_RESOLUTION)

        s_x_final = s_x_vec[-1]
        s_v_final = s_v_vec[-1]
        s_a_final = s_a_final
        d_x_final = (-road_width/2.0 + object_center_lane_latitude) + lane_width * (d_in_lanes + lat)
        d_v_final = 0.0
        d_a_final = 0.0

        obj_final_fstate = np.array([s_x_final, s_v_final, s_a_final, d_x_final, d_v_final, d_a_final])

        return ManeuverSpec(init_state=obj_init_fstate, final_state=obj_final_fstate, T_s=T_s, T_d=T_d)
