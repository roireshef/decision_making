from typing import Optional

from decision_making.src.prediction.ego_aware_prediction.maneuver_recognition.werling_maneuver_classifier import \
    WerlingManeuverClassifier
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.state.state import State
from mapping.src.service.map_service import MapService


class ConstantVelocityManeuverClassifier(WerlingManeuverClassifier):
    """
    Simple physics-based maneuver predictor in Frenet-frame.
    Based on Werling motion planner.
    Assumptions:
        - Object keeps speed and lateral position
    """

    def __init__(self, T_s: float):
        """
        :param T_s: Maneuver duration in s road coordinate [sec]
        """
        self._T_s = T_s

    def classify_maneuver(self, state: State, object_id: int, T_s: Optional[float] = None) -> ManeuverSpec:
        """
        Predicts the type of maneuver an object will execute
        :param state: world state
        :param object_id: of predicted object
        :return: maneuver specification of an object
        """

        if T_s is None:
            T_s = self._T_s

        object_state = State.get_object_from_state(state=state, target_obj_id=object_id)
        road_localization = object_state.road_localization

        map_api = MapService.get_instance()

        # Calculate object's initial state in Frenet frame according to model
        lane_width = map_api.get_road(road_id=road_localization.road_id).lane_width

        # Fetch trajectory parameters
        t_d = T_s
        avg_s_a = 0.0
        s_a_final = 0.0
        d_in_lanes = 0.0

        # Keep same normalized latitude in lane
        lat = (road_localization.intra_lane_lat - lane_width / 2.0) / lane_width

        return self._generate_maneuver_spec(object_state=object_state, T_s=T_s, T_d=t_d, avg_s_a=avg_s_a,
                                            s_a_final=s_a_final, d_in_lanes=d_in_lanes, lat=lat)
