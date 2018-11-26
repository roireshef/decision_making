from decision_making.src.prediction.ego_aware_prediction.ended_maneuver_params import EndedManeuverParams
from decision_making.src.prediction.ego_aware_prediction.maneuver_recognition.manuever_classifier import \
    ManeuverClassifier
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.prediction.utils.prediction_utils import PredictionUtils
from decision_making.src.state.state import State
from mapping.src.service.map_service import MapService


class ConstantVelocityManeuverClassifier(ManeuverClassifier):
    """
    Simple physics-based maneuver predictor in Frenet-frame.
    Based on Werling motion planner.
    Assumptions:
        - Object keeps speed and lateral position
    """

    def classify_maneuver(self, state: State, object_id: int, maneuver_horizon: float) -> ManeuverSpec:
        """
        Predicts the type of maneuver an object will execute
        :param state: world state
        :param object_id: of the object to predict
        :param maneuver_horizon: the horizon of the maneuver to classify
        :return: maneuver specification of an object
        """

        object_state = State.get_object_from_state(state=state, target_obj_id=object_id)
        map_state = object_state.map_state

        map_api = MapService.get_instance()

        # Calculate object's initial state in Frenet frame according to model
        lane_width = map_api.get_road(road_segment_id=map_state.road_id).lane_width

        # Fetch trajectory parameters
        avg_s_a = 0.0
        s_a_final = 0.0
        relative_lane = 0.0

        # Keep same normalized latitude in lane
        lat = (map_state.intra_lane_lat - lane_width / 2.0) / lane_width

        return PredictionUtils.convert_to_maneuver_spec(object_state=object_state,
                                                        ended_maneuver_params=EndedManeuverParams(T_s=maneuver_horizon,
                                                                                                  avg_s_a=avg_s_a,
                                                                                                  s_a_final=s_a_final,
                                                                                                  relative_lane=relative_lane,
                                                                                                  lat_normalized=lat))
