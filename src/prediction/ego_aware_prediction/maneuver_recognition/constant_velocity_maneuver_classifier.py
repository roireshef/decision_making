from decision_making.src.planning.types import FS_DX, FS_SX
from decision_making.src.prediction.ego_aware_prediction.ended_maneuver_params import EndedManeuverParams
from decision_making.src.prediction.ego_aware_prediction.maneuver_recognition.manuever_classifier import \
    ManeuverClassifier
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.prediction.utils.prediction_utils import PredictionUtils
from decision_making.src.planning.behavioral.state import State
from decision_making.src.utils.map_utils import MapUtils


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

        # Calculate object's initial state in Frenet frame according to model
        lane_width = MapUtils.get_lane_width(map_state.lane_id, s=map_state.lane_fstate[FS_SX])

        # Fetch trajectory parameters
        avg_s_a = 0.0
        s_a_final = 0.0
        relative_lane = 0.0

        # Keep same normalized latitude in lane
        lat = map_state.lane_fstate[FS_DX] / lane_width

        return PredictionUtils.convert_to_maneuver_spec(object_state=object_state,
                                                        ended_maneuver_params=EndedManeuverParams(T_s=maneuver_horizon,
                                                                                                  avg_s_a=avg_s_a,
                                                                                                  s_a_final=s_a_final,
                                                                                                  relative_lane=relative_lane,
                                                                                                  lat_normalized=lat))
