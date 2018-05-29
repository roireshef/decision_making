import numpy as np

from decision_making.src.prediction.ego_aware_prediction.maneuver_recognition.werling_maneuver_classifier import \
    WerlingManeuverClassifier
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.state.state import State
from mapping.src.service.map_service import MapService
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec_params import ManeuverSpecParams

KEEP_LAT_D_VALUE = 10.0

class WerlingManeuverClassifierMock(WerlingManeuverClassifier):
    """
    Iterates over all possible parameters permutations every time the maneuver classifier is called.
    """

    def __init__(self, T_s: float, T_d_grid: np.ndarray, average_s_a_grid: np.ndarray, s_a_final_grid: np.ndarray,
                 relative_lane_grid: np.ndarray, lat_normalized_grid: np.ndarray):
        """
        :param T_s: Maneuver duration in s road coordinate [sec]
        :param T_d_grid: Possible maneuver duration in d road coordinate [sec]
        :param average_s_a_grid: Possible average acceleration throughout maneuver in [m/s^2]
        :param s_a_final_grid: Possible acceleration at maneuver end in [m/s^2]
        :param relative_lane_grid: Possible lateral position relative to current object's center lane in [lanes]
        :param lat_normalized_grid: Possible lateral position within lane in [lanes]
        """
        super().__init__(T_s=T_s)

        # Set grids of possible parameters values
        self._s_a_final_grid = s_a_final_grid
        self._relative_lane_grid = relative_lane_grid
        self._lat_normalized_grid = lat_normalized_grid
        self._average_s_a_grid = average_s_a_grid
        self._T_d_grid = T_d_grid

        # Set default values for mock's output parameters
        self._trajectory_class_t_d = T_d_grid[0]
        self._trajectory_class_avg_s_a = average_s_a_grid[0]
        self._trajectory_class_s_a_final = s_a_final_grid[0]
        self._trajectory_class_d_in_lanes = relative_lane_grid[0]
        self._trajectory_class_lat = lat_normalized_grid[0]

    def maneuver_params_options(self):
        """
        Iterates over all permutation of the trajectory specification grid
        :return: (T_d, average acceleration in s road coordinate, lateral position relative to object's center lane)
        """
        params_indx = 0
        for T_d in self._T_d_grid:
            for avg_s_a in self._average_s_a_grid:
                for s_a_final in self._s_a_final_grid:
                    for d_in_lanes in self._relative_lane_grid:
                        for lat in self._lat_normalized_grid:
                            # Set trajectory parameters to be used by classification module
                            self._trajectory_class_t_d = T_d
                            self._trajectory_class_avg_s_a = avg_s_a
                            self._trajectory_class_s_a_final = s_a_final
                            self._trajectory_class_d_in_lanes = d_in_lanes
                            self._trajectory_class_lat = lat

                            params_indx += 1
                            yield ManeuverSpecParams(T_d=T_d, avg_s_a=avg_s_a, s_a_final=s_a_final,
                                                     relative_lane=d_in_lanes, lat_normalized=lat)

    def classify_maneuver(self, state: State, object_id: int) -> ManeuverSpec:
        """
        Predicts the type of maneuver an object will execute
        Assuming zero acceleration in the initial state
        :param state: world state
        :param object_id: of predicted object
        :return: maneuver specification of an object
        """

        map_api = MapService().get_instance()

        # Fetch trajectory parameters
        t_d = self._trajectory_class_t_d
        avg_s_a = self._trajectory_class_avg_s_a
        s_a_final = self._trajectory_class_s_a_final
        d_in_lanes = self._trajectory_class_d_in_lanes
        lat = self._trajectory_class_lat

        object_state = State.get_object_from_state(state=state, target_obj_id=object_id)

        # Keep same normalized latitude in lane
        if lat == KEEP_LAT_D_VALUE:
            lane_width = map_api.get_road(road_id=object_state.road_localization.road_id).lane_width
            lat = (object_state.road_localization.intra_lane_lat - lane_width / 2.0) / lane_width

        return self._generate_maneuver_spec(object_state=object_state, T_d=t_d, avg_s_a=avg_s_a, s_a_final=s_a_final,
                                            d_in_lanes=d_in_lanes, lat=lat)
