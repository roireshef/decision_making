import numpy as np
from typing import List

from decision_making.src.global_constants import WERLING_TIME_RESOLUTION
from decision_making.src.planning.types import FS_SV, FS_SX, \
    CartesianExtendedTrajectory, FrenetTrajectory2D
from decision_making.src.prediction.ego_aware_prediction.ended_maneuver_params import EndedManeuverParams
from decision_making.src.prediction.ego_aware_prediction.maneuver_spec import ManeuverSpec
from decision_making.src.planning.behavioral.state import MapState
from decision_making.src.planning.behavioral.state import DynamicObject, State
from decision_making.src.utils.map_utils import MapUtils


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

        lane_id = object_state.map_state.lane_id

        # Object's initial state in Frenet frame
        obj_init_fstate = object_state.map_state.lane_fstate

        # Calculate object's initial state in Frenet frame according to model
        lane_width = MapUtils.get_lane_width(lane_id, s=obj_init_fstate[FS_SX])

        s_x_final, s_v_final = PredictionUtils.compute_distance_from_average_acceleration(ended_maneuver_params.T_s,
                                                                                          ended_maneuver_params.avg_s_a,
                                                                                          obj_init_fstate[FS_SX],
                                                                                          obj_init_fstate[FS_SV])

        s_a_final = ended_maneuver_params.s_a_final
        d_x_final = lane_width * (ended_maneuver_params.relative_lane + ended_maneuver_params.lat_normalized)
        d_v_final = 0.0
        d_a_final = 0.0

        obj_final_fstate = np.array([s_x_final, s_v_final, s_a_final, d_x_final, d_v_final, d_a_final])

        return ManeuverSpec(init_state=obj_init_fstate, final_state=obj_final_fstate, T_s=ended_maneuver_params.T_s,
                            T_d=ended_maneuver_params.T_s)

    @staticmethod
    def compute_distance_from_average_acceleration(horizon: float, avg_a: float, init_x: float, init_v: float) -> (float, float):
        """
        Computes future physical x and velocity, using average acceleration
        :param horizon: the time horizon for future x computation
        :param avg_a: the average acceleration
        :param init_x: the initial x value
        :param init_v: the initial velocity value
        :return: future physical x and velocity
        """
        # Motion model (in Frenet frame)
        t_axis = np.arange(0.0, horizon + 10 * np.finfo(float).eps, WERLING_TIME_RESOLUTION)
        # Calculate velocity according to average acceleration
        s_v_vec = init_v + avg_a * t_axis
        # Clip negative velocities if starting velocity is positive
        if init_v > 0:
            s_v_vec = np.clip(s_v_vec, 0.0, np.inf)
        s_x_vec = init_x + np.cumsum(s_v_vec[1:] * WERLING_TIME_RESOLUTION)
        s_x_vec = np.r_[init_x, s_x_vec]

        return s_x_vec[-1], s_v_vec[-1]

    @staticmethod
    def convert_ctrajectory_to_dynamic_objects(dynamic_object: DynamicObject,
                                               predictions: CartesianExtendedTrajectory,
                                               prediction_timestamps: np.ndarray) -> List[DynamicObject]:
        """
        Given original dynamic object, its predictions, and their respective time stamps, creates a list of dynamic
         objects corresponding to the predicted object in those timestamps.
        :param dynamic_object: the original dynamic object
        :param predictions: the ctrajectory prediction of the dynamic object
        :param prediction_timestamps: the prediction timestamps
        :return:creates a list of dynamic objects corresponding to the predicted object ctrajectory in those timestamps.
        """

        predicted_object_states = [
            dynamic_object.clone_from_cartesian_state(timestamp_in_sec=prediction_timestamps[t_ind],
                                                      cartesian_state=predictions[t_ind]) for t_ind in
            range(len(prediction_timestamps))]

        return predicted_object_states

    @staticmethod
    def convert_ftrajectory_to_dynamic_objects(dynamic_object: DynamicObject,
                                               predictions: FrenetTrajectory2D,
                                               prediction_timestamps: np.ndarray) -> List[DynamicObject]:
        """
        Given original dynamic object, its predictions, and their respective time stamps, creates a list of dynamic
         objects corresponding to the predicted object in those timestamps.
        :param dynamic_object: the original dynamic object
        :param predictions: the ctrajectory prediction of the dynamic object
        :param prediction_timestamps: the prediction timestamps
        :return:creates a list of dynamic objects corresponding to the predicted object ctrajectory in those timestamps.
        """

        predicted_object_states = [
            dynamic_object.clone_from_map_state(timestamp_in_sec=prediction_timestamps[t_ind],
                                                map_state=MapState(lane_fstate=predictions[t_ind],
                                                                   lane_id=dynamic_object.map_state.lane_id)) for t_ind
            in
            range(len(prediction_timestamps))]

        return predicted_object_states

    @staticmethod
    def extract_most_recent_timestamp(state: State) -> float:
        """
        Returns state with all objects aligned to the most recent timestamp.
        Most recent timestamp is taken as the max between the current_timestamp, and the most recent
        timestamp of all objects in the scene.
        :param state: state containing objects with different timestamps
        :return: most_recent_timestamp:  most recent timestamp among all dynamic objects and ego
        """
        ego_timestamp_in_sec = state.ego_state.timestamp_in_sec
        objects_timestamp_in_sec = [state.dynamic_objects[x].timestamp_in_sec for x in range(len(state.dynamic_objects))]
        objects_timestamp_in_sec.append(ego_timestamp_in_sec)
        return np.max(objects_timestamp_in_sec)
