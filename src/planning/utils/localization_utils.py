from decision_making.src.planning.types import Limits
from logging import Logger

from decision_making.src.global_constants import DEFAULT_OBJECT_Z_VALUE, NEGLIGIBLE_DISPOSITION_LAT, \
    NEGLIGIBLE_DISPOSITION_LON, EGO_ORIGIN_LON_FROM_REAR
from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.types import CartesianExtendedState, C_X, C_Y, C_YAW, FrenetPoint, FP_SX, FP_DX, C_V
from decision_making.src.state.state import EgoState

import numpy as np

from mapping.src.transformations.geometry_utils import CartesianFrame


class LocalizationUtils:
    @staticmethod
    def is_actual_state_close_to_expected_state(current_ego_state: EgoState,
                                                last_trajectory: SamplableTrajectory,
                                                logger: Logger, calling_class_name: str) -> bool:
        """
        checks if the actual ego state at time t[current] is close (currently in terms of Euclidean distance of position
        [x,y] only) to the desired state at t[current] according to the plan of the last trajectory.
        :param current_ego_state: the current EgoState object representing the actual state of ego vehicle
        :param last_trajectory: the trajectory object from the last plan (used to extract expected state)
        :param logger: the logger to use for logging the status of the test in this function
        :param calling_class_name: the name of the calling class (BP policy / TP facade)
        :return: true if actual state is closer than NEGLIGIBLE_DISPOSITION_* to the planned state. false otherwise
        """
        # TODO: update docstring
        current_time = current_ego_state.timestamp_in_sec
        if last_trajectory is None or current_time > last_trajectory.max_sample_time:
            return False

        logger.debug("%s time-difference from last planned trajectory is %s",
                     calling_class_name, current_time - last_trajectory.timestamp_in_sec)

        current_expected_state: CartesianExtendedState = last_trajectory.sample(np.array([current_time]))[0]
        current_actual_location = np.array([current_ego_state.x, current_ego_state.y, DEFAULT_OBJECT_Z_VALUE])

        errors_in_expected_frame, _ = CartesianFrame.convert_global_to_relative_frame(
            global_pos=current_actual_location,
            global_yaw=0.0,  # irrelevant since yaw isn't used.
            frame_position=np.append(current_expected_state[[C_X, C_Y]], [DEFAULT_OBJECT_Z_VALUE]),
            frame_orientation=current_expected_state[C_YAW]
        )

        distances_in_expected_frame: FrenetPoint = np.abs(errors_in_expected_frame)

        logger.debug(("is_actual_state_close_to_expected_state stats called from %s: "
                      "{desired_localization: %s, actual_localization: %s, desired_velocity: %s, "
                      "actual_velocity: %s, lon_lat_errors: %s, velocity_error: %s}" %
                      (calling_class_name, current_expected_state, current_actual_location, current_expected_state[C_V],
                       current_ego_state.v_x, distances_in_expected_frame,
                       current_ego_state.v_x - current_expected_state[C_V])).replace('\n', ' '))

        return distances_in_expected_frame[FP_SX] <= NEGLIGIBLE_DISPOSITION_LON and \
               distances_in_expected_frame[FP_DX] <= NEGLIGIBLE_DISPOSITION_LAT

    @staticmethod
    def transform_trajectory_between_ego_center_and_ego_origin(ego_length: float, trajectory: np.array,
                                                               direction: int) -> np.array:
        """
        Transform ego trajectory points from ego origin (e.g. front axle) to ego center or vice versa
        :param ego_length: the length of ego
        :param trajectory: trajectory points representing ego center
        :param direction: [-1 or 1] If direction=-1 then transform trajectory from ego origin to ego center
        (in case of origin in front axle, move the trajectory backward relatively to ego).
        If direction=1 then transform trajectory from ego center to ego origin.
        :return: transformed trajectory_points representing real ego origin
        """
        yaw_vec = trajectory[:, C_YAW]
        zero_vec = np.zeros(trajectory.shape[0])
        shift = direction * (EGO_ORIGIN_LON_FROM_REAR - ego_length/2)
        return trajectory + shift * np.c_[np.cos(yaw_vec), np.sin(yaw_vec), zero_vec, zero_vec, zero_vec, zero_vec]

    @staticmethod
    def transform_ego_from_origin_to_center(ego_state: EgoState) -> EgoState:
        """
        Transform ego state from ego origin to ego center
        :param ego_state: original ego state
        :return: transformed ego state
        """
        ego_pos = np.array([ego_state.x, ego_state.y, ego_state.yaw, 0, 0, 0])
        transformed_ego_pos = LocalizationUtils.transform_trajectory_between_ego_center_and_ego_origin(
            ego_state.size.length, np.array([ego_pos]), direction=-1)[0]
        # return cloned ego state with transformed position (since road_localization should be recomputed)
        cartesian_state = np.array([transformed_ego_pos[0], transformed_ego_pos[1], ego_state.yaw, ego_state.v_x,
                                    ego_state.acceleration_lon, 0])
        return ego_state.clone_cartesian_state(ego_state.timestamp_in_sec, cartesian_state)
