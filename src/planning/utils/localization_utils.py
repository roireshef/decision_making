from decision_making.src.planning.types import Limits
from logging import Logger

from decision_making.src.global_constants import DEFAULT_OBJECT_Z_VALUE, NEGLIGIBLE_DISPOSITION_LAT, \
    NEGLIGIBLE_DISPOSITION_LON
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
        :return: true if actual state is closer than NEGLIGIBLE_LOCATION_DIFF to the planned state. false otherwise
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
