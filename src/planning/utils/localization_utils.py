from logging import Logger

import numpy as np

from decision_making.src.global_constants import NEGLIGIBLE_DISPOSITION_LAT, NEGLIGIBLE_DISPOSITION_LON, \
    LOG_INVALID_TRAJECTORY_SAMPLING_TIME
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import CartesianExtendedState, C_X, C_Y, C_YAW, FrenetPoint, FP_SX, FP_DX, C_V, \
    C_A
from decision_making.src.state.state import EgoState
from decision_making.src.utils.geometry_utils import CartesianFrame
import rte.python.profiler as prof


class LocalizationUtils:
    @staticmethod
    @prof.ProfileFunction()
    # TODO: can we remove calling_class_name assuming we use the right class logger?
    def is_actual_state_close_to_expected_state(current_ego_state,
                                                last_trajectory,
                                                logger, calling_class_name):
        # type: (EgoState, SamplableTrajectory, Logger, str) -> bool
        """
        checks if the actual ego state at time t[current] is close (currently in terms of Euclidean distance of position
        [x,y] only) to the desired state at t[current] according to the plan of the last trajectory.
        :param current_ego_state: the current EgoState object representing the actual state of ego vehicle
        :param last_trajectory: the trajectory object from the last plan (used to extract expected state)
        :param logger: the logger to use for logging the status of the test in this function
        :param calling_class_name: the name of the calling class (BP policy / TP facade)
        :return: true if actual state is closer than NEGLIGIBLE_DISPOSITION_* to the planned state. false otherwise
        """
        current_time = current_ego_state.timestamp_in_sec
        if last_trajectory is None:
            return False

        if current_time < last_trajectory.timestamp_in_sec or current_time > last_trajectory.max_sample_time:
            logger.warning(LOG_INVALID_TRAJECTORY_SAMPLING_TIME, current_time, last_trajectory.timestamp_in_sec,
                           last_trajectory.max_sample_time)
            return False

        logger.debug("%s time-difference from last planned trajectory is %s",
                     calling_class_name, current_time - last_trajectory.timestamp_in_sec)

        current_expected_state = last_trajectory.sample(np.array([current_time]))[0]  # type: CartesianExtendedState
        current_actual_location = np.array([current_ego_state.x, current_ego_state.y])

        errors_in_expected_frame, _ = CartesianFrame.convert_global_to_relative_frame(
            global_pos=current_actual_location,
            global_yaw=0.0,  # irrelevant since yaw isn't used.
            frame_position=current_expected_state[[C_X, C_Y]],
            frame_orientation=current_expected_state[C_YAW]
        )

        distances_in_expected_frame = np.abs(errors_in_expected_frame)  # type: FrenetPoint

        logger.debug(("is_actual_state_close_to_expected_state stats called from %s: "
                      "{desired_localization: %s, actual_localization(x,y): %s, desired_velocity: %s, "
                      "actual_velocity: %s, lon_lat_errors: %s, velocity_error: %s, acceleration: %s"
                      ", timestamp is: %f}" %
                      (calling_class_name, current_expected_state, current_actual_location, current_expected_state[C_V],
                       current_ego_state.velocity, distances_in_expected_frame,
                       current_ego_state.velocity - current_expected_state[C_V],
                       current_ego_state.acceleration - current_expected_state[C_A],
                       current_ego_state.timestamp_in_sec)).replace('\n', ' '))

        return distances_in_expected_frame[FP_SX] <= NEGLIGIBLE_DISPOSITION_LON and \
               distances_in_expected_frame[FP_DX] <= NEGLIGIBLE_DISPOSITION_LAT
