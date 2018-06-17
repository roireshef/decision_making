from decision_making.src.global_constants import EGO_ORIGIN_LON_FROM_CENTER
from decision_making.src.planning.types import C_YAW
import numpy as np

from decision_making.src.state.state import NewEgoState


class Transformations:
    @staticmethod
    def transform_trajectory_between_ego_center_and_ego_origin(trajectory, direction):
        # type: (np.array, int) -> np.array
        """
        Transform ego trajectory points from ego origin (e.g. front axle) to ego center or vice versa
        :param trajectory: trajectory points representing ego center
        :param direction: [-1 or 1] If direction=-1 then transform trajectory from ego origin to ego center
        (in case of origin in front axle, move the trajectory backward relatively to ego).
        If direction=1 then transform trajectory from ego center to ego origin.
        :return: transformed trajectory_points representing real ego origin
        """
        yaw_vec = trajectory[:, C_YAW]
        zero_vec = np.zeros(trajectory.shape[0])
        shift = direction * EGO_ORIGIN_LON_FROM_CENTER
        return trajectory + shift * np.c_[np.cos(yaw_vec), np.sin(yaw_vec), zero_vec, zero_vec, zero_vec, zero_vec]

    @staticmethod
    def transform_ego_from_origin_to_center(ego_state):
        # type: (NewEgoState) -> NewEgoState
        """
        Transform ego state from ego origin to ego center
        :param ego_state: original ego state
        :return: transformed ego state
        """
        ego_pos = np.array([ego_state.x, ego_state.y, ego_state.yaw, 0, 0, 0])
        transformed_ego_pos = Transformations.transform_trajectory_between_ego_center_and_ego_origin(
            trajectory=np.array([ego_pos]), direction=-1)[0]
        # return cloned ego state with transformed position (since road_localization should be recomputed)
        cartesian_state = np.array([transformed_ego_pos[0], transformed_ego_pos[1], ego_state.yaw, ego_state.velocity,
                                    ego_state.acceleration, 0])
        return ego_state.clone_from_cartesian_state(cartesian_state, ego_state.timestamp_in_sec)
