import numpy as np
import pytest

from decision_making.src.global_constants import EGO_ORIGIN_LON_FROM_REAR
from decision_making.src.planning.utils.localization_utils import LocalizationUtils
from decision_making.src.state.state import EgoState, ObjectSize


def test_localizationUtils_transformEgoFromOriginToCenter():

    ego_pos = np.array([3, 4, 0])
    ego_yaw = np.pi/2
    ego_size = ObjectSize(5, 2, 2)
    ego_state = EgoState(obj_id=0, timestamp=0, x=ego_pos[0], y=ego_pos[1], z=ego_pos[2], yaw=ego_yaw,
                         size=ego_size, confidence=1.0, v_x=0.0, v_y=0.0, steering_angle=0.0,
                         acceleration_lon=0.0, omega_yaw=0.0)
    transformed_ego = LocalizationUtils.transform_ego_from_origin_to_center(ego_state)
    assert transformed_ego.x == ego_pos[0] and \
           transformed_ego.y == ego_pos[1] + (ego_size.length/2 - EGO_ORIGIN_LON_FROM_REAR)

    trajectory = np.array([np.array([transformed_ego.x, transformed_ego.y, ego_yaw, 0, 0, 0])])
    transformed_traj = LocalizationUtils.transform_trajectory_between_ego_center_and_ego_origin(ego_size.length,
                                                                                                trajectory, 1)[0]
    assert transformed_traj[0] == ego_pos[0] and transformed_traj[1] == ego_pos[1]
